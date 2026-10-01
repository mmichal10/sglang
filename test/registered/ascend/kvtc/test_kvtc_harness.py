"""Standalone, real-NPU KVTC smoke tests. Edit DEFAULT_CONFIG, then run this file.

No server, model weights, or KV dumps are loaded. The model and calibration
artifact determine the local KV shape during fixture initialization.
"""

from __future__ import annotations

import atexit
import gc
import re
import sys
import tempfile
import time
import unittest
from array import array
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))


@dataclass(frozen=True)
class HarnessConfig:
    model_dir: Path = Path("/path/to/local/model")
    artifact_path: Path = Path("/path/to/kvtc-calibration.pt")
    worker_name: str = "tp_0_pp_0"
    dtype_name: str = "bfloat16"
    page_size: int = 128
    device_tokens: int = 512
    k_cr: int = 8
    v_cr: int = 8
    quant_disable: bool = False
    device_index: int = 0
    host_size_gb: int = 10


DEFAULT_CONFIG = HarnessConfig()


@dataclass(frozen=True)
class LocalKVShape:
    layers: int
    heads: int
    head_dim: int


class FixtureFactory:
    """Keep SGLang constructor and transfer details here for rebases."""

    @staticmethod
    @lru_cache(maxsize=2)
    def _model_fixture(
        model_dir: Path,
        artifact_path: Path,
        worker_name: str,
        k_cr: int,
        v_cr: int,
        quant_disable: bool,
        page_size: int,
        device_index: int,
    ):
        try:
            import torch
            import torch_npu  # noqa: F401 - registers torch.npu
        except ImportError as exc:
            raise RuntimeError(
                "KVTC harness needs the active PyTorch/torch_npu environment"
            ) from exc
        if not torch.npu.is_available():
            raise RuntimeError("KVTC harness needs an available Ascend NPU")
        if not model_dir.is_dir():
            raise FileNotFoundError(
                "Set DEFAULT_CONFIG.model_dir to a local model config directory: "
                f"{model_dir}"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(
                "Set DEFAULT_CONFIG.artifact_path to an existing calibration "
                f"artifact: {artifact_path}"
            )
        worker = re.fullmatch(r"tp_(\d+)_pp_(\d+)", worker_name)
        if worker is None:
            raise ValueError(f"Invalid worker_name: {worker_name!r}")
        for side, ratio in (("K", k_cr), ("V", v_cr)):
            if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 0:
                raise ValueError(
                    f"{side} compression ratio must be a nonnegative integer"
                )
        worker_rank = tuple(map(int, worker.groups()))
        torch.npu.set_device(device_index)

        from transformers import AutoConfig

        from scripts.kvtc_calibration_data import Rope
        from sglang.srt.distributed.utils import get_pp_indices
        from sglang.srt.mem_cache.kvtc_quant import KVTCQuantizer
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        model = AutoConfig.from_pretrained(
            str(model_dir), trust_remote_code=False, local_files_only=True
        )
        model = getattr(model, "text_config", None) or model
        num_layers = int(model.num_hidden_layers)
        num_attention_heads = int(model.num_attention_heads)
        kv_heads = getattr(model, "num_key_value_heads", None)
        num_kv_heads = int(num_attention_heads if kv_heads is None else kv_heads)
        if min(num_layers, num_attention_heads, num_kv_heads) <= 0:
            raise ValueError("Model KV dimensions must be positive")
        head_dim = getattr(model, "head_dim", None)
        if head_dim is None:
            if model.hidden_size % num_attention_heads:
                raise ValueError(
                    "Model hidden size is not divisible by attention heads"
                )
            head_dim = model.hidden_size // num_attention_heads
        head_dim = int(head_dim)
        if head_dim <= 0:
            raise ValueError("Model KV head dimension must be positive")

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict):
            raise ValueError("KVTC calibration artifact must contain a dictionary")
        worker_sets = []
        worker_sides = [
            side for side, ratio in (("keys", k_cr), ("values", v_cr)) if ratio > 0
        ]
        if not worker_sides:
            worker_sides = [side for side in ("keys", "values") if side in artifact]
        for side in worker_sides:
            entries = artifact.get(side)
            if not isinstance(entries, dict) or worker_name not in entries:
                raise ValueError(f"Artifact is missing {side}/{worker_name}")
            workers = set(entries)
            if not all(
                isinstance(name, str) and re.fullmatch(r"tp_\d+_pp_\d+", name)
                for name in workers
            ):
                raise ValueError(f"Artifact {side} has invalid worker names")
            worker_sets.append(workers)
        if not worker_sets or any(workers != worker_sets[0] for workers in worker_sets):
            raise ValueError("Artifact K/V worker sets must match")
        workers = worker_sets[0]
        ranks = {
            tuple(map(int, re.fullmatch(r"tp_(\d+)_pp_(\d+)", name).groups()))
            for name in workers
        }
        tp_size = max(rank[0] for rank in ranks) + 1
        pp_size = max(rank[1] for rank in ranks) + 1
        if ranks != {(tp, pp) for tp in range(tp_size) for pp in range(pp_size)}:
            raise ValueError("Artifact workers must form a complete TP/PP grid")
        if num_layers < pp_size:
            raise ValueError("Model has fewer layers than artifact PP workers")
        if num_kv_heads >= tp_size:
            if num_kv_heads % tp_size:
                raise ValueError("Model KV heads are not divisible by artifact TP size")
        elif tp_size % num_kv_heads:
            raise ValueError("Artifact TP size cannot replicate model KV heads evenly")
        start_layer, end_layer = get_pp_indices(num_layers, worker_rank[1], pp_size)
        shape = LocalKVShape(
            layers=end_layer - start_layer,
            heads=max(1, num_kv_heads // tp_size),
            head_dim=head_dim,
        )
        if shape.layers <= 0:
            raise ValueError("Selected PP worker has no model layers")
        features = shape.layers * shape.heads * shape.head_dim
        for side, ratio in (("keys", k_cr), ("values", v_cr)):
            if ratio <= 0:
                continue
            entry = artifact[side][worker_name]
            if not isinstance(entry, dict):
                raise ValueError(f"Artifact {side}/{worker_name} is invalid")
            mu, basis = entry.get("mu"), entry.get("basis")
            if (
                not isinstance(mu, torch.Tensor)
                or mu.shape != (features,)
                or mu.dtype != torch.float32
                or not isinstance(basis, torch.Tensor)
                or basis.ndim != 2
                or basis.shape[0] != features
                or basis.shape[1] == 0
                or basis.dtype != torch.float32
            ):
                raise ValueError(
                    f"Artifact {side}/{worker_name} must have FP32 mu [{features}] "
                    f"and basis [{features}, rank>0] for local shape {shape}"
                )
            if quant_disable:
                retained_rank = features // ratio
                if retained_rank == 0:
                    raise ValueError(
                        f"{side} compression ratio {ratio} retains no features"
                    )
                if basis.shape[1] < retained_rank:
                    raise ValueError(
                        f"Artifact {side} basis is too short for ratio {ratio}"
                    )
            else:
                quant = entry.get("quant")
                if not isinstance(quant, dict):
                    raise ValueError(
                        f"Artifact {side}/{worker_name} has no quant schemas"
                    )
                KVTCQuantizer.build_layout(
                    quant.get(str(ratio)),
                    page_size=page_size,
                    basis_rank=basis.shape[1],
                    matrix_name=side,
                )

        # The dummy server arguments avoid server/model initialization. RoPE reads
        # only config.json through AutoConfig and constructs its position cache.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        Rope.load_model_config(str(model_dir))
        rotary = Rope.rotary_emb.to(device="npu")
        if rotary.rotary_dim != shape.head_dim or not rotary.is_neox_style:
            raise ValueError("KVTC harness requires full-head, NeoX-style 1-D RoPE")
        if rotary.cos_sin_cache.shape[0] < 256:
            raise ValueError("Model RoPE cache must cover 256 token positions")
        return torch, rotary, worker_rank, shape

    @staticmethod
    def prepare(config: HarnessConfig):
        if (
            config.page_size != 128
            or config.device_tokens < 256
            or config.device_tokens % config.page_size
        ):
            raise ValueError(
                "This harness uses 128-token pages and a page-aligned device "
                "capacity of at least 256 tokens"
            )
        if config.dtype_name not in ("float16", "bfloat16"):
            raise ValueError("dtype_name must be 'float16' or 'bfloat16'")
        if config.host_size_gb != 10:
            raise ValueError("This harness uses the production 10 GB hybrid pool")
        return FixtureFactory._model_fixture(
            config.model_dir,
            config.artifact_path,
            config.worker_name,
            config.k_cr,
            config.v_cr,
            config.quant_disable,
            config.page_size,
            config.device_index,
        )

    @staticmethod
    def device_pool(config: HarnessConfig, shape: LocalKVShape, torch):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

        pool = NPUMHATokenToKVPool(
            size=config.device_tokens,
            page_size=config.page_size,
            dtype=getattr(torch, config.dtype_name),
            head_num=shape.heads,
            head_dim=shape.head_dim,
            layer_num=shape.layers,
            device="npu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        if not isinstance(pool.k_buffer, torch.Tensor) or pool.k_buffer.ndim != 5:
            raise ValueError(
                "KVTC harness requires the paged NPU layout; unset ASCEND_USE_FIA"
            )
        return pool

    @staticmethod
    def hybrid_pool(config: HarnessConfig, device_pool, rotary, worker):
        from sglang.srt.mem_cache.memory_pool_host import NPUMHATokenToKVPoolHybrid

        return NPUMHATokenToKVPoolHybrid(
            device_pool=device_pool,
            host_to_device_ratio=1,
            host_size=config.host_size_gb,
            page_size=config.page_size,
            layout="page_first_direct",
            kvtc_params_path=str(config.artifact_path),
            kvtc_k_compression_ratio=config.k_cr,
            kvtc_v_compression_ratio=config.v_cr,
            kvtc_quant_disable=config.quant_disable,
            rotary_emb=rotary,
            tp_rank=worker[0],
            pp_rank=worker[1],
        )

    @staticmethod
    def request(
        device_pool,
        *,
        sink_host,
        compressed_host,
        sink_device,
        compressed_device,
        compressed_tokens,
    ):
        from sglang.srt.mem_cache.memory_pool_host import KVTCHostMemoryRequest

        return KVTCHostMemoryRequest(
            device_memory_pool=device_pool,
            host_indices_compressed=compressed_host,
            device_indices_compressed=compressed_device,
            token_indices_compressed=compressed_tokens,
            host_indices_sink=sink_host,
            device_indices_sink=sink_device,
            io_backend="kernel_ascend",
            layer_id=None,
        )

    @staticmethod
    def reload(pool, request, torch):
        for layer in range(request.device_memory_pool.layer_num):
            request.layer_id = layer
            pool.load_to_device_per_layer(request)
        torch.npu.synchronize()

    @staticmethod
    def pca_reference(original, positions, entry, ratio, rotary, is_key, torch):
        layers, tokens, heads, head_dim = original.shape
        features = layers * heads * head_dim
        mu = entry["mu"].to("npu")
        basis = entry["basis"][:, : features // ratio].to("npu")
        values = original.transpose(0, 1).contiguous()
        if is_key:
            values = rotary.invert_native_keys_batch(positions, values)
        projection = (values.reshape(tokens, features) - mu) @ basis
        projection = projection.to(original.dtype).to(basis.dtype)
        reconstructed = (projection @ basis.T + mu).reshape(
            tokens, layers, heads, head_dim
        )
        if is_key:
            reconstructed = rotary.forward_native_keys_batch(
                positions, reconstructed
            )
        return reconstructed.transpose(0, 1).to(original.dtype)

    @staticmethod
    def fill_distinct_pages(
        device_pool, indices, positions, compressed_pool, rotary, torch
    ):
        page_size = compressed_pool.page_size
        page_count = len(indices) // page_size
        coefficients = (
            torch.arange(page_count, device="npu", dtype=torch.float32)
            .repeat_interleave(page_size)
            .mul(4)
            + torch.arange(page_size, device="npu", dtype=torch.float32)
            .repeat(page_count)
            .div(page_size)
        )
        for matrix, buffer in (
            ("k", device_pool.k_buffer),
            ("v", device_pool.v_buffer),
        ):
            mu = getattr(compressed_pool, f"kvtc_{matrix}_mu")
            basis = getattr(compressed_pool, f"kvtc_{matrix}_V")[:, 0]
            values = (mu[None, :] + coefficients[:, None] * basis[None, :]).reshape(
                len(indices),
                device_pool.layer_num,
                device_pool.head_num,
                device_pool.head_dim,
            )
            if matrix == "k":
                values = rotary.forward_native_keys_batch(positions, values)
            buffer.flatten(1, 2).index_copy_(
                1, indices.to("npu"), values.transpose(0, 1).to(buffer.dtype)
            )
        torch.npu.synchronize()

    @staticmethod
    def page_basis_signatures(
        values, positions, compressed_pool, rotary, matrix
    ):
        tokens_first = values.transpose(0, 1).contiguous()
        if matrix == "k":
            tokens_first = rotary.invert_native_keys_batch(positions, tokens_first)
        mu = getattr(compressed_pool, f"kvtc_{matrix}_mu")
        basis = getattr(compressed_pool, f"kvtc_{matrix}_V")[:, 0]
        coefficients = (tokens_first.reshape(len(positions), -1).float() - mu) @ basis
        return coefficients.reshape(-1, compressed_pool.page_size).mean(dim=1)

    @staticmethod
    def tree(config: HarnessConfig, device_pool, rotary, torch):
        from sglang.srt.hardware_backend.npu.allocator_npu import (
            NPUPagedTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        args = ServerArgs(
            model_path="dummy",
            page_size=config.page_size,
            hicache_size=config.host_size_gb,
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="page_first_direct",
            hicache_write_policy="write_back",
            hicache_kvtc_params=str(config.artifact_path),
            hicache_kvtc_k_cr=config.k_cr,
            hicache_kvtc_v_cr=config.v_cr,
            hicache_kvtc_quant_disable=config.quant_disable,
        )
        set_global_server_args_for_scheduler(args)
        allocator = NPUPagedTokenToKVPoolAllocator(
            size=config.device_tokens,
            page_size=config.page_size,
            dtype=getattr(torch, config.dtype_name),
            device="npu",
            kvcache=device_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(1, 256, "npu", False),
            token_to_kv_pool_allocator=allocator,
            page_size=config.page_size,
            tp_cache_group=torch.distributed.group.WORLD,
            rotary_embeddings=rotary,
        )
        return HiRadixCache(params, args), allocator

    @staticmethod
    def fill(device_pool, indices, torch):
        """Write deterministic, finite synthetic K/V and return exact snapshots."""
        snapshots = []
        for offset, buffer in enumerate((device_pool.k_buffer, device_pool.v_buffer)):
            view = buffer.flatten(1, 2)
            count = (
                device_pool.layer_num
                * len(indices)
                * device_pool.head_num
                * device_pool.head_dim
            )
            values = torch.arange(
                count,
                device="npu", dtype=torch.float32,
            ).reshape(
                device_pool.layer_num,
                len(indices),
                device_pool.head_num,
                device_pool.head_dim,
            )
            values = (values.remainder(101) / 101 + offset).to(buffer.dtype)
            view.index_copy_(1, indices.to("npu"), values)
            snapshots.append(values.clone())
        torch.npu.synchronize()
        return snapshots

    @staticmethod
    def read(device_pool, indices):
        return [
            buffer.flatten(1, 2).index_select(1, indices.to("npu"))
            for buffer in (device_pool.k_buffer, device_pool.v_buffer)
        ]


class TimedTestCase(unittest.TestCase):
    def setUp(self):
        self._body_started = time.perf_counter()

    def tearDown(self):
        elapsed = time.perf_counter() - self._body_started
        print(f"{self.id()} body: {elapsed:.3f}s", file=sys.stderr)


class Test01HybridPool(TimedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DEFAULT_CONFIG
        cls.torch, rotary, worker, shape = FixtureFactory.prepare(cls.config)
        cls.device_pool = FixtureFactory.device_pool(cls.config, shape, cls.torch)
        cls.pool = FixtureFactory.hybrid_pool(
            cls.config, cls.device_pool, rotary, worker
        )

    @classmethod
    def tearDownClass(cls):
        cls.torch.npu.synchronize()
        del cls.pool, cls.device_pool
        gc.collect()
        cls.torch.npu.empty_cache()

    def test_01_initialization_allocation_and_reuse(self):
        before = (
            self.pool.sink_pool.available_size(),
            self.pool.compressed_pool.available_size(),
        )
        sink, compressed = self.pool.alloc(128, self.config.page_size)
        self.assertIsNotNone(sink)
        self.assertIsNotNone(compressed)
        self.assertTrue(self.pool.compressed_pool.k_kvtc)
        self.assertTrue(self.pool.compressed_pool.v_kvtc)
        self.assertFalse(self.pool.compressed_pool.kvtc_quant_disable)
        self.assertEqual(len(sink), 128)
        self.assertEqual(len(compressed), self.config.page_size)
        self.assertTrue(bool((sink >= self.pool.sink_token_shift).all()))
        self.assertTrue(bool((compressed < self.pool.sink_token_shift).all()))
        self.pool.free(self.torch.cat((sink, compressed)))
        self.assertEqual(
            (
                self.pool.sink_pool.available_size(),
                self.pool.compressed_pool.available_size(),
            ),
            before,
        )
        reused_sink, reused_compressed = self.pool.alloc(128, self.config.page_size)
        self.assertIsNotNone(reused_sink)
        self.assertIsNotNone(reused_compressed)
        self.assertEqual(len(reused_sink), 128)
        self.assertEqual(len(reused_compressed), self.config.page_size)
        self.pool.free(self.torch.cat((reused_sink, reused_compressed)))
        self.assertEqual(
            (
                self.pool.sink_pool.available_size(),
                self.pool.compressed_pool.available_size(),
            ),
            before,
        )
        self.torch.npu.synchronize()

    def test_02_mixed_offload_reload(self):
        torch = self.torch
        sink, compressed = self.pool.alloc(128, self.config.page_size)
        self.assertIsNotNone(sink)
        self.assertIsNotNone(compressed)
        indices = torch.arange(
            self.config.page_size,
            self.config.page_size + 256,
            dtype=torch.int64,
        )
        original = FixtureFactory.fill(self.device_pool, indices, torch)
        request = FixtureFactory.request(
            self.device_pool,
            sink_host=sink,
            compressed_host=compressed,
            sink_device=indices[:128],
            compressed_device=indices[128:],
            compressed_tokens=torch.arange(128, 256, device="npu", dtype=torch.int64),
        )
        try:
            self.pool.backup_from_device_all_layer(request)
            torch.npu.synchronize()
            for buffer in (self.device_pool.k_buffer, self.device_pool.v_buffer):
                buffer.flatten(1, 2).index_fill_(1, indices.to("npu"), float("nan"))
            torch.npu.synchronize()
            FixtureFactory.reload(self.pool, request, torch)
            for actual, expected in zip(
                FixtureFactory.read(self.device_pool, indices), original
            ):
                self.assertTrue(torch.equal(actual[:, :128], expected[:, :128]))
                self.assertTrue(bool(torch.isfinite(actual[:, 128:]).all()))
        finally:
            torch.npu.synchronize()
            self.pool.free(torch.cat((sink, compressed)))

    def test_03_quantized_groups_are_offloaded(self):
        from sglang.srt.mem_cache.kvtc_quant import KVTCQuantizer

        torch = self.torch
        compressed_pool = self.pool.compressed_pool
        self.assertIsInstance(compressed_pool.quantizer, KVTCQuantizer)
        matrices = (
            (
                "K",
                compressed_pool.quantizer.key_layout(),
                compressed_pool.k_quant_scales,
            ),
            (
                "V",
                compressed_pool.quantizer.value_layout(),
                compressed_pool.v_quant_scales,
            ),
        )
        # Require both integer storage types so this test covers either being skipped.
        quantized_groups = [
            (matrix, scales, group)
            for matrix, layout, scales in matrices
            for groups in layout.integer_quant_groups.values()
            for group in groups
        ]
        self.assertEqual(
            {group.dtype_name for _, _, group in quantized_groups},
            {"int4", "int8"},
            "The harness artifact must contain both INT4 and INT8 groups",
        )

        sink, compressed = self.pool.alloc(0, self.config.page_size)
        self.assertIsNotNone(sink)
        self.assertIsNotNone(compressed)
        try:
            host_page = int(compressed[0]) // self.config.page_size
            device_indices = torch.arange(
                self.config.page_size, 2 * self.config.page_size, dtype=torch.int64
            )
            FixtureFactory.fill(self.device_pool, device_indices, torch)
            # Mark every quantized group's host metadata before offloading.
            for _, scales, group in quantized_groups:
                scales[host_page, :, group.metadata_index].fill_(float("nan"))

            request = FixtureFactory.request(
                self.device_pool,
                sink_host=sink,
                compressed_host=compressed,
                sink_device=device_indices[:0],
                compressed_device=device_indices,
                compressed_tokens=device_indices.to("npu"),
            )
            self.pool.backup_from_device_all_layer(request)
            torch.npu.synchronize()

            # Each group must replace its marker with finite quantization scales.
            for matrix, scales, group in quantized_groups:
                self.assertTrue(
                    bool(
                        torch.isfinite(scales[host_page, :, group.metadata_index]).all()
                    ),
                    f"{matrix} {group.dtype_name} group at feature "
                    f"{group.feature_start} was not offloaded",
                )
        finally:
            torch.npu.synchronize()
            self.pool.free(compressed)


class Test02TransferModes(TimedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DEFAULT_CONFIG
        cls.torch, cls.rotary, cls.worker, shape = FixtureFactory.prepare(cls.config)
        cls.device_pool = FixtureFactory.device_pool(cls.config, shape, cls.torch)

    @classmethod
    def tearDownClass(cls):
        cls.torch.npu.synchronize()
        del cls.device_pool
        gc.collect()
        cls.torch.npu.empty_cache()

    def test_01_k_only_v_only_pca_only_and_baseline(self):
        torch = self.torch
        page_size = self.config.page_size
        indices = torch.arange(page_size, 3 * page_size, dtype=torch.int64)
        positions = torch.arange(
            page_size, 2 * page_size, device="npu", dtype=torch.int64
        )
        artifact = torch.load(
            self.config.artifact_path, map_location="cpu", weights_only=True
        )
        modes = (
            ("K only", replace(self.config, v_cr=0), True, False),
            ("V only", replace(self.config, k_cr=0), False, True),
            ("PCA only", replace(self.config, quant_disable=True), True, True),
            ("baseline", replace(self.config, k_cr=0, v_cr=0), False, False),
        )
        for name, config, compress_k, compress_v in modes:
            with self.subTest(mode=name):
                FixtureFactory.prepare(config)
                pool = FixtureFactory.hybrid_pool(
                    config, self.device_pool, self.rotary, self.worker
                )
                sink = compressed = None
                try:
                    self.assertEqual(pool.compressed_pool.k_kvtc, compress_k)
                    self.assertEqual(pool.compressed_pool.v_kvtc, compress_v)
                    self.assertEqual(
                        pool.compressed_pool.kvtc_quant_disable,
                        config.quant_disable,
                    )
                    sink, compressed = pool.alloc(page_size, page_size)
                    self.assertIsNotNone(sink)
                    self.assertIsNotNone(compressed)
                    original = FixtureFactory.fill(self.device_pool, indices, torch)
                    request = FixtureFactory.request(
                        self.device_pool,
                        sink_host=sink,
                        compressed_host=compressed,
                        sink_device=indices[:page_size],
                        compressed_device=indices[page_size:],
                        compressed_tokens=positions,
                    )
                    pool.backup_from_device_all_layer(request)
                    torch.npu.synchronize()
                    for buffer in (
                        self.device_pool.k_buffer,
                        self.device_pool.v_buffer,
                    ):
                        buffer.flatten(1, 2).index_fill_(
                            1, indices.to("npu"), float("nan")
                        )
                    torch.npu.synchronize()
                    FixtureFactory.reload(pool, request, torch)
                    for is_key, compressed_side, expected, actual in zip(
                        (True, False),
                        (compress_k, compress_v),
                        original,
                        FixtureFactory.read(self.device_pool, indices),
                    ):
                        self.assertTrue(
                            torch.equal(actual[:, :page_size], expected[:, :page_size])
                        )
                        if not compressed_side:
                            self.assertTrue(
                                torch.equal(
                                    actual[:, page_size:], expected[:, page_size:]
                                )
                            )
                        elif config.quant_disable:
                            side = "keys" if is_key else "values"
                            ratio = config.k_cr if is_key else config.v_cr
                            reference = FixtureFactory.pca_reference(
                                expected[:, page_size:],
                                positions,
                                artifact[side][config.worker_name],
                                ratio,
                                self.rotary,
                                is_key,
                                torch,
                            )
                            torch.testing.assert_close(
                                actual[:, page_size:], reference, rtol=0.05, atol=0.05
                            )
                        else:
                            self.assertTrue(
                                bool(torch.isfinite(actual[:, page_size:]).all())
                            )
                finally:
                    torch.npu.synchronize()
                    if sink is not None and compressed is not None:
                        pool.free(torch.cat((sink, compressed)))
                    del pool
                    gc.collect()


class Test03HiRadixCacheWorkflow(TimedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = replace(
            DEFAULT_CONFIG,
            device_tokens=max(DEFAULT_CONFIG.device_tokens, 512),
        )
        if cls.config.worker_name != "tp_0_pp_0":
            raise ValueError(
                "The single-process tree fixture requires worker_name='tp_0_pp_0'"
            )
        cls.torch, rotary, _, shape = FixtureFactory.prepare(cls.config)
        cls.owns_group = False
        cls.group_dir = None
        if not cls.torch.distributed.is_initialized():
            cls.group_dir = tempfile.TemporaryDirectory(prefix="kvtc-harness-pg-")
            init_file = Path(cls.group_dir.name) / "init"
            cls.torch.distributed.init_process_group(
                backend="gloo", init_method=init_file.as_uri(), rank=0, world_size=1
            )
            cls.owns_group = True
        cls.device_pool = FixtureFactory.device_pool(cls.config, shape, cls.torch)
        cls.cache, cls.allocator = FixtureFactory.tree(
            cls.config, cls.device_pool, rotary, cls.torch
        )

    @classmethod
    def tearDownClass(cls):
        cls.torch.npu.synchronize()
        cls.cache.shutdown()
        atexit.unregister(cls.cache.shutdown)
        del cls.cache, cls.allocator, cls.device_pool
        gc.collect()
        cls.torch.npu.empty_cache()
        if cls.owns_group:
            cls.torch.distributed.destroy_process_group()
            cls.group_dir.cleanup()

    def test_01_sink_only_and_compressed_only_routing(self):
        torch = self.torch
        controller = self.cache.cache_controller
        host_pool = self.cache.token_to_kv_pool_host
        page_size = self.config.page_size
        for name, first_position, is_sink in (
            ("sink only", 0, True),
            ("compressed only", page_size, False),
        ):
            with self.subTest(route=name):
                before_host = (
                    host_pool.sink_pool.available_size(),
                    host_pool.compressed_pool.available_size(),
                )
                before_device = len(self.allocator.free_pages)
                source = self.allocator.alloc(page_size)
                self.assertIsNotNone(source)
                loaded = host_indices = None
                source_released = False
                try:
                    positions = torch.arange(
                        first_position,
                        first_position + page_size,
                        device="npu",
                        dtype=torch.int64,
                    )
                    original = FixtureFactory.fill(self.device_pool, source, torch)
                    host_indices = controller.write(source, positions)
                    self.assertIsNotNone(host_indices)
                    controller.ack_write_queue.pop(0).finish_event.synchronize()
                    if is_sink:
                        self.assertTrue(
                            bool((host_indices >= host_pool.sink_token_shift).all())
                        )
                    else:
                        self.assertTrue(
                            bool((host_indices < host_pool.sink_token_shift).all())
                        )
                    self.allocator.free(source)
                    source_released = True
                    for buffer in (
                        self.device_pool.k_buffer,
                        self.device_pool.v_buffer,
                    ):
                        buffer.flatten(1, 2).index_fill_(
                            1, source.to("npu"), float("nan")
                        )
                    torch.npu.synchronize()
                    loaded = controller.load(host_indices, positions)
                    self.assertIsNotNone(loaded)
                    producer = controller.start_loading()
                    self.assertGreaterEqual(producer, 0)
                    controller.layer_done_counter.events[
                        producer
                    ].finish_event.synchronize()
                    controller.ack_load_queue.pop(0).finish_event.synchronize()
                    for actual, expected in zip(
                        FixtureFactory.read(self.device_pool, loaded), original
                    ):
                        if is_sink:
                            self.assertTrue(torch.equal(actual, expected))
                        else:
                            self.assertTrue(bool(torch.isfinite(actual).all()))
                finally:
                    torch.npu.synchronize()
                    if loaded is not None:
                        controller.evict_device(loaded)
                    if not source_released:
                        self.allocator.free(source)
                    if host_indices is not None:
                        controller.evict_host(host_indices)
                self.assertEqual(len(self.allocator.free_pages), before_device)
                self.assertEqual(
                    (
                        host_pool.sink_pool.available_size(),
                        host_pool.compressed_pool.available_size(),
                    ),
                    before_host,
                )

    def test_03_insert_backup_evict_match_reload(self):
        from sglang.srt.managers.cache_controller import KVTCHiCacheController
        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.memory_pool_host import NPUMHATokenToKVPoolHybrid
        from sglang.srt.mem_cache.radix_cache import RadixKey

        torch = self.torch
        cache = self.cache
        self.assertIsInstance(cache.token_to_kv_pool_host, NPUMHATokenToKVPoolHybrid)
        self.assertIsInstance(cache.cache_controller, KVTCHiCacheController)
        self.assertTrue(cache.token_to_kv_pool_host.compressed_pool.k_kvtc)
        self.assertTrue(cache.token_to_kv_pool_host.compressed_pool.v_kvtc)
        self.assertFalse(
            cache.token_to_kv_pool_host.compressed_pool.kvtc_quant_disable
        )
        indices = self.allocator.alloc(256)
        self.assertIsNotNone(indices)
        original = FixtureFactory.fill(self.device_pool, indices, torch)
        key = RadixKey(array("q", range(256)))
        cache.insert(InsertParams(key=key, value=indices))
        match = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(match.device_indices), 256)
        node = match.last_device_node
        self.assertEqual(cache.write_backup(node, write_back=True), 256)
        cache.writing_check(write_back=True)
        torch.npu.synchronize()
        self.assertEqual(
            cache.evict(EvictParams(num_tokens=256)).num_tokens_evicted, 256
        )
        for buffer in (self.device_pool.k_buffer, self.device_pool.v_buffer):
            buffer[:, 1:].fill_(float("nan"))
        torch.npu.synchronize()
        match = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(match.host_hit_length, 256)
        loaded = cache.load_back(match.last_host_node)
        self.assertIsNotNone(loaded)
        consumer = cache.ready_to_load_host_cache()
        self.assertGreaterEqual(consumer, 0)
        finish_event = cache.cache_controller.layer_done_counter.events[
            consumer
        ].finish_event
        finish_event.synchronize()
        cache.loading_check()
        torch.npu.synchronize()
        recovered = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(recovered.device_indices), 256)
        self.assertTrue(torch.equal(recovered.device_indices, loaded))
        for actual, expected in zip(
            FixtureFactory.read(self.device_pool, loaded), original
        ):
            self.assertTrue(torch.equal(actual[:, :128], expected[:, :128]))
            self.assertTrue(bool(torch.isfinite(actual[:, 128:]).all()))


class Test04QuantBatchOrdering(TimedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.rotary, worker, shape = FixtureFactory.prepare(DEFAULT_CONFIG)
        from sglang.srt.mem_cache.memory_pool_host import NPUMHATokenToKVPoolCompressed

        cls.page_count = NPUMHATokenToKVPoolCompressed._QUANT_BATCH_MAX_PAGES + 1
        cls.config = replace(
            DEFAULT_CONFIG,
            device_tokens=max(
                DEFAULT_CONFIG.device_tokens,
                (cls.page_count + 1) * DEFAULT_CONFIG.page_size,
            ),
        )
        FixtureFactory.prepare(cls.config)
        cls.device_pool = FixtureFactory.device_pool(cls.config, shape, cls.torch)
        cls.pool = FixtureFactory.hybrid_pool(
            cls.config, cls.device_pool, cls.rotary, worker
        )

    @classmethod
    def tearDownClass(cls):
        cls.torch.npu.synchronize()
        del cls.pool, cls.device_pool
        gc.collect()
        cls.torch.npu.empty_cache()

    def test_01_out_of_order_host_pages_across_batch_boundary(self):
        torch = self.torch
        page_size = self.config.page_size
        page_count = self.page_count
        self.assertEqual(
            page_count, self.pool.compressed_pool._QUANT_BATCH_MAX_PAGES + 1
        )
        before = (
            self.pool.sink_pool.available_size(),
            self.pool.compressed_pool.available_size(),
        )
        sink, allocated = self.pool.alloc(0, (page_count + 1) * page_size)
        self.assertIsNotNone(sink)
        self.assertIsNotNone(allocated)
        self.assertEqual(sink.numel(), 0)
        device_indices = torch.arange(
            page_size, (page_count + 1) * page_size, dtype=torch.int64
        )
        positions = torch.arange(
            page_size, 2 * page_size, device="npu", dtype=torch.int64
        ).repeat(page_count)
        try:
            FixtureFactory.fill_distinct_pages(
                self.device_pool,
                device_indices,
                positions,
                self.pool.compressed_pool,
                self.rotary,
                torch,
            )
            source_signatures = [
                FixtureFactory.page_basis_signatures(
                    values,
                    positions,
                    self.pool.compressed_pool,
                    self.rotary,
                    matrix,
                )
                for matrix, values in zip(
                    ("k", "v"), FixtureFactory.read(self.device_pool, device_indices)
                )
            ]
            host_pages = allocated.reshape(page_count + 1, page_size)
            page_ids = [
                page for page in range(page_count + 1) if page != page_count // 2
            ][::-1]
            host_indices = host_pages[page_ids].reshape(-1)
            request = FixtureFactory.request(
                self.device_pool,
                sink_host=sink,
                compressed_host=host_indices,
                sink_device=device_indices[:0],
                compressed_device=device_indices,
                compressed_tokens=positions,
            )
            self.pool.backup_from_device_all_layer(request)
            torch.npu.synchronize()
            for buffer in (self.device_pool.k_buffer, self.device_pool.v_buffer):
                buffer.flatten(1, 2).index_fill_(
                    1, device_indices.to("npu"), float("nan")
                )
            torch.npu.synchronize()
            FixtureFactory.reload(self.pool, request, torch)
            reference = [
                values.clone()
                for values in FixtureFactory.read(self.device_pool, device_indices)
            ]
            for matrix, values, source_signature in zip(
                ("k", "v"), reference, source_signatures
            ):
                self.assertTrue(bool(torch.isfinite(values).all()))
                self.assertFalse(
                    torch.equal(
                        values[:, :page_size], values[:, page_size : 2 * page_size]
                    )
                )
                restored_signature = FixtureFactory.page_basis_signatures(
                    values,
                    positions,
                    self.pool.compressed_pool,
                    self.rotary,
                    matrix,
                )
                minimum_gap = torch.diff(source_signature).abs().min()
                self.assertGreater(float(minimum_gap.item()), 0)
                error = (restored_signature - source_signature).abs()
                self.assertTrue(bool((error < minimum_gap / 2).all()))

            reversed_pages = list(range(page_count - 1, -1, -1))
            request.host_indices_compressed = host_indices.reshape(
                page_count, page_size
            )[reversed_pages].reshape(-1)
            for buffer in (self.device_pool.k_buffer, self.device_pool.v_buffer):
                buffer.flatten(1, 2).index_fill_(
                    1, device_indices.to("npu"), float("nan")
                )
            torch.npu.synchronize()
            FixtureFactory.reload(self.pool, request, torch)
            order = torch.tensor(reversed_pages, device="npu", dtype=torch.int64)
            for matrix, actual, expected, source_signature in zip(
                ("k", "v"),
                FixtureFactory.read(self.device_pool, device_indices),
                reference,
                source_signatures,
            ):
                expected = expected.reshape(
                    self.device_pool.layer_num,
                    page_count,
                    page_size,
                    self.device_pool.head_num,
                    self.device_pool.head_dim,
                ).index_select(1, order).reshape_as(actual)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                restored_signature = FixtureFactory.page_basis_signatures(
                    actual,
                    positions,
                    self.pool.compressed_pool,
                    self.rotary,
                    matrix,
                )
                expected_signature = source_signature.index_select(0, order)
                minimum_gap = torch.diff(source_signature).abs().min()
                self.assertTrue(
                    bool(
                        (
                            (restored_signature - expected_signature).abs()
                            < minimum_gap / 2
                        ).all()
                    )
                )
                # Reversing 33 pages changes which page uses the one-page batch.
                # Its PCA matmul may round differently from the 32-page batch.
                torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
        finally:
            torch.npu.synchronize()
            self.pool.free(allocated)
        self.assertEqual(
            (
                self.pool.sink_pool.available_size(),
                self.pool.compressed_pool.available_size(),
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main(verbosity=0)
