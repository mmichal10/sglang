"""CPU checks for compressed-pool initialization from KVTC artifacts."""

import ast
import logging
import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "python/sglang/srt/mem_cache"
QUANT = runpy.run_path(str(CACHE / "kvtc_quant.py"))
WORKER = "tp_0_pp_0"
P = 16
PAGE_SIZE = 2


def load_pool_class():
    path = CACHE / "memory_pool_host.py"
    tree = ast.parse(path.read_text())
    pool_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "NPUMHATokenToKVPoolCompressed"
    )
    pool_class.bases = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(pool_class)
    namespace = {
        **QUANT,
        "torch": torch,
        "logging": logging,
        "threading": threading,
        "logger": Mock(handlers=[]),
        "get_available_gpu_memory": Mock(return_value=1.0),
        "torch_npu": SimpleNamespace(npu=SimpleNamespace(synchronize=Mock())),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[pool_class.name]


Pool = load_pool_class()


def side():
    return {
        "mu": torch.zeros(P, dtype=torch.float32),
        "basis": torch.eye(P, dtype=torch.float32),
        "quant": {
            "2": [(8, "bfloat16")],
            "4": [(4, "float32")],
        },
    }


class TestKVTCPoolLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "artifact.pt"
        self.artifact = {
            "version": QUANT["KVTC_FILE_VERSION"],
            "keys": {WORKER: side()},
            "values": {WORKER: side()},
        }
        self.device_pool = SimpleNamespace(
            store_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=P,
            layer_num=1,
            device="cpu",
            size=PAGE_SIZE,
        )

    def make_pool(self, *, k_cr=2, v_cr=4, quant_disable=False, path=None):
        torch.save(self.artifact, self.path)
        npu = SimpleNamespace(current_device=Mock(return_value=0))
        with (
            patch.object(torch, "npu", npu, create=True),
            patch.object(Pool, "init_kv_buffer", return_value=None),
            patch.object(Pool, "clear", return_value=None, create=True),
        ):
            return Pool(
                device_pool=self.device_pool,
                host_to_device_ratio=2,
                host_size=1,
                page_size=PAGE_SIZE,
                kvtc_params_path=str(self.path) if path is None else path,
                kvtc_k_compression_ratio=k_cr,
                kvtc_v_compression_ratio=v_cr,
                kvtc_quant_disable=quant_disable,
                skip_size_check=True,
            )

    def test_quantized_both_sides_use_one_quantizer_and_selected_schemas(self):
        pool = self.make_pool()
        self.assertTrue(pool.k_kvtc)
        self.assertTrue(pool.v_kvtc)
        self.assertIsInstance(pool.quantizer, QUANT["KVTCQuantizer"])
        self.assertEqual(pool.quantizer.key_layout().feature_count, 8)
        self.assertEqual(pool.quantizer.value_layout().feature_count, 4)
        self.assertEqual(pool.kvtc_k_mu.shape, (P,))
        self.assertEqual(pool.kvtc_k_V.shape, (P, 8))
        self.assertEqual(pool.kvtc_v_V.shape, (P, 4))
        self.assertEqual(pool.offload_page_shape_k, (PAGE_SIZE, 8))
        self.assertEqual(pool.offload_page_shape_v, (PAGE_SIZE, 4))
        self.assertEqual(
            pool.page_size_bytes,
            PAGE_SIZE
            * (
                pool.quantizer.key_bytes_per_token()
                + pool.quantizer.value_bytes_per_token()
            ),
        )

    def test_pca_only_one_sided_and_raw_other_side(self):
        for k_cr, v_cr, compressed_side in ((2, 0, "k"), (0, 4, "v")):
            with self.subTest(compressed_side=compressed_side):
                pool = self.make_pool(k_cr=k_cr, v_cr=v_cr, quant_disable=True)
                self.assertIsNone(pool.quantizer)
                self.assertEqual(pool.k_kvtc, compressed_side == "k")
                self.assertEqual(pool.v_kvtc, compressed_side == "v")
                retained = P // (k_cr or v_cr)
                self.assertEqual(
                    getattr(pool, f"kvtc_{compressed_side}_V").shape,
                    (P, retained),
                )
                self.assertEqual(
                    pool.page_size_bytes,
                    PAGE_SIZE * (retained + P) * torch.bfloat16.itemsize,
                )

    def test_quantized_one_sided_uses_one_quantizer(self):
        for k_cr, v_cr, compressed_side in ((2, 0, "k"), (0, 4, "v")):
            with self.subTest(compressed_side=compressed_side):
                pool = self.make_pool(k_cr=k_cr, v_cr=v_cr)
                self.assertIsInstance(pool.quantizer, QUANT["KVTCQuantizer"])
                self.assertEqual(pool.k_kvtc, compressed_side == "k")
                self.assertEqual(pool.v_kvtc, compressed_side == "v")
                selected_bytes = (
                    pool.quantizer.key_bytes_per_token()
                    if compressed_side == "k"
                    else pool.quantizer.value_bytes_per_token()
                )
                self.assertEqual(
                    pool.page_size_bytes,
                    PAGE_SIZE * (selected_bytes + P * torch.bfloat16.itemsize),
                )

    def test_pca_only_both_sides(self):
        pool = self.make_pool(quant_disable=True)
        self.assertIsNone(pool.quantizer)
        self.assertEqual(pool.kvtc_k_V.shape, (P, 8))
        self.assertEqual(pool.kvtc_v_V.shape, (P, 4))
        self.assertEqual(
            pool.page_size_bytes, PAGE_SIZE * (8 + 4) * torch.bfloat16.itemsize
        )

    def test_no_path_keeps_uncompressed_pool(self):
        pool = self.make_pool(k_cr=2, v_cr=4, path="")
        self.assertFalse(pool.k_kvtc)
        self.assertFalse(pool.v_kvtc)
        self.assertIsNone(pool.quantizer)
        self.assertEqual(
            pool.page_size_bytes, 2 * PAGE_SIZE * P * torch.bfloat16.itemsize
        )

    def test_missing_requested_side_fails_during_loading(self):
        self.artifact.pop("keys")
        with self.assertRaisesRegex(ValueError, "missing keys/tp_0_pp_0"):
            self.make_pool(k_cr=2, v_cr=0)


if __name__ == "__main__":
    unittest.main()
