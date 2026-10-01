#!/usr/bin/env python3
"""Benchmark the production KVTC hybrid pool, without a server or model weights.

Run in the same Ascend/PyTorch environment used to launch SGLang. Setup and
validation are untimed. All measurements include Python and device work; no
model graph is captured. The default modes compare production configurations,
which can retain different PCA ranks, rather than isolated quantizer arithmetic.

Add --profile to capture Ascend CPU/NPU traces instead of latency measurements.
For a short smoke run, reuse your normal input arguments and add:
    --profile --tokens 256 --warmups 1 --profile-iterations 1
Traces go under ./kvtc_profiles/<unique-run>/<mode>/<offload|reload>/.
Inspect trace_view.json in a trace viewer or the exported operator summaries.
The kvtc/* ranges label CPU scopes with correlated NPU operations; their CPU
durations alone are not device-stage latency measurements.

Add --utilization-profile for isolated reload-only PCA PMU replays plus an
msprof physical-core/system-memory timeline. Results go under
./kvtc_profiles/<unique-run>/utilization/; baseline is not part of this mode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

SINK_TOKENS = 128
WORKER_PATTERN = re.compile(r"tp_(\d+)_pp_(\d+)")
DUMP_PATTERN = re.compile(r"(.+)-([KV])-chunk_(\d+)-layer_(\d+)\.bin")
MODE_NAMES = ("quant", "compressed", "baseline")
UTILIZATION_WORKERS = ("pipe", "arithmetic", "memory", "system")
TASK_UTILIZATION_METRICS = {
    "pipe": "PipeUtilization",
    "arithmetic": "ArithmeticUtilization",
    "memory": "Memory",
}
logger = logging.getLogger(__name__)


@dataclass
class Dump:
    name: str
    directory: Path
    token_count: int
    k_paths: tuple[Path, ...]
    v_paths: tuple[Path, ...]


@dataclass
class Mode:
    name: str
    page_bytes: int
    k_rank: int
    v_rank: int
    k_groups: int = 0
    v_groups: int = 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/benchmark_kvtc.py --model-dir /models/Qwen3 "
            "--compression-matrix /data/kvtc.pt --dump-dir /data/kv-dumps "
            "--dump-name request-id --tokens 8192 --k-cr 8 --v-cr 8\n"
            "All three modes use hybrid routing with 128 sink tokens. Host pool "
            "construction uses the production 90/10 split (at least 10 GB total). "
            "Logical GiB/s counts original K+V bytes, not physical bus bandwidth."
        ),
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Local model directory; only configuration/RoPE is loaded",
    )
    parser.add_argument(
        "--compression-matrix", type=Path,
        help="PCA/quantization calibration artifact; required unless only baseline is enabled",
    )
    parser.add_argument(
        "--dump-dir", type=Path, required=True,
        help="Dump dataset, worker directory, or parent containing dump datasets",
    )
    parser.add_argument(
        "--tp-worker-name", default="tp_0_pp_0",
        help="Single TP/PP worker whose dumps and calibration entries are used",
    )
    parser.add_argument(
        "--dump-name",
        help="Conversation name from discovery; otherwise select interactively if multiple exist",
    )
    parser.add_argument(
        "--tokens", type=int,
        help="Use the first N tokens (entire dump when omitted); round down to whole pages",
    )
    parser.add_argument(
        "--page-size", type=int, default=128,
        help="Tokens per page; must be a positive divisor of 128",
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Measured iterations per direction, per enabled mode",
    )
    parser.add_argument(
        "--warmups", type=int, default=5,
        help="Untimed warmup iterations per direction, per enabled mode; zero is allowed",
    )
    descriptions = (
        "PCA plus quantization", "PCA without quantization", "uncompressed hybrid baseline"
    )
    for name, description in zip(MODE_NAMES, descriptions):
        parser.add_argument(
            f"--disable-{name}-benchmark", action="store_true",
            help=f"Skip the {description} benchmark",
        )
    for side in ("k", "v"):
        parser.add_argument(
            f"--{side}-cr", type=int,
            help=(f"{side.upper()} compression ratio; "
                  "infer only if the artifact has exactly one ratio"),
        )
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU device index (uses the selected device with no distributed process group)",
    )
    parser.add_argument(
        "--host-memory-gb", type=int,
        help=("Production host-pool size in decimal GB; "
              "automatically size for all modes when omitted, minimum 10 GB"),
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile", action="store_true",
        help="Collect CPU/NPU traces instead of benchmark timings; --iterations is unused",
    )
    profile_group.add_argument(
        "--utilization-profile", action="store_true",
        help=("Profile reload-only PCA reconstruction utilization using isolated "
              "task and system profiler passes"),
    )
    parser.add_argument(
        "--profile-iterations", type=int, default=1,
        help="Captured iterations per enabled mode and direction when --profile is set",
    )
    parser.add_argument(
        "--profile-dir", type=Path, default=Path("kvtc_profiles"),
        help="Trace output parent; each profiling run creates a unique subdirectory",
    )
    parser.add_argument(
        "--utilization-iterations", type=int, default=3,
        help="Captured reload iterations per PCA mode and utilization profiler pass",
    )
    parser.add_argument(
        "--_utilization-worker", choices=UTILIZATION_WORKERS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_utilization-run-dir", type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def enabled_modes(args):
    names = [name for name in MODE_NAMES if not getattr(args, f"disable_{name}_benchmark")]
    if getattr(args, "utilization_profile", False):
        names = [name for name in names if name != "baseline"]
    return names


def validate_args(parser, args):
    if not enabled_modes(args):
        if args.utilization_profile:
            parser.error("Utilization profiling requires quant or compressed mode")
        parser.error("At least one benchmark must be enabled")
    if args.page_size <= 0 or SINK_TOKENS % args.page_size:
        parser.error("--page-size must be a positive divisor of 128")
    if args.iterations <= 0 or args.warmups < 0:
        parser.error("--iterations must be positive and --warmups must be nonnegative")
    if args.profile_iterations <= 0:
        parser.error("--profile-iterations must be positive")
    if args.utilization_iterations <= 0:
        parser.error("--utilization-iterations must be positive")
    if bool(args._utilization_worker) != bool(args._utilization_run_dir):
        parser.error("Internal utilization worker arguments must be provided together")
    if args._utilization_worker and not args.utilization_profile:
        parser.error("Internal utilization workers require --utilization-profile")
    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.device < 0 or not WORKER_PATTERN.fullmatch(args.tp_worker_name):
        parser.error("Use a nonnegative --device and --tp-worker-name tp_<N>_pp_<N>")
    if any(ratio is not None and ratio <= 0 for ratio in (args.k_cr, args.v_cr)):
        parser.error("Compression ratios must be positive integers")
    if args.host_memory_gb is not None and args.host_memory_gb < 10:
        parser.error("The production hybrid allocation requires --host-memory-gb >= 10")
    if (
        any(name != "baseline" for name in enabled_modes(args))
        and args.compression_matrix is None
    ):
        parser.error("--compression-matrix is required for quant/compressed benchmarks")


def load_cpu(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def discover_dumps(root, worker, loader=load_cpu):
    """Use calibration's request/chunk/layer grouping without sampling filters."""
    if not root.is_dir():
        raise ValueError(f"Dump directory does not exist: {root}")
    if root.name == worker:
        directories = [root]
    elif (root / worker).is_dir():
        directories = [root / worker]
    else:
        directories = sorted(
            path / worker for path in root.iterdir() if (path / worker).is_dir()
        )
    if not directories:
        available = sorted({
            path.name for path in (root, *root.glob("*"), *root.glob("*/*"))
            if path.is_dir() and WORKER_PATTERN.fullmatch(path.name)
        })
        raise ValueError(
            f"Worker {worker} not found in {root}; "
            f"available workers: {', '.join(available) or '(none)'}"
        )

    dumps = []
    for directory in directories:
        requests = defaultdict(lambda: {"K": {}, "V": {}})
        for path in sorted(directory.iterdir()):
            match = DUMP_PATTERN.fullmatch(path.name)
            if path.is_file() and match:
                request, kv, chunk, layer = match.groups()
                requests[request][kv][(int(chunk), int(layer))] = path
        for name, sides in sorted(requests.items()):
            try:
                keys = sorted(sides["K"])
                if not keys or keys != sorted(sides["V"]):
                    raise ValueError("K/V chunk and layer sets differ or are missing")
                chunks = sorted({chunk for chunk, _ in keys})
                layers = sorted({layer for _, layer in keys})
                if (
                    chunks != list(range(len(chunks)))
                    or layers != list(range(len(layers)))
                ):
                    raise ValueError("chunk/layer IDs must be contiguous starting at zero")
                if len(keys) != len(chunks) * len(layers):
                    raise ValueError("incomplete layer set in a chunk")
                token_count = 0
                for chunk in chunks:
                    k = loader(sides["K"][(chunk, 0)])
                    v = loader(sides["V"][(chunk, 0)])
                    if len(k.shape) != 3 or k.shape != v.shape:
                        raise ValueError(
                            "layer-0 K/V must have matching [token, head, head_dim] shapes"
                        )
                    token_count += k.shape[0]
                dumps.append(Dump(
                    name, directory, token_count,
                    tuple(sides["K"][key] for key in keys),
                    tuple(sides["V"][key] for key in keys),
                ))
            except (ValueError, RuntimeError, OSError, EOFError) as error:
                logger.warning("Skipping %s/%s: %s", directory, name, error)
    counts = Counter(dump.name for dump in dumps)
    for dump in dumps:
        if counts[dump.name] > 1:
            dump.name = f"{dump.directory.parent.name}/{dump.name}"
    return dumps


def select_dump(dumps, name, *, interactive=None, input_fn=input):
    if not dumps:
        raise ValueError("No complete K/V dumps found for the selected worker")
    print("Available KV dumps:", flush=True)
    for index, dump in enumerate(dumps, 1):
        print(f"  {index:3d}. {dump.name}  ({dump.token_count:,} tokens)", flush=True)
    if name is not None:
        matches = [dump for dump in dumps if dump.name == name]
        if len(matches) != 1:
            raise ValueError(f"--dump-name {name!r} must match one of the names above")
        return matches[0]
    if len(dumps) == 1:
        return dumps[0]
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError("Multiple dumps found; pass --dump-name when stdin is not interactive")
    while True:
        choice = input_fn("Select a dump by number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(dumps):
            return dumps[int(choice) - 1]
        matches = [dump for dump in dumps if dump.name == choice]
        if len(matches) == 1:
            return matches[0]
        print("Please enter a number or name from the list.", flush=True)


def selected_token_count(total, requested, page_size):
    requested = total if requested is None else requested
    if requested > total:
        raise ValueError(f"Requested {requested} tokens, but the dump contains {total}")
    actual = requested // page_size * page_size
    if actual <= SINK_TOKENS:
        raise ValueError(
            f"Need at least {SINK_TOKENS + page_size} tokens after rounding "
            "(128 sink tokens plus one remaining page)"
        )
    return actual


def load_selected_dump(dump, tokens):
    from scripts.kvtc_calibration_data import load_tensor

    keys, k_count = load_tensor(dump.k_paths)
    values, v_count = load_tensor(dump.v_paths)
    if (
        keys is None or values is None
        or k_count != dump.token_count or v_count != k_count
        or keys.shape != values.shape or keys.dtype != values.dtype
    ):
        raise ValueError("Selected dump has incomplete or mismatched K/V tensors")
    # Unlike load_sample_pool(), preserve the sink and trailing tokens and leave
    # keys rotated: the production hybrid pool performs its own inverse RoPE.
    return keys[:tokens].contiguous(), values[:tokens].contiguous()


def resolve_ratio(params, requested, option):
    if requested is not None:
        return requested
    ratios = params.get("quant", {})
    if len(ratios) == 1:
        ratio = int(next(iter(ratios)))
        if ratio > 0:
            return ratio
    available = ', '.join(map(str, ratios)) or '(none; PCA-only artifact)'
    raise ValueError(f"Pass {option}; available artifact ratios: {available}")


def describe_modes(args, shape, dtype, report=print):
    import torch

    from sglang.srt.mem_cache.kvtc_quant import KVTCQuantizer, quant_group_bits

    _, layers, heads, head_dim = shape
    features = layers * heads * head_dim
    raw_page_bytes = 2 * features * args.page_size * dtype.itemsize
    names = enabled_modes(args)
    params = {}
    if any(name != "baseline" for name in names):
        artifact = load_cpu(args.compression_matrix)
        for side, option in (("keys", "k_cr"), ("values", "v_cr")):
            entry = artifact.get(side, {}).get(args.tp_worker_name)
            if entry is None:
                raise ValueError(f"Artifact is missing {side}/{args.tp_worker_name}")
            basis, mu = entry["basis"], entry["mu"]
            if basis.ndim != 2 or basis.shape[0] != features or mu.shape != (features,):
                raise ValueError(
                    f"Artifact {side} dimensions do not match {features} local KV features"
                )
            if basis.dtype != torch.float32 or mu.dtype != torch.float32:
                raise ValueError(
                    "Expected FP32 PCA means and bases, as produced by kvtc_calibrate.py"
                )
            ratio = resolve_ratio(
                entry, getattr(args, option), "--" + option.replace("_", "-")
            )
            setattr(args, option, ratio)
            params[side] = (entry, ratio)

    modes = []
    for name in names:
        if name == "baseline":
            modes.append(Mode(name, raw_page_bytes, features, features))
            continue
        ranks, groups, page_bytes = [], [], 0
        for side in ("keys", "values"):
            entry, ratio = params[side]
            if name == "compressed":
                rank = min(features // ratio, entry["basis"].shape[1])
                if rank == 0:
                    raise ValueError(f"{side} compression ratio {ratio} retains no features")
                ranks.append(rank)
                groups.append(0)
                page_bytes += args.page_size * rank * dtype.itemsize
            else:
                schema = entry.get("quant", {}).get(str(ratio))
                layout = KVTCQuantizer.build_layout(
                    schema, page_size=args.page_size,
                    basis_rank=entry["basis"].shape[1], matrix_name=side,
                )
                ranks.append(layout.feature_count)
                groups.append(layout.group_count)
                page_bytes += args.page_size * sum(
                    quant_group_bits(size, storage) for size, storage in schema
                ) // 8
                report(f"  quant {side} schema: {schema}")
        modes.append(Mode(name, page_bytes, *ranks, *groups))
    return modes, raw_page_bytes


def required_host_gb(tokens, page_size, modes, raw_page_bytes):
    # Match the real constructor's integer decimal-GB 90/10 split. The remainder
    # pool must be larger than the entire device pool, even though it stores only
    # tokens after the sink. Reserve one page beyond that constructor check.
    remainder_bytes = max(mode.page_bytes for mode in modes) * (tokens // page_size + 1)
    sink_bytes = raw_page_bytes * (SINK_TOKENS // page_size + 1)
    size = max(10, math.ceil(max(remainder_bytes / 0.9, sink_bytes / 0.1) / 1e9))
    while int(size * 0.90) * 1e9 < remainder_bytes or int(size * 0.1) * 1e9 < sink_bytes:
        size += 1
    return size


def measure(operation, warmups, iterations, synchronize, clock=time.perf_counter_ns):
    for _ in range(warmups):
        synchronize()
        operation()
        synchronize()
    elapsed = []
    for _ in range(iterations):
        synchronize()
        started = clock()
        operation()
        synchronize()
        elapsed.append((clock() - started) / 1e6)
    return elapsed


def reload_all_layers(pool, request, layer_count):
    for layer in range(layer_count):
        request.layer_id = layer
        pool.load_to_device_per_layer(request)


def create_profile_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=parent.resolve()
    ))


def profile_operation(operation, host, warmups, iterations, directory, label):
    """Capture only warmed-up transfers; never use profiler timings as benchmarks."""
    import torch
    import torch_npu

    pools = (host, host.compressed_pool)
    for pool in pools:
        pool._profile_kvtc = False
    try:
        for _ in range(warmups):
            torch.npu.synchronize()
            try:
                operation()
            finally:
                torch.npu.synchronize()
        directory.mkdir(parents=True, exist_ok=True)
        profiler = torch_npu.profiler
        config = profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            export_type=[profiler.ExportType.Text],
        )
        torch.npu.synchronize()
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            experimental_config=config,
            on_trace_ready=profiler.tensorboard_trace_handler(
                str(directory), async_mode=False,
            ),
        ) as capture:
            for pool in pools:
                pool._profile_kvtc = True
            try:
                for iteration in range(iterations):
                    torch.npu.synchronize()
                    with torch.profiler.record_function(f"kvtc/{label}/iteration_{iteration}"):
                        operation()
                        # Completion belongs to the iteration, not to individual stages.
                        with torch.profiler.record_function("kvtc/iteration_wait"):
                            torch.npu.synchronize()
                    capture.step()
            finally:
                for pool in pools:
                    pool._profile_kvtc = False
    finally:
        # Also covers warmup, profiler construction, and export failures.
        for pool in pools:
            pool._profile_kvtc = False
    return str(directory)


def profile_utilization_operation(
    operation, host, warmups, iterations, directory, label, metric
):
    """Collect one task-level PMU group around warmed-up reloads only."""
    import torch
    import torch_npu

    if metric not in TASK_UTILIZATION_METRICS:
        raise ValueError(f"Unknown utilization metric worker: {metric}")
    pools = (host, host.compressed_pool)
    for pool in pools:
        pool._profile_kvtc = False
    try:
        for _ in range(warmups):
            torch.npu.synchronize()
            try:
                operation()
            finally:
                torch.npu.synchronize()
        directory.mkdir(parents=True, exist_ok=True)
        profiler = torch_npu.profiler
        config = profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=getattr(
                profiler.AiCMetrics, TASK_UTILIZATION_METRICS[metric]
            ),
            l2_cache=metric == "memory",
            export_type=[profiler.ExportType.Text],
        )
        torch.npu.synchronize()
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            experimental_config=config,
            on_trace_ready=profiler.tensorboard_trace_handler(
                str(directory), async_mode=False,
            ),
        ) as capture:
            for pool in pools:
                pool._profile_kvtc = True
            try:
                for iteration in range(iterations):
                    torch.npu.synchronize()
                    try:
                        with torch.profiler.record_function(
                            f"kvtc/utilization/{label}/iteration_{iteration}"
                        ):
                            operation()
                            with torch.profiler.record_function("kvtc/iteration_wait"):
                                torch.npu.synchronize()
                    except BaseException:
                        torch.npu.synchronize()
                        raise
                    capture.step()
            finally:
                for pool in pools:
                    pool._profile_kvtc = False
    finally:
        for pool in pools:
            pool._profile_kvtc = False
    return str(directory)


def profile_system_operation(operation, warmups, iterations, label):
    """Mark system-profile reloads with device-correlated MSTX ranges."""
    import torch
    import torch_npu

    for _ in range(warmups):
        torch.npu.synchronize()
        try:
            operation()
        finally:
            torch.npu.synchronize()
    mstx = torch_npu.npu.mstx
    for iteration in range(iterations):
        torch.npu.synchronize()
        range_id = None
        try:
            range_id = mstx.range_start(
                f"kvtc_utilization_{label}_iteration_{iteration}",
                torch.npu.current_stream(),
            )
            operation()
        finally:
            # Complete device work even when the Python wrapper raises, then
            # close the range so later marks cannot be accidentally nested.
            try:
                torch.npu.synchronize()
            finally:
                try:
                    if range_id:
                        mstx.range_end(range_id)
                finally:
                    torch.npu.synchronize()
    return "msprof system capture"


def run_direction(args, mode, direction, operation, host):
    if args.profile:
        return profile_operation(
            operation, host, args.warmups, args.profile_iterations,
            args.profile_run_dir / mode.name / direction,
            f"{mode.name}/{direction}",
        )
    import torch

    return measure(operation, args.warmups, args.iterations, torch.npu.synchronize)


def git_version():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        return f"{commit} (worktree {'dirty' if status else 'clean'})"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (not a Git checkout)"


def check_reconstruction(device_pool, keys, values, page_size, mode):
    """Validate once, in bounded CPU chunks, after all timed operations."""
    import torch

    checks = []
    for name, buffer, original in (
        ("K", device_pool.k_buffer, keys), ("V", device_pool.v_buffer, values)
    ):
        # Ignore the dummy first page; compare [layer, token, head, head_dim].
        device_view = buffer[:, 1:].flatten(1, 2)
        expected = original.transpose(0, 1)
        error = energy = 0.0
        for start in range(0, len(original), page_size):
            actual = device_view[:, start : start + page_size].cpu()
            reference = expected[:, start : start + page_size]
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"{mode} reload contains nonfinite {name} values")
            if (
                mode == "baseline" or start < SINK_TOKENS
            ) and not torch.equal(actual, reference):
                raise ValueError(
                    f"{mode} reload did not exactly restore {name} "
                    f"at tokens {start}:{start + page_size}"
                )
            difference = actual.float() - reference.float()
            error += difference.square().sum(dtype=torch.float64).item()
            energy += reference.float().square().sum(dtype=torch.float64).item()
        relative = math.sqrt(error / energy) if energy else (0.0 if error == 0 else math.inf)
        checks.append(f"{name} relative-L2={relative:.6g}")
    validation = f"  {mode} validation: finite, sink exact; " + ", ".join(checks)
    print(validation, flush=True)
    return validation


def run_mode(args, mode, device_pool, keys, values, rotary_emb):
    import torch

    from sglang.srt.mem_cache.memory_pool_host import (
        KVTCHostMemoryRequest,
        NPUMHATokenToKVPoolHybrid,
    )

    tp_rank, pp_rank = map(int, WORKER_PATTERN.fullmatch(args.tp_worker_name).groups())
    baseline = mode.name == "baseline"
    host = NPUMHATokenToKVPoolHybrid(
        device_pool=device_pool,
        host_to_device_ratio=1,
        host_size=args.host_memory_gb,
        page_size=args.page_size,
        layout="page_first_direct",
        kvtc_params_path="" if baseline else str(args.compression_matrix),
        kvtc_k_compression_ratio=0 if baseline else args.k_cr,
        kvtc_v_compression_ratio=0 if baseline else args.v_cr,
        kvtc_quant_disable=mode.name == "compressed",
        rotary_emb=rotary_emb,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
    )
    sink, remainder = host.alloc(SINK_TOKENS, len(keys) - SINK_TOKENS)
    if sink is None or remainder is None:
        raise ValueError("Host pool allocation failed; increase --host-memory-gb")
    indices = torch.arange(args.page_size, args.page_size + len(keys), dtype=torch.int64)
    positions = torch.arange(SINK_TOKENS, len(keys), dtype=torch.int64, device="npu")
    request = KVTCHostMemoryRequest(
        device_memory_pool=device_pool,
        host_indices_compressed=remainder,
        device_indices_compressed=indices[SINK_TOKENS:],
        token_indices_compressed=positions,
        host_indices_sink=sink,
        device_indices_sink=indices[:SINK_TOKENS],
        io_backend="kernel_ascend",
        layer_id=None,
    )
    # Restore the original data once per mode. Offloads cannot alter this source;
    # reloads happen only after every offload, so lossy errors cannot accumulate.
    for buffer, original in ((device_pool.k_buffer, keys), (device_pool.v_buffer, values)):
        buffer[:, 1:].copy_(original.transpose(0, 1).reshape_as(buffer[:, 1:]))
    torch.npu.synchronize()
    if args.utilization_profile:
        print(f"Running {mode.name}: setup offload, then reload-only utilization", flush=True)
    else:
        print(f"Running {mode.name}: offload, then reload", flush=True)
    # Match the controller's separate write/load streams, including eager Python
    # wrappers. Synchronization at measurement boundaries waits for completion.
    write_stream, load_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(write_stream):
        if args.utilization_profile:
            host.backup_from_device_all_layer(request)
            torch.npu.synchronize()
            offload = None
        else:
            offload = run_direction(
                args, mode, "offload",
                lambda: host.backup_from_device_all_layer(request), host,
            )
    # Erase the destination once outside timing: validation must prove that the
    # reload really wrote the data rather than finding the original input there.
    device_pool.k_buffer[:, 1:].fill_(float("nan"))
    device_pool.v_buffer[:, 1:].fill_(float("nan"))
    torch.npu.synchronize()
    with torch.npu.stream(load_stream):
        reload_operation = lambda: reload_all_layers(
            host, request, device_pool.layer_num
        )
        if args.utilization_profile and args._utilization_worker == "system":
            reload = profile_system_operation(
                reload_operation, args.warmups, args.utilization_iterations,
                f"{mode.name}_reload",
            )
        elif args.utilization_profile:
            reload = profile_utilization_operation(
                reload_operation, host, args.warmups, args.utilization_iterations,
                args._utilization_run_dir / "task" / args._utilization_worker
                / mode.name / "reload",
                f"{mode.name}/reload", args._utilization_worker,
            )
        else:
            reload = run_direction(
                args, mode, "reload", reload_operation, host,
            )
    validation = check_reconstruction(device_pool, keys, values, args.page_size, mode.name)
    if args.utilization_profile:
        return {"reload": reload}, validation
    return {"offload": offload, "reload": reload}, validation


def format_results(results, tokens, logical_bytes):
    lines = [
        "\nCompleted-operation wall time (setup, warmups and validation excluded)",
        "Logical GiB/s uses original K+V bytes; speedup = baseline median / mode median.",
        f"{'Mode':<12} {'Direction':<8} {'Mean ms':>10} {'Median ms':>10} "
        f"{'Min ms':>10} {'Max ms':>10} {'P95 ms':>10} "
        f"{'Tokens/s':>12} {'GiB/s':>10} {'Speedup':>9}",
    ]
    for name, directions in results.items():
        for direction, samples in directions.items():
            median = statistics.median(samples)
            baseline = results.get("baseline", {}).get(direction)
            speedup = f"{statistics.median(baseline) / median:.3f}x" if baseline else "n/a"
            p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
            seconds = median / 1000
            lines.append(
                f"{name:<12} {direction:<8} {statistics.mean(samples):10.3f} "
                f"{median:10.3f} {min(samples):10.3f} {max(samples):10.3f} {p95:10.3f} "
                f"{tokens / seconds:12.1f} {logical_bytes / 2**30 / seconds:10.3f} "
                f"{speedup:>9}"
            )
    return "\n".join(lines)


def print_results(results, tokens, logical_bytes):
    print(format_results(results, tokens, logical_bytes), flush=True)


def print_final_report(metadata, validations, results, tokens, logical_bytes, *, profiling=False):
    # Reuse captured settings, including the original commit and resolved
    # defaults. One print keeps the copyable report together after framework logs.
    kind = "PROFILE" if profiling else "BENCHMARK"
    if profiling:
        output = "\n".join([
            "\nTrace directories (CPU/NPU timeline and operator summaries):",
            *(f"  {mode}/{direction}: {path}"
              for mode, directions in results.items() for direction, path in directions.items()),
            "Stage ranges describe CPU scopes and correlated NPU work; "
            "stages are not individually synchronized.",
            "No benchmark timings or speedups were measured in profiling mode.",
        ])
    else:
        output = format_results(results, tokens, logical_bytes)
    report = "\n".join([
        f"\n========== KVTC {kind} REPORT ==========",
        *metadata,
        "\nReconstruction checks (outside timing):",
        *validations,
        output,
        f"========== END KVTC {kind} REPORT ==========",
    ])
    print(report, flush=True)


def normalize_csv_header(value):
    """Normalize profiler headers across CANN releases and CSV spellings."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _csv_number(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "na", "none", "null", "--"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _row_value(row, *aliases):
    for alias in aliases:
        value = row.get(normalize_csv_header(alias))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _profile_path_parts(path, root):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    mode = next((part for part in parts if part in {"quant", "compressed"}), "unknown")
    metric = next((part for part in parts if part in TASK_UTILIZATION_METRICS), "unknown")
    return mode, metric


def parse_op_summary_files(utilization_dir):
    """Merge MatMul rows from all PMU replays using duration-weighted metrics."""
    utilization_dir = Path(utilization_dir)
    grouped = {}
    # Legacy msprof exports use op_summary_*.csv, while current Ascend PyTorch
    # Profiler releases expose the same per-kernel data as kernel_details.csv.
    source_files = sorted({
        *utilization_dir.glob("task/**/op_summary*.csv"),
        *utilization_dir.glob("task/**/kernel_details*.csv"),
    })
    for path in source_files:
        mode, metric = _profile_path_parts(path, utilization_dir)
        try:
            csv_file = path.open(newline="", encoding="utf-8-sig")
        except OSError as error:
            logger.warning("Cannot read profiler summary %s: %s", path, error)
            continue
        with csv_file:
            reader = csv.DictReader(csv_file)
            original_headers = {
                normalize_csv_header(header): header for header in (reader.fieldnames or [])
            }
            for raw_row in reader:
                row = {
                    normalize_csv_header(header): value
                    for header, value in raw_row.items() if header is not None
                }
                kernel = _row_value(row, "Op Name", "Name", "Kernel Name")
                op_type = _row_value(row, "OP Type", "Type")
                if "matmul" not in f"{kernel} {op_type}".lower():
                    continue
                shape = _row_value(row, "Input Shapes", "Input Shape") or "unavailable"
                task_type = _row_value(
                    row, "Task Type", "Accelerator Core", "Core Type"
                ) or op_type or "unavailable"
                duration = _csv_number(_row_value(
                    row, "Task Duration(us)", "Task Duration(µs)",
                    "Duration(us)", "Duration(µs)", "Device Duration(us)",
                    "aicore_time(us)",
                ))
                key = (mode, kernel or op_type or "MatMul", shape, task_type)
                aggregate = grouped.setdefault(key, {
                    "passes": defaultdict(lambda: {
                        "count": 0, "duration": 0.0, "duration_count": 0,
                    }),
                    "columns": {},
                    "headers": {},
                })
                replay = aggregate["passes"][metric]
                replay["count"] += 1
                if duration is not None:
                    replay["duration"] += duration
                    replay["duration_count"] += 1
                weight = duration if duration is not None and duration > 0 else 1.0
                identity = {
                    "opname", "name", "kernelname", "optype", "type", "tasktype",
                    "acceleratorcore", "coretype", "inputshapes", "inputshape",
                    "taskdurationus", "taskdurations", "durationus", "durations",
                    "devicedurationus", "devicedurations", "aicoretimeus", "aicoretimes",
                }
                for header, value in row.items():
                    if header in identity:
                        continue
                    if header in {"blockdim", "blocknum"}:
                        header = "blockdim"
                    stats = aggregate["columns"].setdefault(
                        header, {"weighted": 0.0, "weight": 0.0}
                    )
                    aggregate["headers"].setdefault(
                        header,
                        "Block Dim" if header == "blockdim"
                        else original_headers.get(header, header),
                    )
                    number = _csv_number(value)
                    if number is None:
                        continue
                    stats["weighted"] += number * weight
                    stats["weight"] += weight

    rows = []
    for (mode, kernel, shape, task_type), aggregate in sorted(grouped.items()):
        passes = aggregate["passes"]
        preferred = next(
            (passes[name] for name in ("pipe", "arithmetic", "memory")
             if passes[name]["count"]),
            {"count": 0, "duration": 0.0, "duration_count": 0},
        )
        count = max((item["count"] for item in passes.values()), default=0)
        row = {
            "mode": mode,
            "kernel_name": kernel,
            "input_shape": shape,
            "task_type": task_type,
            "call_count": count,
            "avg_device_duration_us": (
                preferred["duration"] / preferred["duration_count"]
                if preferred["duration_count"] else ""
            ),
        }
        for header, stats in aggregate["columns"].items():
            output_header = aggregate["headers"][header]
            row[output_header] = (
                stats["weighted"] / stats["weight"] if stats["weight"] else ""
            )
        rows.append(row)
    return rows, source_files


def parse_physical_core_utilization(system_dir):
    """Return per-core averages from optional AI/Vector utilization exports."""
    system_dir = Path(system_dir)
    result = {}
    files = sorted(system_dir.glob("**/ai*_core_utilization*.csv"))
    # Older releases use ai_core_utilization while newer ones may insert vector.
    files.extend(path for path in sorted(system_dir.glob("**/ai_core_utilization*.csv"))
                 if path not in files)
    for path in files:
        core_kind = "AI Vector Core" if "vector" in path.name.lower() else "AI Core"
        values = defaultdict(list)
        try:
            csv_file = path.open(newline="", encoding="utf-8-sig")
        except OSError as error:
            logger.warning("Cannot read physical-core summary %s: %s", path, error)
            continue
        with csv_file:
            reader = csv.DictReader(csv_file)
            core_headers = {}
            metric_header = None
            for header in reader.fieldnames or []:
                if normalize_csv_header(header) in {"metric", "metrics"}:
                    metric_header = header
                match = re.fullmatch(r"core(\d+)", normalize_csv_header(header))
                if match:
                    core_headers[header] = int(match.group(1))
            for raw_row in reader:
                if metric_header:
                    metric_name = str(raw_row.get(metric_header, "")).lower()
                    if metric_name and "utilization" not in metric_name:
                        continue
                for header, core_id in core_headers.items():
                    number = _csv_number(raw_row.get(header))
                    if number is not None:
                        values[core_id].append(number)
        if values:
            destination = result.setdefault(core_kind, defaultdict(list))
            for core_id, samples in values.items():
                destination[core_id].extend(samples)
    summaries = {}
    for core_kind, cores in result.items():
        averages = {
            core_id: statistics.mean(samples) for core_id, samples in sorted(cores.items())
        }
        summaries[core_kind] = {
            "cores": averages,
            "min": min(averages.values()),
            "max": max(averages.values()),
        }
    return summaries, files


def write_utilization_summary(path, rows):
    fixed = [
        "mode", "kernel_name", "input_shape", "task_type", "call_count",
        "avg_device_duration_us",
    ]
    extras = []
    seen = set(fixed)
    for row in rows:
        for header in row:
            if header not in seen:
                seen.add(header)
                extras.append(header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fixed + extras)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _matching_metrics(row, patterns):
    matches = []
    for header, value in row.items():
        normalized = normalize_csv_header(header)
        if any(pattern(normalized) for pattern in patterns) and value not in ("", None):
            number = _csv_number(value)
            if number is not None:
                matches.append((header, number))
    return matches


def _metric_text(matches, include_names=False):
    if not matches:
        return "unsupported/unavailable"
    if include_names and len(matches) > 1:
        return ";".join(
            f"{normalize_csv_header(name)}={value:.2f}" for name, value in matches
        )
    return f"{matches[0][1]:.2f}"


def format_utilization_report(rows, core_summaries, utilization_dir, source_files):
    lines = [
        "\nMatMul reload utilization (duration-weighted; PMU ratios are active-cycle ratios,",
        "not direct percentages of theoretical chip FLOP/s):",
        (f"{'Mode':<11} {'Kernel':<28} {'Input shape':<22} {'Task':<12} "
         f"{'Calls':>5} {'Avg us':>10} {'Block':>8} {'Cube/MAC':>12} "
         f"{'Arithmetic':>14} {'Vector':>12} {'MTE':>12} {'Mem R/W GB/s':>22}"),
    ]
    for row in rows:
        cube = _metric_text(_matching_metrics(row, [
            lambda name: name in {"aicmacratio", "macratio", "cubeutilization"},
        ]))
        arithmetic = _metric_text(_matching_metrics(row, [
            lambda name: "ratio" in name and any(
                dtype in name for dtype in ("fp16", "bf16", "fp32", "int8", "int4")
            ),
            lambda name: "arithmetic" in name and "ratio" in name,
        ]), include_names=True)
        vector = _metric_text(_matching_metrics(row, [
            lambda name: name in {"aivvecratio", "vecratio", "vectorratio"},
        ]))
        mte = _matching_metrics(row, [
            lambda name: "mte" in name and "ratio" in name,
        ])
        memory = _matching_metrics(row, [
            lambda name: name in {
                "mainmemreadbwgbs", "mainmemwritebwgbs",
                "mainmemoryreadbandwidthgbs", "mainmemorywritebandwidthgbs",
            },
        ])
        block = _metric_text(_matching_metrics(row, [
            lambda name: name in {"blockdim", "blocknum"},
        ]))
        duration = row["avg_device_duration_us"]
        duration_text = f"{duration:.3f}" if isinstance(duration, (int, float)) else "unavailable"
        lines.append(
            f"{row['mode']:<11} {row['kernel_name'][:28]:<28} "
            f"{row['input_shape'][:22]:<22} {row['task_type'][:12]:<12} "
            f"{row['call_count']:5d} {duration_text:>10} {block:>8} {cube:>12} "
            f"{arithmetic:>14} {vector:>12} "
            f"{_metric_text(mte, include_names=True):>12} "
            f"{_metric_text(memory, include_names=True):>22}"
        )
    if not rows:
        lines.append("  No MatMul rows were exported; task PMU metrics unsupported/unavailable.")
    lines.append("\nPhysical sampled core utilization:")
    if core_summaries:
        for kind, summary in sorted(core_summaries.items()):
            cores = ", ".join(
                f"Core {core_id}={average:.2f}%"
                for core_id, average in summary["cores"].items()
            )
            lines.append(
                f"  {kind}: {cores}; overall min={summary['min']:.2f}%, "
                f"max={summary['max']:.2f}%"
            )
    else:
        lines.append("  AI/Vector per-core CSV summaries: unsupported/unavailable")
    system_dir = Path(utilization_dir) / "system"
    timelines = list(system_dir.glob("**/msprof_*.json"))
    memory_files = list(system_dir.glob("**/*mem*.csv")) + list(
        system_dir.glob("**/*bandwidth*.csv")
    )
    lines.extend([
        (f"\nArtifacts: {len(source_files)} task op-summary file(s), "
         f"{len(timelines)} authoritative msprof timeline(s), "
         f"{len(memory_files)} system-memory/bandwidth CSV file(s)."),
        f"Complete artifacts: {Path(utilization_dir)}",
        f"Consolidated CSV: {Path(utilization_dir) / 'utilization_summary.csv'}",
    ])
    return "\n".join(lines)


def _worker_argv(argv, worker, run_dir):
    cleaned = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in {"--_utilization-worker", "--_utilization-run-dir"}:
            skip = True
            continue
        if item.startswith("--_utilization-worker=") or item.startswith(
            "--_utilization-run-dir="
        ):
            continue
        cleaned.append(item)
    return [
        sys.executable, str(Path(__file__).resolve()), *cleaned,
        "--_utilization-worker", worker,
        "--_utilization-run-dir", str(run_dir),
    ]


def run_utilization_orchestrator(args, argv):
    msprof = shutil.which("msprof")
    if msprof is None:
        raise ValueError(
            "--utilization-profile requires msprof from the matching CANN toolkit in PATH"
        )
    run_dir = create_profile_directory(args.profile_dir)
    utilization_dir = run_dir / "utilization"
    utilization_dir.mkdir(parents=True)
    print(f"Utilization profile output: {utilization_dir}", flush=True)
    try:
        for metric in TASK_UTILIZATION_METRICS:
            print(f"Collecting task PMU replay: {metric}", flush=True)
            subprocess.run(
                _worker_argv(argv, metric, utilization_dir), check=True
            )
        system_dir = utilization_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        system_worker = _worker_argv(argv, "system", utilization_dir)
        msprof_command = [
            msprof,
            f"--output={system_dir.resolve()}",
            f"--sys-devices={args.device}",
            "--ai-core=on",
            "--aic-mode=sample-based",
            "--aic-metrics=PipeUtilization",
            "--aic-freq=100",
            "--sys-hardware-mem=on",
            "--sys-hardware-mem-freq=100",
            "--msproftx=on",
            *system_worker,
        ]
        print("Collecting physical-core and system-memory timeline", flush=True)
        subprocess.run(msprof_command, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Utilization profiler child failed with exit status {error.returncode}: "
            f"{error.cmd}"
        ) from error
    rows, source_files = parse_op_summary_files(utilization_dir)
    core_summaries, _ = parse_physical_core_utilization(system_dir)
    write_utilization_summary(utilization_dir / "utilization_summary.csv", rows)
    print(format_utilization_report(
        rows, core_summaries, utilization_dir, source_files
    ), flush=True)
    return utilization_dir


def run():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.utilization_profile and args._utilization_worker is None:
        return run_utilization_orchestrator(args, sys.argv[1:])
    # Deferred imports keep --help and CPU-only discovery/timing tests lightweight.
    import torch
    import torch_npu

    from scripts.kvtc_calibration_data import Rope
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

    if not torch.npu.is_available():
        raise ValueError("An available NPU and the SGLang Ascend dependencies are required")
    torch.npu.set_device(args.device)
    dump = select_dump(discover_dumps(args.dump_dir, args.tp_worker_name), args.dump_name)
    tokens = selected_token_count(dump.token_count, args.tokens, args.page_size)
    requested = dump.token_count if args.tokens is None else args.tokens
    metadata = []

    def info(message):
        metadata.append(message)
        print(message, flush=True)

    info(f"\nSGLang commit: {git_version()}")
    info(f"Model: {args.model_dir}\nCompression artifact: {args.compression_matrix}")
    info(f"Dump: {dump.name}\nDump directory: {dump.directory}\nTP worker: {args.tp_worker_name}")
    info(
        f"Tokens: original={dump.token_count}, requested={requested}, actual={tokens}, "
        f"dropped by rounding={requested - tokens}"
    )
    info(
        f"Sink tokens: {SINK_TOKENS}; remaining tokens: {tokens - SINK_TOKENS}; "
        f"page size: {args.page_size}"
    )
    if args.profile:
        args.profile_run_dir = create_profile_directory(args.profile_dir)
        info(
            f"Execution: profiling only; captured iterations: {args.profile_iterations}; "
            f"warmups: {args.warmups} (each direction, each mode); "
            f"--iterations={args.iterations} is unused"
        )
        info(f"Profile output: {args.profile_run_dir}")
        info("Profiler: CPU+NPU, Level1, shapes on, memory/stacks off, synchronous export")
    elif args.utilization_profile:
        info(
            f"Execution: reload-only utilization worker={args._utilization_worker}; "
            f"captured iterations: {args.utilization_iterations}; warmups: {args.warmups}; "
            "baseline and --iterations are unused"
        )
        info(f"Utilization output: {args._utilization_run_dir}")
    else:
        info(f"Iterations: {args.iterations}; warmups: {args.warmups} (each direction, each mode)")
    info(f"Enabled modes: {', '.join(enabled_modes(args))}")
    info(
        f"Device: npu:{args.device} ({torch.npu.get_device_name(args.device)}); "
        f"torch={torch.__version__}; torch_npu={torch_npu.__version__}",
    )

    keys, values = load_selected_dump(dump, tokens)
    info(
        f"Selected tensor shape [token, layer, head, head_dim]: {tuple(keys.shape)}; "
        f"dtype={keys.dtype}",
    )
    modes, raw_page_bytes = describe_modes(args, keys.shape, keys.dtype, report=info)
    required = required_host_gb(tokens, args.page_size, modes, raw_page_bytes)
    if args.host_memory_gb is None:
        args.host_memory_gb = required
    elif args.host_memory_gb < required:
        raise ValueError(
            f"This workload requires --host-memory-gb >= {required} "
            "with the production 90/10 split"
        )
    info(f"Host allocation parameter: {args.host_memory_gb} decimal GB (production 90/10 split)")
    info(f"K compression ratio: {args.k_cr}; V compression ratio: {args.v_cr}")
    for mode in modes:
        stored_bytes = (
            raw_page_bytes * (SINK_TOKENS // args.page_size)
            + mode.page_bytes * ((tokens - SINK_TOKENS) // args.page_size)
        )
        info(
            f"  {mode.name}: retained K/V rank={mode.k_rank}/{mode.v_rank}, "
            f"K/V groups={mode.k_groups}/{mode.v_groups}, "
            f"remaining-page bytes={mode.page_bytes}, stored workload bytes={stored_bytes}"
        )
    info(
        "Assumptions: BF16 MHA dump; positions start at 0; 128 sink tokens; "
        "no inference overlap or graph capture.",
    )

    rotary_emb = None
    if any(mode.name != "baseline" for mode in modes):
        Rope.load_model_config(str(args.model_dir))
        rotary_emb = Rope.rotary_emb.to(device="npu")
        if rotary_emb.rotary_dim != keys.shape[-1] or not rotary_emb.is_neox_style:
            raise ValueError("The current hybrid pool requires full-head, NeoX-style RoPE")
        if tokens > rotary_emb.cos_sin_cache.shape[0]:
            raise ValueError("Selected tokens exceed the model's RoPE cache; reduce --tokens")
    _, layers, heads, head_dim = keys.shape
    with torch.inference_mode():
        device_pool = NPUMHATokenToKVPool(
            size=tokens, page_size=args.page_size, dtype=keys.dtype,
            head_num=heads, head_dim=head_dim, layer_num=layers, device="npu",
            enable_memory_saver=False, enable_alt_stream=False,
        )
        if not isinstance(device_pool.k_buffer, torch.Tensor) or device_pool.k_buffer.ndim != 5:
            raise ValueError("Benchmark requires the paged NPU layout; unset ASCEND_USE_FIA")
        results = {}
        validations = []
        for mode in modes:
            results[mode.name], validation = run_mode(
                args, mode, device_pool, keys, values, rotary_emb
            )
            validations.append(validation)
            gc.collect()
            torch.npu.empty_cache()
    if args.utilization_profile:
        print(
            f"Completed utilization worker {args._utilization_worker}: "
            f"{args._utilization_run_dir}",
            flush=True,
        )
    else:
        print_final_report(
            metadata, validations, results, tokens,
            raw_page_bytes * (tokens // args.page_size),
            profiling=args.profile,
        )


if __name__ == "__main__":
    try:
        run()
    except (ValueError, ImportError, EOFError, KeyboardInterrupt) as error:
        raise SystemExit(f"benchmark_kvtc: {error}") from error
