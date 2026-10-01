# KVTC development guide for agents

This repository contains an in-progress KVTC implementation on top of SGLang
HiCache. Use this guide for KVTC changes only; for unrelated work, follow the
normal SGLang conventions and any more specific `AGENTS.md` in the target tree.

## Start here

- Work from the repository root and inspect `git status --short` before editing.
  KVTC development commonly happens across several local branches and may have
  untracked experiments. Never overwrite or clean up unrelated user work.
- Search for both `KVTC` and `kvtc`. The implementation is intentionally spread
  across runtime cache code, scheduler/controller integration, calibration,
  benchmarking, and tests.
- Read the narrowest relevant files from the map below before changing code.
  Do not scan all of `memory_pool_host.py` unless the task truly spans other host
  pool implementations.
- Treat current code and artifacts as the source of truth. Do not infer current
  flags, tensor layouts, or profiler behavior from old commits or branch names.

## Source map

| Area | Primary files |
| --- | --- |
| Quantization schema and byte accounting | `python/sglang/srt/mem_cache/kvtc_quant.py` |
| PCA/quantized host storage and hybrid routing | `python/sglang/srt/mem_cache/memory_pool_host.py`; start at `KVTCHostMemoryRequest`, `NPUMHATokenToKVPoolCompressed`, and `NPUMHATokenToKVPoolHybrid` |
| Async offload/reload orchestration | `python/sglang/srt/managers/cache_controller.py`; start at `KVTCCacheOperation` and `KVTCHiCacheController` |
| HiCache/radix integration | `python/sglang/srt/mem_cache/hiradix_cache.py`, `python/sglang/srt/mem_cache/radix_cache.py` |
| Configuration and cache construction | `python/sglang/srt/server_args.py`, `python/sglang/srt/mem_cache/cache_init_params.py`, `python/sglang/srt/mem_cache/kv_cache_builder.py` |
| RoPE inversion/reapplication | `python/sglang/srt/layers/rotary_embedding/base.py` |
| Prefix sliding-window behavior | `python/sglang/srt/managers/schedule_batch.py` |
| KV dump production | `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` and `--dump-kv-path` |
| Calibration/data discovery | `scripts/kvtc_calibrate.py`, `scripts/kvtc_calibration_data.py`, `scripts/kvtc_calibration_quant.py` |
| Reconstruction parity tool | `scripts/kvtc_compare_reconstruction.py` |
| Isolated production-pool benchmark/profiling | `scripts/benchmark_kvtc.py` |
| Focused tests | `test/registered/unit/test_benchmark_kvtc.py`, `test/registered/unit/kvtc/`, `test/registered/ascend/kvtc/` |

## Current design

KVTC is currently an Ascend NPU, paged MHA HiCache path. Supplying
`--hicache-kvtc-params` selects `NPUMHATokenToKVPoolHybrid` and
`KVTCHiCacheController`. It does not apply to MLA/DSA pools and it does not
support a storage tier.

The hybrid host pool reserves 90% of the configured host size for compressed
pages and 10% for uncompressed sink pages. Public host indices below
`sink_token_shift` belong to the compressed pool; sink indices are shifted by
that boundary. Preserve that namespace when changing allocation, free, or
controller code.

The controller treats the initial contiguous tokens `0..127` as attention sinks
when they are present and routes them uncompressed. Remaining full pages are
sent to the compressed pool. Operations may be merged. The controller invokes
reload per layer, but the compressed pool performs its all-layer PCA restore on
`layer_id == 0`; the sink pool follows the normal per-layer path. Keep index
ordering and tensor lifetimes valid across the dedicated load and write streams;
the `record_stream` calls and layer completion events are correctness mechanisms,
not cosmetic cleanup.

Compressed offload/reload is:

1. Read one full page across all layers/heads.
2. For K, undo RoPE using the absolute token positions. V is not rotated.
3. Flatten each token to `p = layers * heads * head_dim`, subtract the calibrated
   mean, and project onto the retained PCA basis.
4. Store the projection either in the base FP16/BF16 dtype (PCA-only mode) or in
   calibrated mixed-dtype groups (quant mode).
5. On reload, reverse quantization and PCA, reapply RoPE to K, and write the
   requested layer into the device cache.

K and V compression are independently enabled by their positive compression
ratios and artifact entries. `--hicache-kvtc-quant-disable` means PCA-only
compressed storage; it does not disable KVTC compression. The benchmark names
the three useful comparisons `quant`, `compressed`, and `baseline`; baseline
still uses the production hybrid routing but has no PCA or quantization.

## Artifact and tensor contracts

- Dump files live under `tp_<tp>_pp_<pp>/` and are named
  `<request>-K|V-chunk_<chunk>-layer_<layer>.bin`.
- Loaded dumps have logical shape `[token, layer, head, head_dim]`. Validate
  worker sets, K/V token counts, chunks, and layers instead of relying on file
  system or lexicographic order.
- The current calibration version is defined by `KVTC_FILE_VERSION` in
  `scripts/kvtc_calibration_quant.py`; never duplicate its literal in new code.
- The artifact contains `keys` and `values`, then worker keys such as
  `tp_0_pp_0`, then FP32 `mu`, FP32 `basis`, and `quant` schemas keyed by the
  integer compression ratio rendered as a string.
- `mu` has shape `[p]`; `basis` has shape `[p, rank]`. Runtime retains a prefix of
  basis columns. A quant schema is an ordered list of `(feature_count,
  dtype_name)` groups covering that retained prefix.
- Supported schema dtypes are `float32`, `bfloat16`, `int8`, and packed `int4`.
  INT4 group sizes must be at least 8 and divisible by 8. It stores eight values
  per `torch.int32` payload element.
- Each INT8/INT4 group has a per-token FP16 scale and offset. Runtime stores the
  negative of the dynamic quantizer offset because `npu_anti_quant` reconstructs
  with `(q + offset) * scale`. Preserve this sign convention.
- Compression budgets include payload plus scale/offset metadata. Use
  `quant_group_bits()` and `build_quant_layout()` for all accounting and
  validation; do not reproduce their arithmetic elsewhere.
- Runtime compression ratios must resolve to positive integers. In PCA-only
  mode the retained rank is `p // ratio`; in quant mode it comes from the schema.

RoPE is part of the data contract. Dumps contain rotated K values, while PCA is
fit in unrotated space. The present production benchmark additionally requires
full-head, NeoX-style, one-dimensional RoPE. Calibration rejects mRoPE because
the dumps do not carry multimodal position IDs. Any model-support expansion must
update runtime, calibration, reconstruction tests, and benchmark validation
together.

Several constants exist at different pipeline stages. In particular, the
runtime/benchmark sink boundary and calibration sampling exclusions are defined
in different files, and `--hicache-kvtc-sw` changes prefix matching rather than
the sink-pool split. Verify every use before changing or consolidating them; do
not assume identically named concepts currently have one shared value.

## Performance-sensitive rules

- Quantization and dequantization are page-batched and currently bounded by
  `_QUANT_BATCH_MAX_PAGES`. Preserve page order and the per-page storage layout.
- Host payloads and metadata are pinned. Dequant payload staging buffers are
  reused and keyed to their layout/source buffers. Avoid reintroducing hot-path
  allocations or one-device-transfer-per-group behavior without measurement.
- Host page IDs used for scatter/gather are CPU `int64`. Quantized payloads are
  grouped by storage dtype, while reconstructed features must return to their
  original schema positions.
- The packed INT4 anti-quant input sometimes must be materialized when it has a
  nonzero storage offset. Keep the existing guard unless an Ascend kernel test
  proves it unnecessary.
- Do not replace nonblocking copies, stream placement, or synchronization based
  only on CPU tests. Validate such changes on the same Ascend/CANN environment
  used for production.

## Profiling and benchmark semantics

- `_profile_kvtc` is false in normal execution. When false, no
  `torch.profiler.record_function` context should be created. Keep profiled and
  unprofiled branches operation-equivalent, and restore both the hybrid and
  compressed-pool flags in `finally` paths.
- Treat existing `kvtc/...` range names as an analysis interface. Update tests
  and report parsing if labels change.
- CPU range duration is not NPU kernel duration. Stage ranges identify CPU
  scopes and correlated NPU work; use exported operator summaries for device
  claims.
- Benchmark setup and reconstruction validation are untimed. Timing includes
  Python and device work with synchronization at measurement boundaries.
- Reported logical GiB/s uses original uncompressed K+V bytes and is not physical
  bus bandwidth. The three benchmark modes may retain different PCA ranks, so
  their result is an end-to-end production-mode comparison, not an isolated
  quantizer comparison.
- `--utilization-profile` is reload-only, excludes baseline, and uses separate
  PMU replay processes. Do not merge counters from different replays as though
  they came from one execution.

## Validation workflow

Use the Python executable from the active SGLang/Ascend environment. The system
Python may not contain PyTorch or `torch_npu`.

For argument parsing, dump discovery, report formatting, and profiling-control
changes, run the CPU-light benchmark suite:

```bash
PYTHONPATH=test/registered/unit python3 -m unittest test_benchmark_kvtc
```

For quant packing/dequant batching changes, run:

```bash
python3 -m unittest discover -s test/registered/unit/kvtc \
  -p 'test_kvtc_quant_batch.py'
```

These tests use NPU-operator stand-ins. They do not validate actual Ascend
numerics, asynchronous copies, streams, or performance. For runtime changes,
also run the relevant test under `test/registered/ascend/kvtc/` in the target
Ascend environment and state the model, device, CANN, PyTorch, and `torch_npu`
versions in the result.

For benchmark smoke testing, use real model configuration, calibration artifact,
and dump paths. A short profiling run is:

```bash
python3 scripts/benchmark_kvtc.py \
  --model-dir MODEL_DIR \
  --compression-matrix KVTC_ARTIFACT \
  --dump-dir DUMP_DIR \
  --k-cr 8 --v-cr 8 --tokens 256 \
  --profile --warmups 1 --profile-iterations 1
```

Before handoff, run focused tests first, then formatting/lint on only the touched
files when available. Report skipped NPU validation explicitly; never describe a
CPU stand-in test as end-to-end KVTC validation.

## Change checklist

- Preserve the non-KVTC HiCache path and behavior when no calibration path is
  supplied.
- Test K-only, V-only, both-enabled, PCA-only, quantized, and empty-page cases
  when the changed logic can affect them.
- Test noncontiguous/out-of-order host page IDs for packing or gathering changes.
- Keep artifact writers, readers, runtime layout, reconstruction tooling, and
  versioning in sync. Bump `KVTC_FILE_VERSION` for incompatible formats.
- Check page-byte accounting and the effective compression ratio after storage
  changes, including quant metadata.
- Check exact sink restoration and finite compressed reconstruction before
  trusting latency numbers.
- Keep debug logging bounded; KV pages and calibration tensors are large.
- Avoid drive-by refactors in shared SGLang cache/controller code. KVTC branches
  often carry upstream divergence, so small reviewable changes rebase better.
