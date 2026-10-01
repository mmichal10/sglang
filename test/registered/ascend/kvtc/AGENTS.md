# KVTC standalone NPU harness

Edit `DEFAULT_CONFIG` in `test_kvtc_harness.py` before running. Set the local
model directory, existing KVTC calibration artifact, worker name, K/V
compression ratios, NPU index, dtype, and quantization mode. The defaults
require no command line arguments. The tree test uses a single process, so its
worker must be `tp_0_pp_0`; keep `quant_disable=False` for the initial quantized
suite.
Run with the active Ascend SGLang environment from the repository root and use
`time` to check total duration:

```bash
time python3 test/registered/ascend/kvtc/test_kvtc_harness.py
```

The script requires a real NPU and artifact. Setup raises an error if either is
missing. It does not initialize a server or load model weights or KV dumps.
`FixtureFactory` owns SGLang object construction and transfer details. Tests
may use `dataclasses.replace(DEFAULT_CONFIG, ...)` for a different setup. When
upstream constructors change after a rebase, update only this factory unless a
behavioral contract has changed. The tree fixture creates a local one-process
Gloo group only when no process group exists.

The factory reads the model config and artifact once per model/artifact/worker
setup. It derives TP/PP sizes from the artifact worker grid, uses SGLang's PP
layer partition and the model's KV head count to determine the local cache
shape, and checks the worker's FP32 PCA tensors and quant schemas before pool
allocation. If calibration used `SGLANG_PP_LAYER_PARTITION`, set the same value
when running this harness.

The fixture contract is a BF16/FP16 paged MHA cache with 128-token pages,
full-head NeoX-style one-dimensional RoPE, and a calibration worker whose
`mu`, `basis`, and quant schemas match `layers * heads * head_dim` and the
configured ratios. Device allocations are page aligned. Synthetic positions
are `0..255`; no dump data is required.

The hybrid pool fixture is shared by the first two tests and released before
the tree test builds its own pool. Tests cover host index range/allocation and
reuse, a mixed 128-sink plus one-page offload/reload with exact sink and finite
compressed values, and selection of the KVTC pool/controller followed by a
256-token tree insert, backup, eviction, match, and reload. Synchronization
precedes data assertions.

Current hybrid sizing splits whole decimal GB into 90% compressed and 10% sink
capacity. A real fixture therefore needs roughly 10 GB of host memory. The
acceptance target is three passing NPU tests in under one minute total, with
each test body taking only seconds. Measure setup and test times on the target
Ascend host; record the device, CANN, PyTorch, and `torch_npu` versions. If
setup misses that target, a later production change may allow smaller pools,
but must preserve the split and update benchmark sizing together.

Missing coverage and implementation routes:

- Baseline and PCA-only: use `replace` with zero ratios or
  `quant_disable=True`, then assert transfer behavior.
- K-only and V-only: set one compression ratio to zero and exercise both
  transfer directions.
- Empty requests: allocate zero sink/compressed lengths and call the pool
  transfer methods with empty index tensors.
- Out-of-order pages: allocate several pages, permute host page IDs and
  matching device/token IDs, then check reload ordering.
- Allocation failure: exhaust one host namespace and verify that a mixed
  allocation fails without leaking the other namespace.
- Repeated tree cycles: loop backup, eviction, match, and reload with fresh
  device data, checking restored prefix and event completion each time.
