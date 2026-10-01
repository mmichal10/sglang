"""Isolated real-NPU check for the standalone KVTC quantizer.

Run from the repository root with the active Ascend environment:
python3 test/registered/ascend/kvtc/test_kvtc_quantizer.py
"""

import sys
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers torch.npu


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.kvtc_quant import (  # noqa: E402
    KVTCQuantizer,
    KVTC_QUANT_METADATA_DTYPE,
    KVTC_QUANT_STORAGE_DTYPES,
)


def host_buffers(layout, page_count):
    payloads = {
        dtype_name: torch.empty(
            (page_count, elements),
            dtype=KVTC_QUANT_STORAGE_DTYPES[dtype_name],
            device="cpu",
            pin_memory=True,
        )
        for dtype_name, elements in layout.payload_elements.items()
    }
    shape = (page_count, 128, layout.metadata_count)
    scales = torch.empty(
        shape, dtype=KVTC_QUANT_METADATA_DTYPE, device="cpu", pin_memory=True
    )
    offsets = torch.empty_like(scales)
    return payloads, scales, offsets


class TestKVTCQuantizerNPU(unittest.TestCase):
    def test_mixed_kv_roundtrip_with_out_of_order_host_pages(self):
        if not torch.npu.is_available():
            self.fail("An available Ascend NPU is required")
        device = torch.device("npu:0")
        quantizer = KVTCQuantizer(
            keys_schema=[
                (8, "int4"), (4, "bfloat16"), (8, "int8"), (8, "int4")
            ],
            values_schema=[(8, "int8")],
            keys_basis_rank=28,
            values_basis_rank=8,
            artifact_path="<synthetic>",
            page_size=128,
            device=device,
            cache_dtype=torch.bfloat16,
            staging_capacity_pages=2,
        )
        host_pages = torch.tensor([3, 0], dtype=torch.int64)
        k_source = torch.randn(2, 128, 28, device=device)
        v_source = torch.randn(2, 128, 8, device=device)
        k_host = host_buffers(quantizer.key_layout(), page_count=4)
        v_host = host_buffers(quantizer.value_layout(), page_count=4)

        quantizer.quantize_pages_keys(k_source, host_pages, *k_host)
        quantizer.quantize_pages_values(v_source, host_pages, *v_host)
        torch.npu.synchronize()
        restored_k = quantizer.dequantize_pages_keys(host_pages, *k_host)
        restored_v = quantizer.dequantize_pages_values(host_pages, *v_host)
        torch.npu.synchronize()

        self.assertTrue(torch.isfinite(restored_k).all().item())
        self.assertTrue(torch.isfinite(restored_v).all().item())
        torch.testing.assert_close(
            restored_k[:, :, 8:12],
            k_source[:, :, 8:12].to(torch.bfloat16).to(torch.float32),
            rtol=0,
            atol=0,
        )
        self.assertLess(
            (restored_k - k_source).abs().mean().item(), 0.3
        )
        self.assertLess(
            (restored_v - v_source).abs().mean().item(), 0.1
        )


if __name__ == "__main__":
    unittest.main()
