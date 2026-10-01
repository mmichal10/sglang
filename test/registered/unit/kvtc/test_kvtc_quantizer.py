"""CPU operator-stand-in checks for the standalone KVTC quantizer."""

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[4]
QUANT = runpy.run_path(
    str(ROOT / "python/sglang/srt/mem_cache/kvtc_quant.py")
)
KVTCQuantizer = QUANT["KVTCQuantizer"]


def dynamic_quant(values, *, dst_type):
    values = values.float()
    int4 = dst_type == torch.quint4x2
    low, high = (-8, 7) if int4 else (-128, 127)
    minimum, maximum = values.amin(dim=-1), values.amax(dim=-1)
    scale = ((maximum - minimum) / (high - low)).clamp_min(1e-8)
    offset = low - minimum / scale
    quantized = (values / scale[:, None] + offset[:, None]).round().clamp(low, high)
    if int4:
        nibbles = (quantized.to(torch.int64) & 15).reshape(values.shape[0], -1, 8)
        quantized = (nibbles << (torch.arange(8) * 4)).sum(dim=-1).to(torch.int32)
    else:
        quantized = quantized.to(torch.int8)
    return quantized, scale, offset


def anti_quant(quantized, scale, *, offset, dst_dtype, **kwargs):
    if quantized.dtype == torch.int32:
        nibbles = (
            quantized.to(torch.int64).unsqueeze(-1) >> (torch.arange(8) * 4)
        ) & 15
        quantized = torch.where(nibbles >= 8, nibbles - 16, nibbles).reshape(1, -1)
    return ((quantized.float() + offset) * scale).to(dst_dtype)


def host_buffers(layout, page_count=8):
    payloads = {
        dtype_name: torch.full(
            (page_count, elements),
            -1,
            dtype=QUANT["KVTC_QUANT_STORAGE_DTYPES"][dtype_name],
        )
        for dtype_name, elements in layout.payload_elements.items()
    }
    metadata_shape = (page_count, 2, layout.metadata_count)
    return (
        payloads,
        torch.full(metadata_shape, -1, dtype=torch.float16),
        torch.full(metadata_shape, -1, dtype=torch.float16),
    )


def layout_groups(layout):
    for groups_by_dtype in (
        layout.direct_storage_groups,
        layout.integer_quant_groups,
    ):
        for dtype_name, groups in groups_by_dtype.items():
            for group in groups:
                yield dtype_name, group


def reference_pages(source, host_pages, layout, target, cache_dtype):
    payloads, scales, offsets = target
    for page, host_page in zip(source, host_pages):
        for dtype_name, group in layout_groups(layout):
            values = page[:, group.feature_start : group.feature_end]
            if group.metadata_index is None:
                quantized = values.to(QUANT["KVTC_QUANT_STORAGE_DTYPES"][dtype_name])
            else:
                dst_type = torch.quint4x2 if dtype_name == "int4" else torch.int8
                quantized, scale, offset = dynamic_quant(
                    values.to(cache_dtype), dst_type=dst_type
                )
                scales[host_page, :, group.metadata_index] = scale
                offsets[host_page, :, group.metadata_index] = -offset
            payloads[dtype_name][
                host_page, group.payload_start : group.payload_end
            ] = quantized.flatten()


def reference_restore(host_pages, layout, source, cache_dtype):
    payloads, scales, offsets = source
    result = torch.empty((len(host_pages), 2, layout.feature_count))
    for output_page, host_page in enumerate(host_pages):
        for dtype_name, group in layout_groups(layout):
            width = group.feature_end - group.feature_start
            payload = payloads[dtype_name][
                host_page, group.payload_start : group.payload_end
            ]
            if group.metadata_index is None:
                values = payload.reshape(2, width)
            else:
                scale = scales[host_page, :, group.metadata_index]
                offset = offsets[host_page, :, group.metadata_index]
                expanded_scale = scale.repeat_interleave(width).float()
                expanded_offset = offset.repeat_interleave(width).float()
                values = anti_quant(
                    payload.reshape(1, -1),
                    expanded_scale,
                    offset=expanded_offset,
                    dst_dtype=cache_dtype,
                ).reshape(2, width)
            result[output_page, :, group.feature_start : group.feature_end] = values
    return result


class TestKVTCQuantizer(unittest.TestCase):
    def setUp(self):
        self.npu_ops = types.ModuleType("torch_npu")
        self.npu_ops.npu_dynamic_quant_asymmetric = Mock(side_effect=dynamic_quant)
        self.npu_ops.npu_anti_quant = Mock(side_effect=anti_quant)
        self.module_patch = patch.dict(sys.modules, {"torch_npu": self.npu_ops})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_quantizer(self, keys_schema=None, values_schema=None, capacity=3):
        return KVTCQuantizer(
            keys_schema=keys_schema,
            values_schema=values_schema,
            keys_basis_rank=64 if keys_schema is not None else None,
            values_basis_rank=64 if values_schema is not None else None,
            artifact_path="/calibration/test.pt",
            page_size=2,
            device="cpu",
            cache_dtype=torch.bfloat16,
            staging_capacity_pages=capacity,
        )

    def test_disabled_side_fails_fast_even_for_empty_pages(self):
        quantizer = self.make_quantizer(keys_schema=[(8, "int4")])
        self.assertEqual(quantizer.key_bytes_per_token(), 8)
        with self.assertRaisesRegex(
            RuntimeError, "values quantization is not initialized"
        ):
            quantizer.value_bytes_per_token()
        with self.assertRaisesRegex(
            RuntimeError, "values quantization is not initialized"
        ):
            quantizer.value_layout()
        empty_ids = torch.empty(0, dtype=torch.int64)
        empty_pages = torch.empty(0, 2, 8)
        with self.assertRaisesRegex(
            RuntimeError, "values quantization is not initialized"
        ):
            quantizer.quantize_pages_values(empty_pages, empty_ids, {}, None, None)
        with self.assertRaisesRegex(
            RuntimeError, "values quantization is not initialized"
        ):
            quantizer.dequantize_pages_values(empty_ids, {}, None, None)
        buffers = host_buffers(quantizer.key_layout())
        quantizer.quantize_pages_keys(empty_pages, empty_ids, *buffers)
        self.assertEqual(
            tuple(quantizer.dequantize_pages_keys(empty_ids, *buffers).shape),
            (0, 2, 8),
        )

    def test_float_only_values_need_no_npu_operator(self):
        with patch.dict(sys.modules, {"torch_npu": None}):
            quantizer = self.make_quantizer(
                values_schema=[(3, "float32"), (2, "bfloat16")]
            )
        self.assertEqual(quantizer.value_bytes_per_token(), 16)
        self.assertEqual(quantizer.value_layout().metadata_count, 0)
        self.assertIsNone(quantizer._keys)
        self.assertEqual(set(quantizer._values.staging), {"float32", "bfloat16"})
        source = torch.randn(1, 2, 5)
        host_pages = torch.tensor([4], dtype=torch.int64)
        buffers = host_buffers(quantizer.value_layout())
        quantizer.quantize_pages_values(source, host_pages, *buffers)
        restored = quantizer.dequantize_pages_values(host_pages, *buffers)
        torch.testing.assert_close(restored[:, :, :3], source[:, :, :3], rtol=0, atol=0)
        torch.testing.assert_close(
            restored[:, :, 3:], source[:, :, 3:].to(torch.bfloat16).float(),
            rtol=0, atol=0,
        )

    def test_constructor_requires_at_least_one_schema(self):
        with self.assertRaisesRegex(
            ValueError, "requires a keys or values schema"
        ):
            self.make_quantizer()

    def test_mixed_sides_match_reference_and_preserve_host_order(self):
        keys_schema = [
            (8, "int4"),
            (3, "float32"),
            (5, "int8"),
            (16, "int4"),
            (2, "bfloat16"),
            (1, "float32"),
        ]
        values_schema = [(4, "int8")]
        quantizer = self.make_quantizer(keys_schema, values_schema)
        self.assertEqual(quantizer.key_bytes_per_token(), 8 + 12 + 9 + 12 + 4 + 4)
        self.assertEqual(quantizer.value_bytes_per_token(), 8)
        ids = torch.tensor([5, 1, 3], dtype=torch.int64)

        for name, schema in (("keys", keys_schema), ("values", values_schema)):
            layout = (
                quantizer.key_layout()
                if name == "keys"
                else quantizer.value_layout()
            )
            source = torch.randn(3, 2, layout.feature_count)
            actual = host_buffers(layout)
            expected = host_buffers(layout)
            reference_pages(source, ids, layout, expected, torch.bfloat16)
            getattr(quantizer, f"quantize_pages_{name}")(source, ids, *actual)
            for dtype_name in layout.payload_elements:
                torch.testing.assert_close(
                    actual[0][dtype_name], expected[0][dtype_name], rtol=0, atol=0
                )
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
            restored = getattr(quantizer, f"dequantize_pages_{name}")(
                ids, *actual
            )
            torch.testing.assert_close(
                restored,
                reference_restore(ids, layout, expected, torch.bfloat16),
                rtol=0,
                atol=0,
            )

        self.assertEqual(self.npu_ops.npu_dynamic_quant_asymmetric.call_count, 4)
        self.assertEqual(self.npu_ops.npu_anti_quant.call_count, 3)
        self.assertIsNot(quantizer._keys.staging, quantizer._values.staging)

    def test_capacity_validation_and_logging(self):
        with self.assertLogs(QUANT["logger"], level="DEBUG") as captured:
            quantizer = self.make_quantizer(keys_schema=[(4, "int8")], capacity=1)
            ids = torch.tensor([5], dtype=torch.int64)
            buffers = host_buffers(quantizer._keys.layout)
            quantizer.quantize_pages_keys(torch.randn(1, 2, 4), ids, *buffers)
            quantizer.dequantize_pages_keys(ids, *buffers)
        self.assertIn("/calibration/test.pt", " ".join(captured.output))
        self.assertIn("schema=[(4, 'int8')]", " ".join(captured.output))
        self.assertIn("host_page_ids=[5]", " ".join(captured.output))
        with self.assertRaisesRegex(ValueError, "staging capacity is 1"):
            quantizer.dequantize_pages_keys(
                torch.tensor([5, 1], dtype=torch.int64), *buffers
            )

    def test_constructor_rejects_missing_rank_and_missing_operator(self):
        with self.assertRaisesRegex(ValueError, "schema requires a basis rank"):
            KVTCQuantizer(
                keys_schema=[(4, "int8")],
                values_schema=None,
                keys_basis_rank=None,
                values_basis_rank=None,
                artifact_path="x",
                page_size=2,
                device="cpu",
                cache_dtype=torch.bfloat16,
                staging_capacity_pages=1,
            )
        del self.npu_ops.npu_anti_quant
        with self.assertRaisesRegex(RuntimeError, "npu_anti_quant"):
            self.make_quantizer(keys_schema=[(4, "int8")])


if __name__ == "__main__":
    unittest.main()
