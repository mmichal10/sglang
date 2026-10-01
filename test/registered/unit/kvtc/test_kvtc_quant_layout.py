"""CPU checks for the dtype-grouped KVTC quantizer layout."""

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


class TestKVTCQuantGroupedLayout(unittest.TestCase):
    def test_calibration_import_uses_runtime_builder(self):
        self.assertIs(
            QUANT["build_quant_layout"], QUANT["KVTCQuantizer"].build_layout
        )

    def build_new(self, schema, *, page_size, basis_rank):
        npu_ops = types.ModuleType("torch_npu")
        npu_ops.npu_dynamic_quant_asymmetric = Mock()
        npu_ops.npu_anti_quant = Mock()
        with patch.dict(sys.modules, {"torch_npu": npu_ops}):
            quantizer = QUANT["KVTCQuantizer"](
                keys_schema=schema,
                values_schema=None,
                keys_basis_rank=basis_rank,
                values_basis_rank=None,
                artifact_path="test.pt",
                page_size=page_size,
                device="cpu",
                cache_dtype=torch.bfloat16,
                staging_capacity_pages=1,
            )
        return quantizer._keys.layout

    def build_both(self, schema, *, page_size=128, basis_rank=64):
        return (
            QUANT["KVTCQuantizer"].build_layout(
                schema, page_size=page_size, basis_rank=basis_rank, matrix_name="keys"
            ),
            self.build_new(schema, page_size=page_size, basis_rank=basis_rank),
        )

    def test_interleaved_dtypes_keep_feature_and_metadata_positions(self):
        schema = [
            (3, "float32"),
            (8, "int4"),
            (5, "int8"),
            (16, "int4"),
            (2, "bfloat16"),
            (4, "int8"),
        ]
        layout, runtime_layout = self.build_both(schema)

        self.assertEqual(layout, runtime_layout)
        self.assertEqual(list(layout.direct_storage_groups), ["float32", "bfloat16"])
        self.assertEqual(list(layout.integer_quant_groups), ["int8", "int4"])
        self.assertEqual(
            {
                dtype: [group.feature_start for group in groups]
                for groups_by_dtype in (
                    layout.direct_storage_groups,
                    layout.integer_quant_groups,
                )
                for dtype, groups in groups_by_dtype.items()
            },
            {
                "float32": [0],
                "bfloat16": [32],
                "int8": [11, 34],
                "int4": [3, 16],
            },
        )
        self.assertEqual(layout.feature_count, 38)
        self.assertEqual(layout.metadata_count, 4)
        self.assertEqual(
            layout.payload_elements,
            {"float32": 384, "bfloat16": 256, "int8": 1152, "int4": 384},
        )
        self.assertEqual(layout.bytes_per_token, 53)
        self.assertEqual(layout.group_count, len(schema))
        self.assertEqual(
            sorted(
                (
                    group
                    for groups_by_dtype in (
                        layout.direct_storage_groups,
                        layout.integer_quant_groups,
                    )
                    for groups in groups_by_dtype.values()
                    for group in groups
                ),
                key=lambda group: group.feature_start,
            ),
            [
                QUANT["KVTCQuantGroup"](0, 3, "float32", 0, 384, None),
                QUANT["KVTCQuantGroup"](3, 11, "int4", 0, 128, 0),
                QUANT["KVTCQuantGroup"](11, 16, "int8", 0, 640, 1),
                QUANT["KVTCQuantGroup"](16, 32, "int4", 128, 384, 2),
                QUANT["KVTCQuantGroup"](32, 34, "bfloat16", 0, 256, None),
                QUANT["KVTCQuantGroup"](34, 38, "int8", 640, 1152, 3),
            ],
        )

    def test_single_dtype_keeps_separate_groups(self):
        layout, runtime_layout = self.build_both(
            [(5, "int8"), (7, "int8")], page_size=2
        )
        self.assertEqual(layout, runtime_layout)
        self.assertEqual(layout.direct_storage_groups, {})
        self.assertEqual(list(layout.integer_quant_groups), ["int8"])
        self.assertEqual(len(layout.integer_quant_groups["int8"]), 2)
        self.assertEqual(layout.payload_elements, {"int8": 24})
        self.assertEqual(layout.metadata_count, 2)
        self.assertEqual(layout.bytes_per_token, 20)

    def test_float_groups_need_no_metadata(self):
        layout, runtime_layout = self.build_both([(3, "bfloat16"), (2, "float32")])
        self.assertEqual(layout, runtime_layout)
        self.assertEqual(layout.metadata_count, 0)
        self.assertEqual(layout.integer_quant_groups, {})
        self.assertEqual(
            [
                group.metadata_index
                for groups in layout.direct_storage_groups.values()
                for group in groups
            ],
            [None, None],
        )
        self.assertEqual(
            layout.payload_elements, {"bfloat16": 384, "float32": 256}
        )
        self.assertEqual(layout.bytes_per_token, 14)

    def test_validation_matches_runtime_quantizer(self):
        invalid = [
            ([], 64),
            ([(3, "float32", "extra")], 64),
            ([(True, "int8")], 64),
            ([(0, "int8")], 64),
            ([(3, "unsupported")], 64),
            ([(7, "int4")], 64),
            ([(10, "int4")], 64),
            ([(16, "int4")], 8),
        ]
        for schema, basis_rank in invalid:
            with self.subTest(schema=schema, basis_rank=basis_rank):
                with self.assertRaises(ValueError) as layout_error:
                    QUANT["KVTCQuantizer"].build_layout(
                        schema,
                        page_size=128,
                        basis_rank=basis_rank,
                        matrix_name="keys",
                    )
                with self.assertRaises(ValueError) as runtime_error:
                    self.build_new(schema, page_size=128, basis_rank=basis_rank)
                self.assertEqual(str(layout_error.exception), str(runtime_error.exception))


if __name__ == "__main__":
    unittest.main()
