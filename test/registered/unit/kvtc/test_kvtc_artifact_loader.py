"""CPU checks for the standalone KVTC calibration artifact loader."""

import runpy
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
QUANT = runpy.run_path(str(ROOT / "python/sglang/srt/mem_cache/kvtc_quant.py"))
KVTCArtifactLoader = QUANT["KVTCArtifactLoader"]
VERSION = QUANT["KVTC_FILE_VERSION"]
WORKER = "tp_0_pp_0"
P = 16


def side(basis_rank=12):
    return {
        "mu": torch.arange(P, dtype=torch.float32),
        "basis": torch.arange(P * basis_rank, dtype=torch.float32).reshape(
            P, basis_rank
        ),
        "quant": {
            "2": [(8, "bfloat16")],
            "4": [(4, "float32")],
        },
    }


class TestKVTCArtifactLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "artifact.pt"
        self.artifact = {
            "version": VERSION,
            "keys": {WORKER: side()},
            "values": {WORKER: side()},
        }

    def load(self, *, k_cr=2, v_cr=4, quant_disable=False):
        torch.save(self.artifact, self.path)
        return KVTCArtifactLoader(
            self.path,
            worker_key=WORKER,
            p=P,
            page_size=128,
            k_cr=k_cr,
            v_cr=v_cr,
            quant_disable=quant_disable,
        )

    def test_quantized_selects_requested_schema_and_trims_basis(self):
        loaded = self.load()
        self.assertEqual(loaded.keys.schema, [(8, "bfloat16")])
        self.assertEqual(loaded.values.schema, [(4, "float32")])
        self.assertEqual(loaded.keys.basis.shape, (P, 8))
        self.assertEqual(loaded.values.basis.shape, (P, 4))
        self.assertEqual(loaded.keys.source_basis_rank, 12)
        self.assertEqual(loaded.keys.mu.shape, (P,))
        self.assertEqual(loaded.keys.mu.device.type, "cpu")
        self.assertEqual(loaded.keys.basis.device.type, "cpu")
        torch.testing.assert_close(
            loaded.keys.basis, self.artifact["keys"][WORKER]["basis"][:, :8]
        )
        self.assertEqual(self.load(k_cr=2.0, v_cr=0).keys.basis.shape, (P, 8))

    def test_one_side_can_be_disabled(self):
        self.artifact.pop("keys")
        loaded = self.load(k_cr=0, v_cr=4)
        self.assertIsNone(loaded.keys)
        self.assertEqual(loaded.values.basis.shape, (P, 4))

        self.artifact["keys"] = {WORKER: side()}
        self.artifact.pop("values")
        loaded = self.load(k_cr=2, v_cr=0)
        self.assertEqual(loaded.keys.basis.shape, (P, 8))
        self.assertIsNone(loaded.values)

    def test_pca_only_uses_ratio_without_quant_schema(self):
        self.artifact["keys"][WORKER].pop("quant")
        self.artifact["values"][WORKER].pop("quant")
        loaded = self.load(k_cr=2, v_cr=4, quant_disable=True)
        self.assertIsNone(loaded.keys.schema)
        self.assertIsNone(loaded.values.schema)
        self.assertEqual(loaded.keys.basis.shape, (P, 8))
        self.assertEqual(loaded.values.basis.shape, (P, 4))

    def test_rejects_missing_or_invalid_schema(self):
        with self.assertRaisesRegex(ValueError, "K/tp_0_pp_0.*quant\\['3'\\]"):
            self.load(k_cr=3, v_cr=0)
        self.artifact["keys"][WORKER]["quant"]["2"] = [(7, "int4")]
        with self.assertRaisesRegex(ValueError, "K/tp_0_pp_0.*int4"):
            self.load(k_cr=2, v_cr=0)
        self.artifact["keys"][WORKER]["quant"]["2"] = [(16, "bfloat16")]
        with self.assertRaisesRegex(ValueError, "basis rank is only 12"):
            self.load(k_cr=2, v_cr=0)

    def test_rejects_missing_worker_and_bad_tensors(self):
        self.artifact["keys"] = {"tp_1_pp_0": side()}
        with self.assertRaisesRegex(ValueError, "missing keys/tp_0_pp_0"):
            self.load(k_cr=2, v_cr=0)

        self.artifact["keys"] = {WORKER: side()}
        self.artifact["keys"][WORKER]["mu"] = torch.zeros(P, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "K/tp_0_pp_0/mu"):
            self.load(k_cr=2, v_cr=0)

        self.artifact["keys"][WORKER]["mu"] = torch.zeros(P, dtype=torch.float32)
        self.artifact["keys"][WORKER]["basis"] = torch.zeros(P + 1, 12)
        with self.assertRaisesRegex(ValueError, "K/tp_0_pp_0/basis"):
            self.load(k_cr=2, v_cr=0)

        self.artifact["keys"][WORKER]["basis"] = torch.zeros(
            P, 12, dtype=torch.float16
        )
        with self.assertRaisesRegex(ValueError, "K/tp_0_pp_0/basis"):
            self.load(k_cr=2, v_cr=0)

    def test_rejects_invalid_pca_rank(self):
        self.artifact["keys"][WORKER] = side(basis_rank=3)
        with self.assertRaisesRegex(ValueError, "shorter than the retained rank 8"):
            self.load(k_cr=2, v_cr=0, quant_disable=True)
        with self.assertRaisesRegex(ValueError, "retains no features"):
            self.load(k_cr=17, v_cr=0, quant_disable=True)

    def test_rejects_invalid_version_or_file(self):
        self.artifact = ["not a dictionary"]
        with self.assertRaisesRegex(ValueError, "must contain a dictionary"):
            self.load(k_cr=2, v_cr=0)
        self.artifact = {"version": VERSION, "keys": {WORKER: side()}}
        self.artifact["version"] = "older"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            self.load(k_cr=2, v_cr=0)
        self.artifact.pop("version")
        with self.assertRaisesRegex(ValueError, "found '<missing>'"):
            self.load(k_cr=2, v_cr=0)
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            KVTCArtifactLoader(
                self.path.with_name("missing.pt"),
                worker_key=WORKER,
                p=P,
                page_size=128,
                k_cr=2,
                v_cr=0,
                quant_disable=False,
            )

    def test_rejects_invalid_ratios_and_dimensions(self):
        for ratio in (-1, 1.5, True, float("nan"), float("inf"), "2"):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "compression ratio"):
                    self.load(k_cr=ratio, v_cr=0)

        torch.save(self.artifact, self.path)
        for argument, value in (("p", 0), ("page_size", 0)):
            kwargs = dict(
                worker_key=WORKER,
                p=P,
                page_size=128,
                k_cr=2,
                v_cr=0,
                quant_disable=False,
            )
            kwargs[argument] = value
            with self.subTest(argument=argument):
                with self.assertRaises(ValueError):
                    KVTCArtifactLoader(self.path, **kwargs)


if __name__ == "__main__":
    unittest.main()
