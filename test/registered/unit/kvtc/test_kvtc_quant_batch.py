"""CPU checks that the compressed pool batches and routes quantizer calls."""

import ast
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "python/sglang/srt/mem_cache"
QUANT = runpy.run_path(str(CACHE / "kvtc_quant.py"))


def load_pool():
    path = CACHE / "memory_pool_host.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "NPUMHATokenToKVPoolCompressed"
    )
    cls.bases = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    namespace = {**QUANT, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls.name]


Pool = load_pool()


class TestKVTCQuantBatch(unittest.TestCase):
    def make_pool(self):
        pool = Pool.__new__(Pool)
        pool.page_size = 2
        pool._QUANT_BATCH_MAX_PAGES = 2
        pool.device = "cpu"
        pool.dtype = torch.float32
        pool.layer_num = 1
        pool.head_num = 1
        pool.head_dim = 4
        pool.device_page_shape = (1, 2, 1, 4)
        pool.k_kvtc = pool.v_kvtc = True
        pool.kvtc_quant_disable = False
        pool.kvtc_k_mu = pool.kvtc_v_mu = torch.zeros(4)
        pool.kvtc_k_V = pool.kvtc_v_V = torch.eye(4)
        pool.rotary_emb = SimpleNamespace(
            invert_native_keys_batch=Mock(side_effect=lambda _, values: values),
            forward_native_keys_batch=Mock(side_effect=lambda _, values: values),
        )
        pool.quantizer = SimpleNamespace(
            quantize_pages_keys=Mock(),
            quantize_pages_values=Mock(),
            dequantize_pages_keys=Mock(),
            dequantize_pages_values=Mock(),
        )
        for side in ("k", "v"):
            setattr(pool, f"{side}_quant_buffers", {})
            setattr(pool, f"{side}_quant_scales", torch.empty(0))
            setattr(pool, f"{side}_quant_offsets", torch.empty(0))
        return pool

    @staticmethod
    def page_indices(page_ids):
        return (page_ids[:, None] * 2 + torch.arange(2)).flatten()

    def test_backup_dispatches_ordered_batches_and_host_buffers(self):
        pool = self.make_pool()
        quantize_order = []
        pool.quantizer.quantize_pages_keys.side_effect = (
            lambda _, host_pages, *args: quantize_order.append(("k", host_pages.tolist()))
        )
        pool.quantizer.quantize_pages_values.side_effect = (
            lambda _, host_pages, *args: quantize_order.append(("v", host_pages.tolist()))
        )
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.randn(1, 5, 2, 1, 4),
            v_buffer=torch.randn(1, 5, 2, 1, 4),
        )
        host_pages = torch.tensor([5, 1, 3], dtype=torch.int64)
        device_pages = torch.tensor([2, 4, 0], dtype=torch.int64)
        pool.backup_from_device_all_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            list(range(6)),
            "direct",
        )
        for side, method in (
            ("k", pool.quantizer.quantize_pages_keys),
            ("v", pool.quantizer.quantize_pages_values),
        ):
            self.assertEqual(method.call_count, 2)
            for call, expected_pages in zip(
                method.call_args_list, (host_pages[:2], host_pages[2:])
            ):
                self.assertEqual(call.args[0].shape, (len(expected_pages), 2, 4))
                torch.testing.assert_close(call.args[1], expected_pages)
                self.assertIs(call.args[2], getattr(pool, f"{side}_quant_buffers"))
                self.assertIs(call.args[3], getattr(pool, f"{side}_quant_scales"))
                self.assertIs(call.args[4], getattr(pool, f"{side}_quant_offsets"))
        self.assertEqual(
            quantize_order,
            [("k", [5, 1]), ("v", [5, 1]), ("k", [3]), ("v", [3])],
        )

    def test_backup_raw_pages_preserve_order(self):
        pool = self.make_pool()
        pool.k_kvtc = pool.v_kvtc = False
        pool.k_buffer = torch.full((2, 8, 2, 1, 4), float("nan"))
        pool.v_buffer = torch.full((2, 8, 2, 1, 4), float("nan"))
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.arange(2 * 5 * 2 * 4).reshape(2, 5, 2, 1, 4).float(),
            v_buffer=torch.arange(2 * 5 * 2 * 4).reshape(2, 5, 2, 1, 4).float()
            + 1000,
        )
        host_pages = torch.tensor([5, 1, 3])
        device_pages = torch.tensor([2, 4, 0])

        pool.backup_from_device_all_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            list(range(6)),
            "direct",
        )

        torch.testing.assert_close(
            pool.k_buffer.index_select(1, host_pages),
            device.k_buffer.index_select(1, device_pages),
        )
        torch.testing.assert_close(
            pool.v_buffer.index_select(1, host_pages),
            device.v_buffer.index_select(1, device_pages),
        )
        self.assertTrue(torch.isnan(pool.k_buffer[:, 0]).all())
        pool.rotary_emb.invert_native_keys_batch.assert_not_called()
        pool.quantizer.quantize_pages_keys.assert_not_called()
        pool.quantizer.quantize_pages_values.assert_not_called()

    def test_backup_pca_only_projects_unrotated_keys(self):
        pool = self.make_pool()
        pool.kvtc_quant_disable = True
        pool.kvtc_k_mu = torch.tensor([10.0, 20.0, 30.0, 40.0])
        pool.kvtc_v_mu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pool.kvtc_k_V = pool.kvtc_k_V[:, :2]
        pool.kvtc_v_V = pool.kvtc_v_V[:, :2]
        pool.k_buffer = torch.full((8, 2, 2), float("nan"))
        pool.v_buffer = torch.full((8, 2, 2), float("nan"))
        pool.rotary_emb.invert_native_keys_batch.side_effect = (
            lambda positions, values: values - positions.reshape(-1, 1, 1, 1)
        )
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.arange(5 * 2 * 4).reshape(1, 5, 2, 1, 4).float(),
            v_buffer=torch.arange(5 * 2 * 4).reshape(1, 5, 2, 1, 4).float()
            + 100,
        )
        host_pages = torch.tensor([5, 1, 3])
        device_pages = torch.tensor([2, 4, 0])
        positions = [10, 11, 30, 31, 50, 51]

        pool.backup_from_device_all_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            positions,
            "direct",
        )

        source_k = (
            device.k_buffer.index_select(1, device_pages)
            .permute(1, 2, 0, 3, 4)
            .reshape(3, 2, 4)
        )
        source_v = (
            device.v_buffer.index_select(1, device_pages)
            .permute(1, 2, 0, 3, 4)
            .reshape(3, 2, 4)
        )
        expected_k = (
            source_k - torch.tensor(positions).reshape(3, 2, 1) - pool.kvtc_k_mu
        )[:, :, :2]
        expected_v = (source_v - pool.kvtc_v_mu)[:, :, :2]
        torch.testing.assert_close(
            pool.k_buffer.index_select(0, host_pages), expected_k
        )
        torch.testing.assert_close(
            pool.v_buffer.index_select(0, host_pages), expected_v
        )
        self.assertEqual(pool.rotary_emb.invert_native_keys_batch.call_count, 3)
        pool.quantizer.quantize_pages_keys.assert_not_called()
        pool.quantizer.quantize_pages_values.assert_not_called()

    def test_reload_dispatches_ordered_batches(self):
        pool = self.make_pool()
        for method, value in (
            (pool.quantizer.dequantize_pages_keys, 1.0),
            (pool.quantizer.dequantize_pages_values, 2.0),
        ):
            method.side_effect = [
                torch.full((count, 2, 4), value) for count in (2, 1)
            ]
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.full((1, 5, 2, 1, 4), float("nan")),
            v_buffer=torch.full((1, 5, 2, 1, 4), float("nan")),
        )
        host_pages = torch.tensor([5, 1, 3], dtype=torch.int64)
        device_pages = torch.tensor([2, 4, 0], dtype=torch.int64)
        pool.load_to_device_per_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            0,
            list(range(6)),
            "direct",
        )
        for side, method in (
            ("k", pool.quantizer.dequantize_pages_keys),
            ("v", pool.quantizer.dequantize_pages_values),
        ):
            self.assertEqual(method.call_count, 2)
            for call, expected_pages in zip(
                method.call_args_list, (host_pages[:2], host_pages[2:])
            ):
                torch.testing.assert_close(call.args[0], expected_pages)
                self.assertIs(call.args[1], getattr(pool, f"{side}_quant_buffers"))
                self.assertIs(call.args[2], getattr(pool, f"{side}_quant_scales"))
                self.assertIs(call.args[3], getattr(pool, f"{side}_quant_offsets"))

    def test_reload_raw_batches_preserve_page_order(self):
        pool = self.make_pool()
        pool.k_kvtc = pool.v_kvtc = False
        pool.layer_num = 2
        host_shape = (2, 8, 2, 1, 4)
        pool.k_buffer = torch.arange(2 * 8 * 2 * 4).reshape(host_shape).float()
        pool.v_buffer = pool.k_buffer + 1000
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.full((2, 5, 2, 1, 4), float("nan")),
            v_buffer=torch.full((2, 5, 2, 1, 4), float("nan")),
        )
        host_pages = torch.tensor([5, 1, 3])
        device_pages = torch.tensor([2, 4, 0])

        pool.load_to_device_per_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            1,
            list(range(6)),
            "direct",
        )
        self.assertTrue(torch.isnan(device.k_buffer).all())
        self.assertTrue(torch.isnan(device.v_buffer).all())

        pool.load_to_device_per_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            0,
            list(range(6)),
            "direct",
        )

        torch.testing.assert_close(
            device.k_buffer.index_select(1, device_pages),
            pool.k_buffer.index_select(1, host_pages),
        )
        torch.testing.assert_close(
            device.v_buffer.index_select(1, device_pages),
            pool.v_buffer.index_select(1, host_pages),
        )
        self.assertTrue(torch.isnan(device.k_buffer[:, 1]).all())
        pool.rotary_emb.forward_native_keys_batch.assert_not_called()
        pool.quantizer.dequantize_pages_keys.assert_not_called()
        pool.quantizer.dequantize_pages_values.assert_not_called()

    def test_reload_pca_only_batches_reconstruct_and_rotate_keys(self):
        pool = self.make_pool()
        pool.kvtc_quant_disable = True
        pool.kvtc_k_mu = torch.tensor([10.0, 20.0, 30.0, 40.0])
        pool.kvtc_v_mu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pool.kvtc_k_V = pool.kvtc_k_V[:, :2]
        pool.kvtc_v_V = pool.kvtc_v_V[:, :2]
        pool.k_buffer = torch.arange(8 * 2 * 2).reshape(8, 2, 2).float()
        pool.v_buffer = pool.k_buffer + 100
        pool.rotary_emb.forward_native_keys_batch.side_effect = (
            lambda positions, values: values + positions.reshape(-1, 1, 1, 1)
        )
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.full((1, 5, 2, 1, 4), float("nan")),
            v_buffer=torch.full((1, 5, 2, 1, 4), float("nan")),
        )
        host_pages = torch.tensor([5, 1, 3])
        device_pages = torch.tensor([2, 4, 0])
        positions = [10, 11, 30, 31, 50, 51]

        pool.load_to_device_per_layer(
            device,
            self.page_indices(host_pages),
            self.page_indices(device_pages),
            0,
            positions,
            "direct",
        )

        projected_k = pool.k_buffer.index_select(0, host_pages)
        projected_v = pool.v_buffer.index_select(0, host_pages)
        expected_k = (
            torch.cat((projected_k, torch.zeros_like(projected_k)), dim=-1)
            + pool.kvtc_k_mu
            + torch.tensor(positions).reshape(3, 2, 1)
        )
        expected_v = (
            torch.cat((projected_v, torch.zeros_like(projected_v)), dim=-1)
            + pool.kvtc_v_mu
        )
        torch.testing.assert_close(
            device.k_buffer.index_select(1, device_pages),
            expected_k.reshape(3, 2, 1, 1, 4).permute(2, 0, 1, 3, 4),
        )
        torch.testing.assert_close(
            device.v_buffer.index_select(1, device_pages),
            expected_v.reshape(3, 2, 1, 1, 4).permute(2, 0, 1, 3, 4),
        )
        self.assertEqual(pool.rotary_emb.forward_native_keys_batch.call_count, 2)
        pool.quantizer.dequantize_pages_keys.assert_not_called()
        pool.quantizer.dequantize_pages_values.assert_not_called()

    def test_reload_pca_only_one_sided_routes_other_side_raw(self):
        for compressed_side in ("k", "v"):
            with self.subTest(compressed_side=compressed_side):
                pool = self.make_pool()
                pool.kvtc_quant_disable = True
                pool.k_kvtc = compressed_side == "k"
                pool.v_kvtc = compressed_side == "v"
                raw_side = "v" if compressed_side == "k" else "k"
                compressed_pages = torch.arange(8 * 2 * 4).reshape(8, 2, 4).float()
                raw_pages = torch.arange(8 * 2 * 4).reshape(1, 8, 2, 1, 4).float()
                setattr(pool, f"{compressed_side}_buffer", compressed_pages)
                setattr(pool, f"{raw_side}_buffer", raw_pages)
                device = SimpleNamespace(
                    device="cpu",
                    k_buffer=torch.full((1, 3, 2, 1, 4), float("nan")),
                    v_buffer=torch.full((1, 3, 2, 1, 4), float("nan")),
                )
                host_pages = torch.tensor([5, 1])
                device_pages = torch.tensor([2, 0])

                pool.load_to_device_per_layer(
                    device,
                    self.page_indices(host_pages),
                    self.page_indices(device_pages),
                    0,
                    [0, 1, 2, 3],
                    "direct",
                )

                expected_compressed = (
                    compressed_pages.index_select(0, host_pages)
                    .reshape(2, 2, 1, 1, 4)
                    .permute(2, 0, 1, 3, 4)
                )
                torch.testing.assert_close(
                    getattr(device, f"{compressed_side}_buffer").index_select(
                        1, device_pages
                    ),
                    expected_compressed,
                )
                torch.testing.assert_close(
                    getattr(device, f"{raw_side}_buffer").index_select(1, device_pages),
                    raw_pages.index_select(1, host_pages),
                )
                self.assertEqual(
                    pool.rotary_emb.forward_native_keys_batch.call_count,
                    int(compressed_side == "k"),
                )
                pool.quantizer.dequantize_pages_keys.assert_not_called()
                pool.quantizer.dequantize_pages_values.assert_not_called()

    def test_pca_only_skips_quantizer(self):
        pool = self.make_pool()
        pool.kvtc_quant_disable = True
        pool.k_buffer = torch.empty(8, 2, 4)
        pool.v_buffer = torch.empty(8, 2, 4)
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.randn(1, 3, 2, 1, 4),
            v_buffer=torch.randn(1, 3, 2, 1, 4),
        )
        pool.device_pool = device
        host_indices = self.page_indices(torch.tensor([5]))
        device_indices = self.page_indices(torch.tensor([2]))
        pool.backup_from_device_all_layer(
            device,
            host_indices,
            device_indices,
            [0, 1],
            "direct",
        )
        pool.load_to_device_per_layer(
            device, host_indices, device_indices, 0, [0, 1], "direct"
        )
        pool.quantizer.quantize_pages_keys.assert_not_called()
        pool.quantizer.quantize_pages_values.assert_not_called()
        pool.quantizer.dequantize_pages_keys.assert_not_called()
        pool.quantizer.dequantize_pages_values.assert_not_called()

    def test_one_sided_quantization_routes_other_side_directly(self):
        for quantized_side in ("k", "v"):
            with self.subTest(quantized_side=quantized_side):
                pool = self.make_pool()
                pool.k_kvtc = quantized_side == "k"
                pool.v_kvtc = quantized_side == "v"
                raw_side = "v" if quantized_side == "k" else "k"
                quantized_name = "keys" if quantized_side == "k" else "values"
                raw_name = "values" if quantized_side == "k" else "keys"
                setattr(pool, f"{raw_side}_buffer", torch.empty(1, 8, 2, 1, 4))
                device = SimpleNamespace(
                    device="cpu",
                    k_buffer=torch.randn(1, 3, 2, 1, 4),
                    v_buffer=torch.randn(1, 3, 2, 1, 4),
                )
                expected_raw = getattr(device, f"{raw_side}_buffer").clone()
                pool.device_pool = device
                host_indices = self.page_indices(torch.tensor([5]))
                device_indices = self.page_indices(torch.tensor([2]))
                pool.backup_from_device_all_layer(
                    device, host_indices, device_indices, [0, 1], "direct"
                )
                getattr(
                    pool.quantizer, f"quantize_pages_{quantized_name}"
                ).assert_called_once()
                getattr(
                    pool.quantizer, f"quantize_pages_{raw_name}"
                ).assert_not_called()
                dequantize = getattr(
                    pool.quantizer, f"dequantize_pages_{quantized_name}"
                )
                dequantize.return_value = torch.zeros(1, 2, 4)
                device.k_buffer.fill_(float("nan"))
                device.v_buffer.fill_(float("nan"))
                pool.load_to_device_per_layer(
                    device, host_indices, device_indices, 0, [0, 1], "direct"
                )
                dequantize.assert_called_once()
                getattr(
                    pool.quantizer, f"dequantize_pages_{raw_name}"
                ).assert_not_called()
                torch.testing.assert_close(
                    getattr(device, f"{raw_side}_buffer")[:, 2],
                    expected_raw[:, 2],
                )

    def test_empty_page_request_does_not_call_quantizer(self):
        pool = self.make_pool()
        device = SimpleNamespace(device="cpu")
        empty_indices = torch.empty(0, dtype=torch.int64)
        pool.backup_from_device_all_layer(
            device, empty_indices, empty_indices, [], "direct"
        )
        pool.load_to_device_per_layer(
            device, empty_indices, empty_indices, 0, [], "direct"
        )
        for method in (
            pool.quantizer.quantize_pages_keys,
            pool.quantizer.quantize_pages_values,
            pool.quantizer.dequantize_pages_keys,
            pool.quantizer.dequantize_pages_values,
        ):
            method.assert_not_called()


if __name__ == "__main__":
    unittest.main()
