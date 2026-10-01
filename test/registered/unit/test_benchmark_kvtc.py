"""CPU-only checks for KVTC benchmark discovery, routing and timing boundaries.

Run from the repository root with:
PYTHONPATH=test/registered/unit python -m unittest test_benchmark_kvtc
"""

import ast
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import benchmark_kvtc as bench


class TestBenchmarkKVTC(unittest.TestCase):
    def args(self, *extra):
        return bench.build_parser().parse_args(
            ["--model-dir", "/model", "--dump-dir", "/dumps", *extra]
        )

    def test_defaults_and_final_flag_names(self):
        args = self.args("--compression-matrix", "/pca.pt")
        bench.validate_args(bench.build_parser(), args)
        self.assertEqual(args.tp_worker_name, "tp_0_pp_0")
        self.assertEqual((args.page_size, args.warmups, args.iterations), (128, 5, 20))
        self.assertEqual(bench.enabled_modes(args), ["quant", "compressed", "baseline"])
        for name in bench.MODE_NAMES:
            disabled = self.args(f"--disable-{name}-benchmark")
            self.assertNotIn(name, bench.enabled_modes(disabled))

    def test_baseline_only_needs_no_calibration(self):
        args = self.args("--disable-quant-benchmark", "--disable-compressed-benchmark")
        bench.validate_args(bench.build_parser(), args)
        self.assertEqual(bench.enabled_modes(args), ["baseline"])

    def test_profile_defaults_and_unique_directories(self):
        args = self.args()
        self.assertFalse(args.profile)
        self.assertEqual(args.profile_iterations, 1)
        self.assertEqual(args.profile_dir, Path("kvtc_profiles"))
        with tempfile.TemporaryDirectory() as directory:
            first = bench.create_profile_directory(Path(directory))
            second = bench.create_profile_directory(Path(directory))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_profiling_capture_boundaries_and_failure_cleanup(self):
        for failure in (None, "warmup", "capture", "export"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                host = SimpleNamespace(compressed_pool=SimpleNamespace())
                active = False
                calls = []
                capture = MagicMock()

                @contextlib.contextmanager
                def profile(**kwargs):
                    nonlocal active
                    self.assertFalse(host._profile_kvtc)
                    active = True
                    try:
                        yield capture
                    finally:
                        active = False
                        self.assertFalse(host._profile_kvtc)
                        self.assertFalse(host.compressed_pool._profile_kvtc)
                    if failure == "export":
                        raise RuntimeError("export")

                def operation():
                    flags = (host._profile_kvtc, host.compressed_pool._profile_kvtc)
                    self.assertEqual(flags, (active, active))
                    calls.append(active)
                    if failure == ("capture" if active else "warmup"):
                        raise RuntimeError(failure)

                profiler = SimpleNamespace(
                    _ExperimentalConfig=MagicMock(),
                    ProfilerLevel=SimpleNamespace(Level1="level1"),
                    ExportType=SimpleNamespace(Text="text"),
                    ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
                    profile=MagicMock(side_effect=profile),
                    tensorboard_trace_handler=MagicMock(),
                )
                torch = SimpleNamespace(
                    npu=SimpleNamespace(synchronize=MagicMock()),
                    profiler=SimpleNamespace(record_function=MagicMock(
                        side_effect=lambda _: contextlib.nullcontext()
                    )),
                )
                with patch.dict("sys.modules", {
                    "torch": torch, "torch_npu": SimpleNamespace(profiler=profiler)
                }):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, failure):
                            bench.profile_operation(operation, host, 2, 3, Path(directory), "quant/offload")
                    else:
                        result = bench.profile_operation(operation, host, 2, 3, Path(directory), "quant/offload")
                        self.assertEqual(result, directory)
                        self.assertEqual(calls, [False, False, True, True, True])
                        self.assertEqual(capture.step.call_count, 3)
                        self.assertEqual(torch.npu.synchronize.call_count, 11)
                        self.assertEqual(torch.profiler.record_function.call_count, 6)
                        options = profiler.profile.call_args.kwargs
                        self.assertEqual(options["activities"], ["cpu", "npu"])
                        self.assertTrue(options["record_shapes"])
                        self.assertFalse(options["profile_memory"])
                        self.assertFalse(options["with_stack"])
                        profiler.tensorboard_trace_handler.assert_called_once_with(directory, async_mode=False)
                self.assertFalse(host._profile_kvtc)
                self.assertFalse(host.compressed_pool._profile_kvtc)

    def test_profile_and_benchmark_paths_are_exclusive(self):
        args = self.args("--profile", "--profile-iterations", "3")
        args.profile_run_dir = Path("/traces/run")
        mode = SimpleNamespace(name="quant")
        host, operation = object(), MagicMock()
        with patch.object(bench, "profile_operation", return_value="trace") as profile, patch.object(bench, "measure", return_value=[1.0]) as measure:
            self.assertEqual(bench.run_direction(args, mode, "reload", operation, host), "trace")
            profile.assert_called_once_with(operation, host, 5, 3, Path("/traces/run/quant/reload"), "quant/reload")
            measure.assert_not_called()
            profile.reset_mock()
            args.profile = False
            with patch.dict("sys.modules", {"torch": SimpleNamespace(npu=SimpleNamespace(synchronize=MagicMock()))}):
                self.assertEqual(bench.run_direction(args, mode, "reload", operation, host), [1.0])
            profile.assert_not_called()
            measure.assert_called_once()

    def test_pool_disabled_guards_execute_without_context_creation(self):
        source = bench.REPO_ROOT / "python/sglang/srt/mem_cache/memory_pool_host.py"
        tree = ast.parse(source.read_text())
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "self._profile_kvtc"]
        self.assertGreater(len(guards), 30)
        for guard in guards:
            self.assertEqual(len(guard.body), 1)
            scope = guard.body[0]
            self.assertIsInstance(scope, ast.With)
            self.assertEqual([ast.dump(n) for n in scope.body], [ast.dump(n) for n in guard.orelse])
            # Replace real tensor operations with an observable probe, retaining
            # the actual guard/context expression from the production method.
            probe = ast.parse("events.append('operation')").body
            scope.body = probe
            guard.orelse = probe
            code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(source), "exec")
            record = MagicMock(side_effect=AssertionError("disabled context"))
            namespace = {"self": SimpleNamespace(_profile_kvtc=False), "torch": SimpleNamespace(profiler=SimpleNamespace(record_function=record)), "events": []}
            exec(code, namespace)
            record.assert_not_called()
            self.assertEqual(namespace["events"], ["operation"])
            namespace["self"]._profile_kvtc = True
            record.side_effect = lambda _: contextlib.nullcontext()
            exec(code, namespace)
            record.assert_called_once()
            self.assertEqual(namespace["events"], ["operation", "operation"])

    def test_profile_report_contains_paths_without_metrics(self):
        results = {"quant": {"offload": "/trace/quant/offload", "reload": "/trace/quant/reload"}}
        with patch("builtins.print") as output, patch.object(bench, "format_results") as metrics:
            bench.print_final_report(["commit: abc", "warmups: 5"], ["validation OK"], results, 256, 1024, profiling=True)
        output.assert_called_once()
        metrics.assert_not_called()
        report = output.call_args.args[0]
        for value in ("KVTC PROFILE REPORT", "commit: abc", "warmups: 5", "validation OK", "/trace/quant/offload", "/trace/quant/reload"):
            self.assertIn(value, report)
        self.assertNotIn("Mean ms", report)

    def test_invalid_cli_settings(self):
        cases = [
            [],  # Missing calibration for the default modes.
            ["--page-size", "100"],
            ["--iterations", "0"],
            ["--profile-iterations", "0"],
            ["--warmups", "-1"],
            ["--host-memory-gb", "9"],
            ["--tp-worker-name", "worker0"],
            [f"--disable-{name}-benchmark" for name in bench.MODE_NAMES],
        ]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()):
                extra = [] if not case else ["--compression-matrix", "/pca.pt"]
                with self.assertRaises(SystemExit):
                    bench.validate_args(bench.build_parser(), self.args(*extra, *case))

    def test_prefix_rounding_and_too_short_or_long(self):
        self.assertEqual(bench.selected_token_count(1100, None, 128), 1024)
        self.assertEqual(bench.selected_token_count(1100, 777, 128), 768)
        self.assertEqual(bench.selected_token_count(1100, 256, 128), 256)
        with self.assertRaisesRegex(ValueError, "contains"):
            bench.selected_token_count(1100, 1101, 128)
        with self.assertRaisesRegex(ValueError, "one remaining page"):
            bench.selected_token_count(1100, 255, 128)

    def test_ratio_inference_does_not_silently_choose(self):
        self.assertEqual(bench.resolve_ratio({"quant": {"8": []}}, None, "--k-cr"), 8)
        self.assertEqual(bench.resolve_ratio({}, 4, "--k-cr"), 4)
        for params in ({}, {"quant": {"4": [], "8": []}}):
            with self.assertRaisesRegex(ValueError, "Pass --k-cr"):
                bench.resolve_ratio(params, None, "--k-cr")

    def make_dump(self, directory, name, chunks=2, layers=2):
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for chunk in reversed(range(chunks)):
            for layer in reversed(range(layers)):
                for kv in ("V", "K"):
                    path = directory / f"{name}-{kv}-chunk_{chunk}-layer_{layer}.bin"
                    path.touch()
                    paths.append(path)
        return paths

    @staticmethod
    def shape_loader(path):
        chunk = int(bench.DUMP_PATTERN.fullmatch(path.name).group(3))
        return SimpleNamespace(shape=(chunk + 1, 2, 8))

    def test_discovery_orders_numeric_chunks_and_keeps_short_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "tp_0_pp_0"
            self.make_dump(worker, "conversation", chunks=12)
            dumps = bench.discover_dumps(root, "tp_0_pp_0", self.shape_loader)
            self.assertEqual(len(dumps), 1)
            self.assertEqual(dumps[0].token_count, sum(range(1, 13)))
            chunk_ids = [
                int(bench.DUMP_PATTERN.fullmatch(path.name).group(3))
                for path in dumps[0].k_paths
            ]
            self.assertEqual(chunk_ids, [chunk for chunk in range(12) for _ in range(2)])
            direct = bench.discover_dumps(worker, "tp_0_pp_0", self.shape_loader)
            self.assertEqual(direct, dumps)

    def test_incomplete_dump_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_dump(root / "tp_0_pp_0", "broken")
            paths[0].unlink()
            with self.assertLogs(bench.logger, level="WARNING"):
                self.assertEqual(bench.discover_dumps(root, "tp_0_pp_0", self.shape_loader), [])

    def test_worker_is_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dump(root / "tp_1_pp_0", "request")
            with self.assertRaisesRegex(ValueError, "available workers: tp_1_pp_0"):
                bench.discover_dumps(root, "tp_0_pp_0", self.shape_loader)

    def test_duplicate_names_are_qualified_by_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in ("a", "b"):
                self.make_dump(root / dataset / "tp_0_pp_0", "same")
            dumps = bench.discover_dumps(root, "tp_0_pp_0", self.shape_loader)
            self.assertEqual([dump.name for dump in dumps], ["a/same", "b/same"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_real_dump_reconstruction_preserves_prefix_and_sink(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_dump(root / "tp_0_pp_0", "real", chunks=12)
            for path in paths:
                _, side, chunk, layer = bench.DUMP_PATTERN.fullmatch(path.name).groups()
                value = int(chunk) * 10 + int(layer) + (1000 if side == "V" else 0)
                torch.save(torch.full((32, 2, 8), value, dtype=torch.bfloat16), path)
            dump = bench.discover_dumps(root, "tp_0_pp_0")[0]
            self.assertEqual(dump.token_count, 384)
            keys, values = bench.load_selected_dump(dump, 352)
            self.assertEqual(keys.shape, (352, 2, 2, 8))
            for chunk in range(11):  # Includes chunk_10, after chunk_9.
                for layer in range(2):
                    for tensor, offset in ((keys, 0), (values, 1000)):
                        expected = torch.full(
                            (32, 2, 8), chunk * 10 + layer + offset,
                            dtype=torch.bfloat16,
                        )
                        self.assertTrue(torch.equal(
                            tensor[chunk * 32 : (chunk + 1) * 32, layer], expected
                        ))

    def test_interactive_and_explicit_selection(self):
        dumps = [bench.Dump(name, Path("/dump"), 1000, (), ()) for name in ("a", "b")]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIs(bench.select_dump(dumps, "b", interactive=False), dumps[1])
            self.assertIs(
                bench.select_dump(dumps, None, interactive=True, input_fn=lambda _: "2"),
                dumps[1],
            )
            self.assertIs(bench.select_dump(dumps[:1], None, interactive=False), dumps[0])
            with self.assertRaisesRegex(ValueError, "--dump-name"):
                bench.select_dump(dumps, None, interactive=False)
        self.assertIn("1,000 tokens", output.getvalue())

    def test_host_sizing_accounts_for_integer_split_and_device_capacity(self):
        mode = bench.Mode("baseline", 1_000_000, 1, 1)
        for tokens in (256, 32768, 2_000_000):
            size = bench.required_host_gb(tokens, 128, [mode], mode.page_bytes)
            self.assertGreaterEqual(size, 10)
            self.assertGreaterEqual(int(size * 0.9) * 1e9, (tokens // 128 + 1) * mode.page_bytes)
            self.assertGreaterEqual(int(size * 0.1) * 1e9, 2 * mode.page_bytes)

    def test_timing_excludes_warmups_and_waits_for_completion(self):
        events = []
        times = iter((10_000_000, 12_000_000, 20_000_000, 23_000_000))

        def clock():
            events.append("clock")
            return next(times)

        samples = bench.measure(
            lambda: events.append("op"), 1, 2, lambda: events.append("sync"), clock
        )
        self.assertEqual(samples, [2.0, 3.0])
        self.assertEqual(
            events,
            ["sync", "op", "sync"] + ["sync", "clock", "op", "sync", "clock"] * 2,
        )

    def test_zero_warmups_still_executes_measured_operations(self):
        operation = MagicMock()
        times = iter((0, 1_000_000))
        self.assertEqual(bench.measure(operation, 0, 1, lambda: None, lambda: next(times)), [1.0])
        operation.assert_called_once()

    def test_reload_calls_every_layer(self):
        seen = []
        host = SimpleNamespace(load_to_device_per_layer=lambda req: seen.append(req.layer_id))
        bench.reload_all_layers(host, SimpleNamespace(layer_id=None), 5)
        self.assertEqual(seen, list(range(5)))

    def test_run_mode_uses_real_interface_and_separate_direction_batches(self):
        # Exercise orchestration with a fake device, not a replacement benchmark
        # implementation. Verify all modes preserve sink routing and never reload
        # between offloads (which would repeatedly quantize reconstructed data).
        for name in bench.MODE_NAMES:
            with self.subTest(mode=name):
                events = []
                host = MagicMock()
                host.alloc.return_value = (list(range(128)), list(range(128)))
                host.backup_from_device_all_layer.side_effect = lambda req: events.append("offload")
                host.load_to_device_per_layer.side_effect = (
                    lambda req: events.append(f"reload{req.layer_id}")
                )
                constructor = MagicMock(return_value=host)
                module = SimpleNamespace(
                    NPUMHATokenToKVPoolHybrid=constructor,
                    KVTCHostMemoryRequest=SimpleNamespace,
                )
                npu = SimpleNamespace(
                    synchronize=lambda: None, Stream=object,
                    stream=lambda _: contextlib.nullcontext(),
                )
                torch = SimpleNamespace(
                    npu=npu, int64="int64",
                    arange=lambda start, end, **kw: list(range(start, end)),
                )
                keys, values = MagicMock(), MagicMock()
                keys.__len__.return_value = values.__len__.return_value = 256
                device = SimpleNamespace(layer_num=3, k_buffer=MagicMock(), v_buffer=MagicMock())
                args = self.args(
                    "--compression-matrix", "/pca.pt", "--iterations", "2",
                    "--warmups", "0", "--host-memory-gb", "10",
                    "--k-cr", "8", "--v-cr", "4",
                )
                modules = {"torch": torch, "sglang.srt.mem_cache.memory_pool_host": module}
                with (
                    patch.dict("sys.modules", modules),
                    patch.object(bench, "check_reconstruction", return_value="validation OK"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result, validation = bench.run_mode(
                        args, bench.Mode(name, 1, 1, 1), device, keys, values, None
                    )
                self.assertEqual(events, ["offload"] * 2 + ["reload0", "reload1", "reload2"] * 2)
                self.assertEqual(validation, "validation OK")
                self.assertEqual(
                    {key: len(value) for key, value in result.items()},
                    {"offload": 2, "reload": 2},
                )
                host.alloc.assert_called_once_with(128, 128)
                request = host.backup_from_device_all_layer.call_args.args[0]
                self.assertEqual(request.device_indices_sink, list(range(128, 256)))
                self.assertEqual(request.device_indices_compressed, list(range(256, 384)))
                self.assertEqual(request.token_indices_compressed, list(range(128, 256)))
                options = constructor.call_args.kwargs
                self.assertEqual(options["kvtc_quant_disable"], name == "compressed")
                self.assertEqual(
                    options["kvtc_params_path"], "" if name == "baseline" else "/pca.pt"
                )
                self.assertEqual(
                    options["kvtc_k_compression_ratio"], 0 if name == "baseline" else 8
                )

    def test_result_comparison_and_missing_baseline(self):
        results = {"quant": {"offload": [2.0]}, "baseline": {"offload": [4.0]}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            bench.print_results(results, 256, 1024)
        self.assertIn("2.000x", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            bench.print_results({"quant": results["quant"]}, 256, 1024)
        self.assertIn("n/a", output.getvalue())

    def test_maximum_latency_column_and_single_iteration(self):
        for samples in ([3.0, 1.0, 9.0, 2.0], [7.0]):
            with self.subTest(samples=samples):
                report = bench.format_results(
                    {"baseline": {"offload": samples}}, 256, 1024
                )
                header = next(line for line in report.splitlines() if line.startswith("Mode"))
                row = next(line for line in report.splitlines() if line.startswith("baseline"))
                self.assertLess(header.index("Min ms"), header.index("Max ms"))
                self.assertLess(header.index("Max ms"), header.index("P95 ms"))
                values = row.split()
                self.assertEqual(float(values[4]), min(samples))
                self.assertEqual(float(values[5]), max(samples))

    def test_final_report_groups_metadata_validation_and_metrics(self):
        metadata = [
            "SGLang commit: original-hash (worktree dirty)",
            "Dump: request-1; TP worker: tp_0_pp_0",
            "Iterations: 20; warmups: 5; actual tokens: 29952",
            "K compression ratio: 8; V compression ratio: 4",
        ]
        validations = ["baseline validation: finite, sink exact; K relative-L2=0"]
        results = {"baseline": {"offload": [1.0], "reload": [2.0]}}
        with patch("builtins.print") as output, patch.object(bench, "git_version") as git:
            bench.print_final_report(metadata, validations, results, 29952, 1024)
        output.assert_called_once()
        git.assert_not_called()
        report = output.call_args.args[0]
        self.assertTrue(output.call_args.kwargs["flush"])
        for line in metadata + validations:
            self.assertIn(line, report)
        self.assertLess(report.index(metadata[-1]), report.index(validations[0]))
        self.assertLess(report.index(validations[0]), report.index("Mean ms"))
        self.assertIn("Max ms", report)
        self.assertIn("offload", report)
        self.assertIn("reload", report)
        self.assertTrue(report.endswith("========== END KVTC BENCHMARK REPORT =========="))


if __name__ == "__main__":
    unittest.main()
