import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from scripts import kvtc_calibrate


class TestKVTCArtifactCache(unittest.TestCase):
    def setUp(self):
        self.workers = ["tp_2_pp_0", "tp_10_pp_0"]
        self.cache_metadata = {
            "artifact_version": kvtc_calibrate.SVD_ARTIFACT_FILE_VERSION,
            "kvtc_version": kvtc_calibrate.KVTC_FILE_VERSION,
            "sample_tokens": 200_000,
            "sampling_policy": "strict",
            "svd_dim": 2,
            "svd_iter": 4,
        }
        self.selection_digest = "a" * 64

    def _make_jobs(self, directory):
        jobs = {}
        marker = 1
        # Insert jobs in reverse order to ensure assembly never relies on dict,
        # completion, filename, or lexicographic worker order.
        for worker in reversed(self.workers):
            for kv in reversed(list(kvtc_calibrate.TensorFileManager.KV)):
                metadata = kvtc_calibrate.build_svd_job_metadata(
                    self.cache_metadata, self.selection_digest, worker, kv
                )
                path = directory / f"artifact-{marker}.pt"
                temporary_path = directory / f".artifact-{marker}.tmp"
                mu = torch.full((3,), float(marker), dtype=torch.float32)
                basis = torch.full((3, 2), float(marker), dtype=torch.float32)
                kvtc_calibrate.save_svd_artifact(
                    path, temporary_path, metadata, mu, basis
                )
                jobs[(worker, kv)] = {
                    "path": path,
                    "temporary_path": temporary_path,
                    "metadata": metadata,
                    "marker": marker,
                }
                marker += 1
        return jobs

    def test_assembly_assigns_each_artifact_to_its_explicit_final_index(self):
        with TemporaryDirectory() as temporary_directory:
            jobs = self._make_jobs(Path(temporary_directory))

            output = kvtc_calibrate.assemble_svd_output(self.workers, jobs)

            for (worker, kv), job in jobs.items():
                section = (
                    "keys"
                    if kv == kvtc_calibrate.TensorFileManager.KV.K
                    else "values"
                )
                expected = torch.full(
                    (3,), float(job["marker"]), dtype=torch.float32
                )
                self.assertTrue(torch.equal(output[section][worker]["mu"], expected))
                expected_basis = torch.full(
                    (3, 2), float(job["marker"]), dtype=torch.float32
                )
                self.assertTrue(
                    torch.equal(
                        output[section][worker]["basis"], expected_basis
                    )
                )

    def test_assembly_rejects_an_artifact_from_a_different_index(self):
        with TemporaryDirectory() as temporary_directory:
            jobs = self._make_jobs(Path(temporary_directory))
            first_pair, second_pair = list(jobs)[:2]
            jobs[first_pair]["path"], jobs[second_pair]["path"] = (
                jobs[second_pair]["path"],
                jobs[first_pair]["path"],
            )

            with self.assertRaisesRegex(
                kvtc_calibrate.SVDArtifactError, "expected final index"
            ):
                kvtc_calibrate.assemble_svd_output(self.workers, jobs)

    def test_artifact_loader_does_not_validate_tensor_shapes(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = {**self.cache_metadata, "svd_dim": 5}
            metadata = kvtc_calibrate.build_svd_job_metadata(
                metadata,
                self.selection_digest,
                self.workers[0],
                kvtc_calibrate.TensorFileManager.KV.K,
            )
            path = directory / "artifact.pt"
            mu = torch.ones(8, dtype=torch.float32)
            basis = torch.ones((7, 3), dtype=torch.float32)
            kvtc_calibrate.save_svd_artifact(
                path, directory / ".artifact.tmp", metadata, mu, basis
            )

            loaded_mu, loaded_basis = kvtc_calibrate.load_svd_artifact(
                path, metadata
            )

            self.assertTrue(torch.equal(loaded_mu, mu))
            self.assertTrue(torch.equal(loaded_basis, basis))

    def test_artifact_loader_ignores_non_identity_metadata(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = kvtc_calibrate.build_svd_job_metadata(
                self.cache_metadata,
                self.selection_digest,
                self.workers[0],
                kvtc_calibrate.TensorFileManager.KV.K,
            )
            path = directory / "artifact.pt"
            mu = torch.ones(3, dtype=torch.float32)
            basis = torch.ones((3, 2), dtype=torch.float32)
            kvtc_calibrate.save_svd_artifact(
                path, directory / ".artifact.tmp", metadata, mu, basis
            )
            expected_metadata = {
                **metadata,
                "sample_tokens": 500,
                "svd_dim": 500,
            }

            loaded_mu, loaded_basis = kvtc_calibrate.load_svd_artifact(
                path, expected_metadata
            )

            self.assertTrue(torch.equal(loaded_mu, mu))
            self.assertTrue(torch.equal(loaded_basis, basis))

    def test_selection_file_accepts_matching_sample_tokens(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            selection_path = directory / "selected-tokens.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "version": kvtc_calibrate.TOKEN_SELECTION_FILE_VERSION,
                        "metadata": {"sample_tokens": 500},
                        "selections": [],
                    }
                ),
                encoding="utf-8",
            )

            store = kvtc_calibrate.TokenSelectionStore(
                mode="load",
                path=selection_path,
                input_dir=directory,
                model_dir=directory / "model",
                sample_tokens=500,
                sampling_policy=(
                    kvtc_calibrate.TensorFileManager.SamplingPolicy.STRICT
                ),
                dump_directories=[directory / "dump"],
                workers=self.workers,
            )

            self.assertEqual(store.metadata["sample_tokens"], 500)

    def test_selection_file_rejects_mismatching_sample_tokens(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            selection_path = directory / "selected-tokens.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "version": kvtc_calibrate.TOKEN_SELECTION_FILE_VERSION,
                        "metadata": {"sample_tokens": 450},
                        "selections": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                kvtc_calibrate.TokenSelectionError,
                "sample_tokens=450, but current -N is 500",
            ):
                kvtc_calibrate.TokenSelectionStore(
                    mode="load",
                    path=selection_path,
                    input_dir=directory,
                    model_dir=directory / "model",
                    sample_tokens=500,
                    sampling_policy=(
                        kvtc_calibrate.TensorFileManager.SamplingPolicy.STRICT
                    ),
                    dump_directories=[directory / "dump"],
                    workers=self.workers,
                )

    def test_reuse_skips_existing_artifact_and_logs_its_worker_index(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            jobs = {}
            for worker in self.workers:
                for kv in kvtc_calibrate.TensorFileManager.KV:
                    jobs[(worker, kv)] = {
                        "path": directory / f"{worker}-{kv.name}.pt",
                        "metadata": {},
                    }
            existing_pair = (self.workers[0], kvtc_calibrate.TensorFileManager.KV.K)
            jobs[existing_pair]["path"].touch()

            with self.assertLogs(kvtc_calibrate.logger, level="INFO") as logs:
                scheduled = kvtc_calibrate.select_svd_jobs(
                    self.workers, jobs, "reuse"
                )

            self.assertEqual(len(scheduled), len(jobs) - 1)
            log_output = "\n".join(logs.output)
            self.assertIn("Skipping SVD for worker=tp_2_pp_0 kv=K", log_output)
            self.assertIn("1 reused, 3 scheduled, 4 total", log_output)

    def test_overwrite_schedules_every_artifact(self):
        with TemporaryDirectory() as temporary_directory:
            jobs = self._make_jobs(Path(temporary_directory))

            scheduled = kvtc_calibrate.select_svd_jobs(
                self.workers, jobs, "overwrite"
            )

            self.assertEqual(len(scheduled), len(jobs))

    def test_worker_job_returns_only_identity_after_persisting_tensors(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tp_2_pp_0-K.pt"
            kv = kvtc_calibrate.TensorFileManager.KV.K
            metadata = kvtc_calibrate.build_svd_job_metadata(
                self.cache_metadata, self.selection_digest, self.workers[0], kv
            )
            svd_result = (
                torch.ones(3, dtype=torch.float32),
                torch.ones((4, 2), dtype=torch.float32),
                torch.ones(2, dtype=torch.float32),
                torch.ones((3, 2), dtype=torch.float32),
            )

            with patch.object(kvtc_calibrate, "SVD", return_value=svd_result):
                result = kvtc_calibrate.run_svd_job(
                    None,
                    None,
                    self.workers[0],
                    kv,
                    2,
                    4,
                    path,
                    path.with_name(".tp_2_pp_0-K.tmp"),
                    metadata,
                )

            self.assertEqual(result, (self.workers[0], kv, path))
            self.assertTrue(path.is_file())
            self.assertFalse(any(isinstance(value, torch.Tensor) for value in result))

    def test_cache_path_exposes_sampling_parameters_and_changes_with_them(self):
        output_path = Path("/tmp/calibration.pt")
        first = kvtc_calibrate.build_svd_cache_directory(
            output_path, self.cache_metadata
        )
        changed_metadata = {**self.cache_metadata, "sample_tokens": 100_000}
        second = kvtc_calibrate.build_svd_cache_directory(
            output_path, changed_metadata
        )

        self.assertIn("N-200000", first.name)
        self.assertIn("policy-strict", first.name)
        self.assertIn("q-2", first.name)
        self.assertIn("niter-4", first.name)
        self.assertNotEqual(first, second)

    def test_svd_cache_policy_is_mandatory(self):
        parser = kvtc_calibrate.create_argument_parser()
        arguments = [
            "-N",
            "1000",
            "--niter",
            "2",
            "-q",
            "4",
            "-i",
            "/input",
            "-o",
            "/output",
            "-m",
            "/model",
            "--log-dir",
            "/logs",
        ]

        with self.assertRaises(SystemExit):
            parser.parse_args(arguments)

        parsed = parser.parse_args(
            [*arguments, "--svd-cache-policy", "reuse"]
        )
        self.assertEqual(parsed.svd_cache_policy, "reuse")

    def test_version_flag_does_not_require_cache_policy(self):
        parser = kvtc_calibrate.create_argument_parser()
        with self.assertRaises(SystemExit) as exit_context:
            parser.parse_args(["--kvtc-version"])
        self.assertEqual(exit_context.exception.code, 0)

    def test_request_with_noncanonical_shape_is_excluded_globally(self):
        dataset = Path("/input/dump")
        request_ids = ("good-a", "good-b", "bad")
        workers = ["tp_0_pp_0", "tp_1_pp_0"]
        managers = {}
        tensor_shapes = {}

        for worker in workers:
            datasets = {dataset: {}}
            for kv in kvtc_calibrate.TensorFileManager.KV:
                request_paths = []
                for request_id in request_ids:
                    path = Path(
                        f"/input/dump/{worker}/"
                        f"{request_id}{kv.value}chunk_0-layer_0.bin"
                    )
                    request_paths.append([path])
                    shape = (1000, 2, 3, 4)
                    if (
                        worker == "tp_0_pp_0"
                        and kv == kvtc_calibrate.TensorFileManager.KV.K
                        and request_id == "bad"
                    ):
                        shape = (1000, 1, 3, 4)
                    tensor_shapes[path] = shape
                datasets[dataset][kv] = request_paths
            manager = Mock()
            manager.datasets_list = [dataset]
            manager.datasets = datasets
            managers[worker] = manager

        inventory = {
            (dataset, request_id): 1000 for request_id in request_ids
        }

        def fake_load_tensor(paths):
            shape = tensor_shapes[paths[0]]
            return torch.empty(shape, dtype=torch.bfloat16), shape[0]

        with (
            patch.object(kvtc_calibrate, "load_tensor", side_effect=fake_load_tensor),
            patch.object(
                kvtc_calibrate.Rope,
                "invert_rope",
                side_effect=lambda tensor: tensor,
            ),
        ):
            validated = kvtc_calibrate.validate_global_requests(
                managers, workers, inventory
            )

        self.assertEqual(
            set(validated),
            {(dataset, "good-a"), (dataset, "good-b")},
        )

    def test_transform_tensors_rejects_mixed_feature_shapes(self):
        tensors = [
            torch.empty((50, 2, 3, 4)),
            torch.empty((50, 1, 3, 4)),
        ]

        with self.assertRaisesRegex(ValueError, "different feature shapes"):
            kvtc_calibrate.transform_tensors(tensors)

    def test_svd_does_not_skip_a_selected_request_that_fails_to_load(self):
        dataset = Path("/input/dump")
        kv = kvtc_calibrate.TensorFileManager.KV.K
        path = Path("/input/dump/tp_0_pp_0/request-K-chunk_0-layer_0.bin")
        manager = Mock()
        manager.datasets_list = [dataset]
        manager.datasets = {dataset: {kv: [[path]]}}
        selection_store = Mock()
        selection_store.contains.return_value = True

        with patch.object(kvtc_calibrate, "load_tensor", return_value=(None, None)):
            with self.assertRaisesRegex(RuntimeError, "failed dump reconstruction"):
                kvtc_calibrate.SVD(
                    manager,
                    selection_store,
                    svd_dim=2,
                    svd_iter=1,
                    kv=kv,
                    undo_rope=True,
                )


if __name__ == "__main__":
    unittest.main()
