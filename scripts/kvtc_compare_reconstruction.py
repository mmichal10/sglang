#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from scripts.kvtc_calibration_data import (  # noqa: E402
    KV,
    Rope,
    SamplingPolicy,
    discover_dump_directories,
    load_sample_pool,
    sample_pool,
    scan_dump_manifest,
)
from scripts.kvtc_calibration_quant import (  # noqa: E402
    KVTC_FILE_VERSION,
    _npu_integer_batch_errors,
)
from sglang.srt.mem_cache.kvtc_quant import (  # noqa: E402
    KVTC_QUANT_METADATA_DTYPE,
    KVTC_QUANT_STORAGE_DTYPES,
    KVTCQuantizer,
    quant_group_bits,
)


@dataclass(frozen=True)
class Metrics:
    sse: float
    relative_l2: float
    rmse: float
    cosine: float


@dataclass(frozen=True)
class GroupMetrics:
    feature_start: int
    feature_end: int
    dtype: str
    bits: int
    energy: float
    production_sse: float
    calibration_sse: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibrated KVTC quantization with FP32 PCA cutoffs using "
            "the production NPU host-cache quantization methods."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Dump directory or parent containing dataset dump directories; "
            "repeat for multiple locations"
        ),
    )
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-m", "--model-dir", required=True)
    parser.add_argument("-r", "--compression-ratio", type=int, required=True)
    parser.add_argument("-N", "--sample-tokens", type=int, default=4096)
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling-policy",
        choices=[policy.value for policy in SamplingPolicy],
        default=SamplingPolicy.STRICT.value,
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Device KV-cache dtype used by the production dequantizer",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for machine-readable diagnostic results",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.compression_ratio <= 0:
        parser.error("--compression-ratio must be positive")
    if args.sample_tokens <= 0:
        parser.error("--sample-tokens must be positive")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.sample_tokens < args.page_size:
        parser.error("--sample-tokens must be at least --page-size")


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"KVTC config does not exist: {path}")
    config = torch.load(path, map_location="cpu")
    if not isinstance(config, dict):
        raise ValueError("KVTC config must contain a dictionary")
    if config.get("version") != KVTC_FILE_VERSION:
        raise ValueError(
            f"Expected KVTC config version {KVTC_FILE_VERSION!r}, "
            f"found {config.get('version', '<missing>')!r}"
        )
    return config


def production_quant_roundtrip(
    projected: torch.Tensor,
    matrix_params: dict,
    compression_ratio: int,
    page_size: int,
    cache_dtype: torch.dtype,
    matrix_name: str,
    artifact_path: str,
) -> tuple[torch.Tensor, object]:
    page_count = projected.shape[0] // page_size
    ratio_key = str(compression_ratio)
    quant_configs = matrix_params.get("quant")
    if not isinstance(quant_configs, dict) or ratio_key not in quant_configs:
        raise ValueError(
            f"{matrix_name} KVTC config is missing quantization schema "
            f"quant[{ratio_key!r}]"
        )
    is_key = matrix_name.split("/", 1)[0] == "K"
    schema = quant_configs[ratio_key]
    basis_rank = matrix_params["basis"].shape[1]
    quantizer = KVTCQuantizer(
        keys_schema=schema if is_key else None,
        values_schema=None if is_key else schema,
        keys_basis_rank=basis_rank if is_key else None,
        values_basis_rank=None if is_key else basis_rank,
        artifact_path=artifact_path,
        page_size=page_size,
        device=projected.device,
        cache_dtype=cache_dtype,
        staging_capacity_pages=1,
    )
    layout = quantizer.key_layout() if is_key else quantizer.value_layout()
    payloads = {
        dtype_name: torch.empty(
            (page_count, count),
            dtype=KVTC_QUANT_STORAGE_DTYPES[dtype_name],
            device="cpu",
            pin_memory=True,
        )
        for dtype_name, count in layout.payload_elements.items()
    }
    metadata_shape = (page_count, page_size, layout.metadata_count)
    scales = torch.empty(
        metadata_shape, dtype=KVTC_QUANT_METADATA_DTYPE, pin_memory=True
    )
    offsets = torch.empty(
        metadata_shape, dtype=KVTC_QUANT_METADATA_DTYPE, pin_memory=True
    )
    quantize_pages = (
        quantizer.quantize_pages_keys if is_key else quantizer.quantize_pages_values
    )
    dequantize_pages = (
        quantizer.dequantize_pages_keys if is_key else quantizer.dequantize_pages_values
    )

    reconstructed = []
    for page in range(page_count):
        page_values = projected[page * page_size : (page + 1) * page_size]
        host_page = torch.tensor([page], dtype=torch.int64)
        quantize_pages(page_values.unsqueeze(0), host_page, payloads, scales, offsets)
        reconstructed.append(dequantize_pages(host_page, payloads, scales, offsets)[0])
    return torch.cat(reconstructed), layout


def layout_groups(layout):
    return sorted(
        (
            group
            for grouped in (layout.direct_storage_groups, layout.integer_quant_groups)
            for groups in grouped.values()
            for group in groups
        ),
        key=lambda group: group.feature_start,
    )


def reconstruct_cutoff(
    projected: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank == 0:
        return mean.expand(projected.shape[0], -1)
    return projected[:, :rank] @ basis[:, :rank].T + mean


def measure(source: torch.Tensor, reconstructed: torch.Tensor) -> Metrics:
    difference = source - reconstructed
    source_norm = torch.linalg.vector_norm(source)
    difference_norm = torch.linalg.vector_norm(difference)
    relative_l2 = (
        float((difference_norm / source_norm).item())
        if source_norm.item() != 0
        else 0.0
    )
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    cosine = float(F.cosine_similarity(source, reconstructed, dim=1).mean().item())
    return Metrics(float(difference.square().sum().item()), relative_l2, rmse, cosine)


def measure_group_errors(
    projected: torch.Tensor,
    reconstructed: torch.Tensor,
    layout: object,
) -> list[GroupMetrics]:
    results = []
    for group in layout_groups(layout):
        source = projected[:, group.feature_start : group.feature_end]
        runtime = reconstructed[:, group.feature_start : group.feature_end]
        production_sse = float((source - runtime).square().sum().item())
        energy = float(source.square().sum().item())

        if group.dtype_name == "float32":
            calibration_sse = 0.0
        elif group.dtype_name == "bfloat16":
            calibration_sse = float(
                (source - source.to(torch.bfloat16).to(torch.float32))
                .square()
                .sum()
                .item()
            )
        else:
            calibration_sse = float(
                _npu_integer_batch_errors(
                    source.unsqueeze(0), group.dtype_name
                ).item()
            )

        group_size = group.feature_end - group.feature_start
        results.append(
            GroupMetrics(
                feature_start=group.feature_start,
                feature_end=group.feature_end,
                dtype=group.dtype_name,
                bits=quant_group_bits(group_size, group.dtype_name),
                energy=energy,
                production_sse=production_sse,
                calibration_sse=calibration_sse,
            )
        )
    return results


def measure_basis(basis: torch.Tensor, seed: int) -> dict[str, float]:
    column_norms = basis.square().sum(dim=0)
    max_column_norm_error = float((column_norms - 1).abs().max().item())

    generator = torch.Generator(device="cpu").manual_seed(seed)
    probe_errors = []
    for _ in range(4):
        probe = torch.randn(
            basis.shape[1], generator=generator, dtype=basis.dtype
        ).to(basis.device)
        transformed = basis.T @ (basis @ probe)
        probe_errors.append(
            float(
                (
                    torch.linalg.vector_norm(transformed - probe)
                    / torch.linalg.vector_norm(probe)
                ).item()
            )
        )
    return {
        "max_column_norm_error": max_column_norm_error,
        "max_orthogonality_probe_error": max(probe_errors),
    }


def print_result(
    worker: str,
    kv: KV,
    method: str,
    rank: int,
    bits: int,
    metrics: Metrics,
) -> None:
    print(
        f"{worker:<14} {kv.name:<1} {method:<18} rank={rank:<5} "
        f"bits/token={bits:<6} sse={metrics.sse:.7g} "
        f"rel_l2={metrics.relative_l2:.7f} "
        f"rmse={metrics.rmse:.7f} cosine={metrics.cosine:.7f}"
    )


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This comparison requires an available NPU")

    config = load_config(args.config)
    Rope.load_model_config(args.model_dir)
    dump_dirs, workers = discover_dump_directories(args.input_dir)
    manifest = scan_dump_manifest(dump_dirs, workers)
    policy = SamplingPolicy(args.sampling_policy)
    cache_dtype = getattr(torch, args.cache_dtype)
    sample_tokens = args.sample_tokens - args.sample_tokens % args.page_size
    print(
        f"samples={sample_tokens} page_size={args.page_size} "
        f"ratio={args.compression_ratio} cache_dtype={cache_dtype}"
    )
    all_metrics = defaultdict(list)
    diagnostic_results = []

    for worker_index, worker in enumerate(workers):
        for kv_index, kv in enumerate(KV):
            matrix_configs = config.get(kv.matrix_name)
            if not isinstance(matrix_configs, dict) or worker not in matrix_configs:
                raise ValueError(f"KVTC config is missing {kv.matrix_name}/{worker}")
            matrix_params = matrix_configs[worker]
            if not isinstance(matrix_params, dict):
                raise ValueError(
                    f"KVTC config has invalid parameters at {kv.matrix_name}/{worker}"
                )
            mean_cpu = matrix_params["mu"]
            basis_cpu = matrix_params["basis"]
            if not isinstance(mean_cpu, torch.Tensor) or not isinstance(
                basis_cpu, torch.Tensor
            ):
                raise ValueError(f"Invalid PCA tensors for {kv.matrix_name}/{worker}")

            records = [
                record
                for record in manifest
                if record.worker == worker and record.kv is kv
            ]
            pool = load_sample_pool(records, kv is KV.K)
            samples_cpu = sample_pool(
                pool,
                sample_tokens,
                policy,
                args.seed + worker_index * 2 + kv_index,
                "reconstruction comparison",
            )
            del pool

            device = torch.device("npu")
            source = samples_cpu.to(device)
            mean = mean_cpu.to(device)
            basis = basis_cpu.to(device)
            projected = (source - mean) @ basis

            dp_coefficients, layout = production_quant_roundtrip(
                projected,
                matrix_params,
                args.compression_ratio,
                args.page_size,
                cache_dtype,
                f"{kv.name}/{worker}",
                str(args.config),
            )
            dp_reconstructed = (
                dp_coefficients.to(basis.dtype) @ basis[:, : layout.feature_count].T
                + mean
            )
            dp_bits = layout.bytes_per_token * 8
            group_metrics = measure_group_errors(
                projected, dp_coefficients, layout
            )
            quantization_sse = sum(
                group.production_sse for group in group_metrics
            )
            calibration_sse = sum(
                group.calibration_sse for group in group_metrics
            )
            tail_sse = float(
                projected[:, layout.feature_count :].square().sum().item()
            )
            coefficient_sse = quantization_sse + tail_sse

            full_pca_reconstructed = projected @ basis.T + mean
            pca_floor_metrics = measure(source, full_pca_reconstructed)

            original_features = mean.shape[0]
            pca_only_rank = min(
                basis.shape[1], original_features // args.compression_ratio
            )
            equal_rank = min(basis.shape[1], dp_bits // 32)
            equal_bf16_rank = min(basis.shape[1], dp_bits // 16)
            pca_only_reconstructed = reconstruct_cutoff(
                projected.to(cache_dtype).to(torch.float32),
                mean,
                basis,
                pca_only_rank,
            )
            equal_reconstructed = reconstruct_cutoff(projected, mean, basis, equal_rank)
            equal_bf16_reconstructed = reconstruct_cutoff(
                projected.to(torch.bfloat16).to(torch.float32),
                mean,
                basis,
                equal_bf16_rank,
            )

            dp_metrics = measure(source, dp_reconstructed)
            pca_only_metrics = measure(source, pca_only_reconstructed)
            equal_metrics = measure(source, equal_reconstructed)
            equal_bf16_metrics = measure(source, equal_bf16_reconstructed)
            incremental_sse = dp_metrics.sse - pca_floor_metrics.sse
            identity_gap = incremental_sse - coefficient_sse
            identity_relative_gap = abs(identity_gap) / max(coefficient_sse, 1e-30)
            estimator_relative_gap = abs(calibration_sse - quantization_sse) / max(
                quantization_sse, 1e-30
            )
            basis_metrics = measure_basis(
                basis,
                args.seed + worker_index * 2 + kv_index,
            )
            print_result(
                worker,
                kv,
                "DP production",
                layout.feature_count,
                dp_bits,
                dp_metrics,
            )
            print_result(
                worker,
                kv,
                f"PCA-only {args.cache_dtype}",
                pca_only_rank,
                pca_only_rank * cache_dtype.itemsize * 8,
                pca_only_metrics,
            )
            print_result(
                worker,
                kv,
                "equal FP32 PCA",
                equal_rank,
                equal_rank * 32,
                equal_metrics,
            )
            print_result(
                worker,
                kv,
                "equal BF16 PCA",
                equal_bf16_rank,
                equal_bf16_rank * 16,
                equal_bf16_metrics,
            )
            print(
                f"{worker:<14} {kv.name:<1} decomposition "
                f"pca_floor_sse={pca_floor_metrics.sse:.7g} "
                f"quant_sse={quantization_sse:.7g} "
                f"calibration_sse={calibration_sse:.7g} "
                f"tail_sse={tail_sse:.7g} coeff_sse={coefficient_sse:.7g} "
                f"incremental_sse={incremental_sse:.7g} "
                f"identity_gap={identity_gap:.7g} "
                f"identity_relative_gap={identity_relative_gap:.7g} "
                f"estimator_relative_gap={estimator_relative_gap:.7g}"
            )
            print(
                f"{worker:<14} {kv.name:<1} basis "
                f"max_column_norm_error={basis_metrics['max_column_norm_error']:.7g} "
                "max_orthogonality_probe_error="
                f"{basis_metrics['max_orthogonality_probe_error']:.7g}"
            )
            for group_index, group in enumerate(group_metrics):
                relative_error = (
                    (group.production_sse / group.energy) ** 0.5
                    if group.energy > 0
                    else 0.0
                )
                parity_gap = group.production_sse - group.calibration_sse
                print(
                    f"{worker:<14} {kv.name:<1} group={group_index:<3} "
                    f"range={group.feature_start}:{group.feature_end} "
                    f"dtype={group.dtype:<8} bits={group.bits:<6} "
                    f"energy={group.energy:.7g} production_sse={group.production_sse:.7g} "
                    f"calibration_sse={group.calibration_sse:.7g} "
                    f"parity_gap={parity_gap:.7g} relative_error={relative_error:.7g}"
                )

            diagnostic_results.append(
                {
                    "worker": worker,
                    "matrix": kv.name,
                    "cache_dtype": args.cache_dtype,
                    "original_features": mean.shape[0],
                    "basis_rank": basis.shape[1],
                    "retained_rank": layout.feature_count,
                    "bits_per_token": dp_bits,
                    "effective_compression_ratio": 16 * mean.shape[0] / dp_bits,
                    "schema": [
                        [
                            group.feature_end - group.feature_start,
                            group.dtype_name,
                        ]
                        for group in layout_groups(layout)
                    ],
                    "basis": basis_metrics,
                    "pca_floor_sse": pca_floor_metrics.sse,
                    "production_quantization_sse": quantization_sse,
                    "calibration_quantization_sse": calibration_sse,
                    "tail_sse": tail_sse,
                    "coefficient_sse": coefficient_sse,
                    "incremental_reconstruction_sse": incremental_sse,
                    "identity_gap": identity_gap,
                    "identity_relative_gap": identity_relative_gap,
                    "estimator_relative_gap": estimator_relative_gap,
                    "groups": [group.__dict__ for group in group_metrics],
                }
            )
            all_metrics["DP production"].append(dp_metrics)
            all_metrics[f"PCA-only {args.cache_dtype}"].append(pca_only_metrics)
            all_metrics["equal FP32 PCA"].append(equal_metrics)
            all_metrics["equal BF16 PCA"].append(equal_bf16_metrics)
            torch.npu.synchronize()
            torch.npu.empty_cache()

    print("\nMean across worker/K/V entries:")
    for method, entries in all_metrics.items():
        count = len(entries)
        print(
            f"{method:<18} "
            f"rel_l2={sum(item.relative_l2 for item in entries) / count:.7f} "
            f"rmse={sum(item.rmse for item in entries) / count:.7f} "
            f"cosine={sum(item.cosine for item in entries) / count:.7f}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "sample_tokens": sample_tokens,
                    "page_size": args.page_size,
                    "compression_ratio": args.compression_ratio,
                    "cache_dtype": args.cache_dtype,
                    "entries": diagnostic_results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote diagnostics to {args.output_json}")


if __name__ == "__main__":
    run()
