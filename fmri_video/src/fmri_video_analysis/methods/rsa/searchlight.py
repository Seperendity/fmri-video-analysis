"""Whole-brain cross-session diagonal-crossnobis searchlight analysis."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy import ndimage

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.core.data_loading import load_resms_volume
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    require_matching_cache,
)
from fmri_video_analysis.core.validation import validate_nifti_geometry
from fmri_video_analysis.methods.activation.group_contrasts import cluster_table
from fmri_video_analysis.methods.rsa.crossvalidated_distance import (
    CONDITIONS,
    PAIRS,
    pairwise_crossproducts,
    whiten_patterns,
)


def sphere_kernel(radius_mm: float, zooms: tuple[float, float, float]) -> np.ndarray:
    radii = [int(np.ceil(radius_mm / zoom)) for zoom in zooms]
    grid = np.meshgrid(
        *[np.arange(-radius, radius + 1) for radius in radii], indexing="ij"
    )
    squared_distance = sum((axis * zoom) ** 2 for axis, zoom in zip(grid, zooms))
    return (squared_distance <= radius_mm**2 + 1e-8).astype(np.float32)


def subject_maps(
    patterns: np.ndarray,
    noise_variance: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    shrinkage: float,
) -> dict[str, np.ndarray]:
    flat_mask = mask.ravel()
    values = patterns.reshape(patterns.shape[:2] + (-1,))[:, :, flat_mask]
    variance = np.asarray(noise_variance).ravel()[flat_mask]
    whitened = whiten_patterns(values, variance, shrinkage)
    local_count = ndimage.convolve(mask.astype(np.float32), kernel, mode="constant")
    outputs = {}
    for name, voxel_crossproduct in pairwise_crossproducts(whitened).items():
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume.ravel()[flat_mask] = voxel_crossproduct
        local_sum = ndimage.convolve(volume, kernel, mode="constant")
        result = np.divide(
            local_sum,
            local_count,
            out=np.zeros_like(local_sum),
            where=local_count > 0,
        )
        result[~mask] = 0
        outputs[name] = result.astype(np.float32)
    return outputs


def descriptive(merged: Path, output: Path, name: str):
    image = nib.load(str(merged))
    data = np.asarray(image.dataobj, dtype=np.float32)
    mean = data.mean(axis=3)
    sd = data.std(axis=3, ddof=1)
    tstat = np.divide(
        mean, sd / np.sqrt(data.shape[3]), out=np.zeros_like(mean), where=sd > 0
    )
    dz = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    paths = {}
    for suffix, values in (("mean", mean), ("tstat", tstat), ("cohens_dz", dz)):
        path = output / f"{name}_{suffix}.nii.gz"
        nib.save(
            nib.Nifti1Image(values.astype(np.float32), image.affine, image.header), path
        )
        paths[suffix] = path
    return paths


def run_randomise(
    merged: Path,
    mask: Path,
    output: Path,
    name: str,
    n_permutations: int,
    seed: int,
    force: bool,
):
    prefix = output / name
    corrected = Path(f"{prefix}_tfce_corrp_tstat1.nii.gz")
    manifest_path = output / f"{name}_randomise_cache_manifest.json"
    expected_manifest = cache_manifest(
        stage=f"rsa.searchlight.randomise.{name}",
        parameters={
            "n_permutations": n_permutations,
            "seed": seed,
            "one_sample": True,
            "tfce": True,
            "uncorrected_p_output": True,
        },
        inputs={
            "merged": file_fingerprint(merged, full_hash=True),
            "mask": file_fingerprint(mask, full_hash=True),
        },
    )
    if require_matching_cache(
        manifest_path,
        expected_manifest,
        artifacts_exist=corrected.exists(),
        force=force,
    ):
        return corrected
    executable = shutil.which("randomise") or "/usr/local/fsl/bin/randomise"
    command = [
        executable,
        "-i",
        str(merged),
        "-o",
        str(prefix),
        "-m",
        str(mask),
        "-1",
        "-T",
        "-n",
        str(n_permutations),
        f"--seed={seed}",
        "--uncorrp",
    ]
    with (output / f"{name}_randomise.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not corrected.exists():
        raise RuntimeError(f"Missing randomise output: {corrected}")
    atomic_write_json(manifest_path, expected_manifest)
    return corrected


def threshold_and_plot(
    name: str,
    t_path: Path,
    corrp_path: Path,
    output: Path,
    alpha: float,
    suffix: str,
):
    t_img = nib.load(str(t_path))
    t_data = np.asarray(t_img.dataobj)
    corrp = np.asarray(nib.load(str(corrp_path)).dataobj)
    thresholded = np.where(corrp >= 1 - alpha, t_data, 0).astype(np.float32)
    path = output / f"{name}_{suffix}_tstat.nii.gz"
    nib.save(nib.Nifti1Image(thresholded, t_img.affine, t_img.header), path)
    cluster_csv = output / f"{name}_{suffix}_clusters.csv"
    # cluster_table applies p<.05 internally, so use it only for mapwise alpha=.05.
    if np.isclose(alpha, 0.05):
        cluster_table(name, t_path, corrp_path, cluster_csv)
    display = plotting.plot_glass_brain(
        str(path),
        colorbar=True,
        cmap="inferno",
        plot_abs=False,
        title=f"{name.replace('_', ' ').title()} searchlight | {suffix}",
    )
    display.savefig(output / f"{name}_{suffix}_glass.png", dpi=300)
    display.savefig(output / f"{name}_{suffix}_glass.pdf")
    display.close()
    return path, cluster_csv


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask_img = nib.load(str(args.mask))
    mask = np.asarray(mask_img.dataobj) > 0
    kernel = sphere_kernel(args.radius_mm, mask_img.header.get_zooms()[:3])
    (output / "searchlight_parameters.txt").write_text(
        f"radius_mm={args.radius_mm}\nkernel_voxels={int(kernel.sum())}\n"
        f"diagonal_shrinkage={args.diagonal_shrinkage}\n"
        "noise_normalization=subject_first_level_ResMS_diagonal\n"
        f"n_permutations={args.n_permutations}\nseed={args.seed}\n",
        encoding="utf-8",
    )
    atomic_write_json(
        output / "searchlight_run_metadata.json",
        {
            "first_level_dir": str(Path(args.first_level_dir).resolve()),
            "mask": str(Path(args.mask).resolve()),
            "radius_mm": args.radius_mm,
            "diagonal_shrinkage": args.diagonal_shrinkage,
            "noise_normalization": "subject_first_level_ResMS_diagonal",
            "noise_estimate_is_external_to_condition_beta_array": True,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "runtime": runtime_metadata(),
        },
    )
    table = build_four_session_manifest(Path(args.first_level_dir))
    subjects = sorted(table.subject_id.unique())
    merged_paths = {name: output / f"{name}_subjects_4d.nii.gz" for _, _, name in PAIRS}
    subject_map_manifest = cache_manifest(
        stage="rsa.searchlight.subject_maps",
        parameters={
            "radius_mm": args.radius_mm,
            "diagonal_shrinkage": args.diagonal_shrinkage,
            "conditions": list(CONDITIONS),
            "pairs": [list(pair) for pair in PAIRS],
        },
        inputs={
            "mask": file_fingerprint(Path(args.mask), full_hash=True),
            "beta_maps": [
                file_fingerprint(Path(path), full_hash=False)
                for path in table.beta_path
            ],
            "resms_maps": [
                file_fingerprint(
                    Path(args.first_level_dir) / str(int(subject)) / "ResMS.nii",
                    full_hash=False,
                )
                for subject in subjects
            ],
        },
        code={
            "searchlight.py": file_fingerprint(Path(__file__), full_hash=True),
            "crossvalidated_distance.py": file_fingerprint(
                Path(__file__).with_name("crossvalidated_distance.py"), full_hash=True
            ),
        },
    )
    subject_map_manifest_path = output / "subject_maps_cache_manifest.json"
    reuse_subject_maps = require_matching_cache(
        subject_map_manifest_path,
        subject_map_manifest,
        artifacts_exist=any(path.exists() for path in merged_paths.values()),
        force=args.force,
    )
    need_subject_maps = not reuse_subject_maps
    merged_arrays = {}
    if need_subject_maps:
        merged_arrays = {
            name: np.zeros(mask.shape + (len(subjects),), dtype=np.float32)
            for _, _, name in PAIRS
        }
        for subject_index, subject in enumerate(subjects):
            part = table[table.subject_id == subject]
            resms_path, noise_variance = load_resms_volume(
                Path(args.first_level_dir), subject, mask_img
            )
            masked_variance = noise_variance[mask]
            if not np.all(np.isfinite(masked_variance)) or np.any(
                masked_variance <= 0
            ):
                raise RuntimeError(
                    "ResMS must be finite and positive in every searchlight mask "
                    f"voxel: {resms_path}"
                )
            patterns = np.empty((4, 3) + mask.shape, dtype=np.float32)
            for session in (1, 2, 3, 4):
                for condition_index, condition in enumerate(CONDITIONS):
                    match = part[
                        (part.session == session) & (part.condition == condition)
                    ]
                    if len(match) != 1:
                        raise RuntimeError(
                            f"Missing cell: {subject}, {session}, {condition}"
                        )
                    beta_path = match.beta_path.iloc[0]
                    beta_img = nib.load(beta_path)
                    validate_nifti_geometry(
                        mask_img,
                        beta_img,
                        context=f"searchlight mask and beta {beta_path}",
                    )
                    patterns[session - 1, condition_index] = np.asarray(
                        beta_img.dataobj, dtype=np.float32
                    )
            maps = subject_maps(
                patterns,
                noise_variance,
                mask,
                kernel,
                args.diagonal_shrinkage,
            )
            for name, values in maps.items():
                merged_arrays[name][..., subject_index] = values
            print(
                f"Searchlight subject {subject_index + 1}/{len(subjects)}", flush=True
            )

        for name, merged in merged_paths.items():
            nib.save(
                nib.Nifti1Image(merged_arrays[name], mask_img.affine, mask_img.header),
                merged,
            )
        atomic_write_json(subject_map_manifest_path, subject_map_manifest)

    rows = []
    for _, _, name in PAIRS:
        merged = merged_paths[name]
        maps = descriptive(merged, output, name)
        corrected = run_randomise(
            merged,
            Path(args.mask),
            output,
            name,
            args.n_permutations,
            args.seed,
            args.force,
        )
        thresholded, clusters = threshold_and_plot(
            name, maps["tstat"], corrected, output, 0.05, "tfce_fwer05"
        )
        bonferroni, _ = threshold_and_plot(
            name, maps["tstat"], corrected, output, 0.05 / 3, "tfce_fwer_bonferroni3"
        )
        n_clusters = (
            len(pd.read_csv(clusters))
            if clusters.exists() and clusters.stat().st_size
            else 0
        )
        rows.append(
            {
                "distance": name,
                "n_subjects": len(subjects),
                "radius_mm": args.radius_mm,
                "kernel_voxels": int(kernel.sum()),
                "n_permutations": args.n_permutations,
                "mapwise_significant_clusters": n_clusters,
                "tfce_fwer05_map": str(thresholded),
                "three_contrast_bonferroni_map": str(bonferroni),
            }
        )
        if need_subject_maps:
            del merged_arrays[name]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "searchlight_summary.csv", index=False)
    print(summary.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "rsa/searchlight"))
    p.add_argument(
        "--mask",
        default=str(
            RESULTS_DIR / "activation/group_contrasts/group_intersection_mask.nii.gz"
        ),
    )
    p.add_argument("--radius-mm", type=float, default=9.0)
    p.add_argument("--diagonal-shrinkage", type=float, default=0.2)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
