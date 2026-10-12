#!/usr/bin/env python3
"""Three-condition repeated-measures whole-brain F test with randomise TFCE."""

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

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    require_matching_cache,
)
from fmri_video_analysis.core.validation import validate_nifti_geometry
from fmri_video_analysis.methods.activation.group_contrasts import cluster_table

CONDITIONS = ("Simple", "Complex", "Control")


def write_vest(path: Path, matrix: np.ndarray, kind: str):
    matrix = np.asarray(matrix)
    if kind == "mat":
        header = (
            f"/NumWaves\t{matrix.shape[1]}\n"
            f"/NumPoints\t{matrix.shape[0]}\n"
            "/PPheights\t" + "\t".join(["1"] * matrix.shape[1]) + "\n\n"
        )
    elif kind == "con":
        header = (
            f"/NumWaves\t{matrix.shape[1]}\n"
            f"/NumContrasts\t{matrix.shape[0]}\n"
            "/PPheights\t" + "\t".join(["1"] * matrix.shape[0]) + "\n"
            "/RequiredEffect\t" + "\t".join(["1"] * matrix.shape[0]) + "\n\n"
        )
    elif kind == "fts":
        header = f"/NumWaves\t{matrix.shape[1]}\n/NumContrasts\t{matrix.shape[0]}\n\n"
    elif kind == "grp":
        header = f"/NumWaves\t1\n/NumPoints\t{matrix.shape[0]}\n\n"
    else:
        raise ValueError(kind)
    lines = [header + "/Matrix"]
    lines.extend("\t".join(f"{value:.8g}" for value in row) for row in matrix)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_data(args: argparse.Namespace, output: Path):
    merged_path = output / "condition_means_261_observations.nii.gz"
    design_path = output / "design.mat"
    contrast_path = output / "design.con"
    ftest_path = output / "design.fts"
    group_path = output / "design.grp"
    label_path = output / "condition_mean_labels.csv"
    artifacts = (
        merged_path,
        design_path,
        contrast_path,
        ftest_path,
        group_path,
        label_path,
    )
    table = build_four_session_manifest(Path(args.first_level_dir))
    expected_manifest = cache_manifest(
        stage="activation.repeated_measures.prepare",
        parameters={
            "conditions": list(CONDITIONS),
            "sessions_averaged_per_condition": 4,
            "condition_codes": {
                "Simple": [1.0, 1.0],
                "Complex": [-1.0, 1.0],
                "Control": [0.0, -2.0],
            },
        },
        inputs={
            "first_level_dir": str(Path(args.first_level_dir).resolve()),
            "beta_maps": [
                file_fingerprint(Path(path), full_hash=False)
                for path in table.beta_path
            ],
        },
        code={"repeated_measures.py": file_fingerprint(Path(__file__), full_hash=True)},
    )
    manifest_path = output / "prepare_cache_manifest.json"
    if require_matching_cache(
        manifest_path,
        expected_manifest,
        artifacts_exist=any(path.exists() for path in artifacts),
        force=args.force,
    ):
        return merged_path, design_path, contrast_path, ftest_path, group_path

    subjects = sorted(table.subject_id.unique())
    reference = nib.load(table.beta_path.iloc[0])
    data = np.empty(reference.shape + (len(subjects) * 3,), dtype=np.float32)
    rows = []
    row_index = 0
    for subject_index, subject in enumerate(subjects):
        subject_table = table[table.subject_id == subject]
        for condition in CONDITIONS:
            paths = subject_table.loc[
                subject_table.condition == condition, "beta_path"
            ].tolist()
            if len(paths) != 4:
                raise RuntimeError(
                    f"Subject {subject} {condition}: expected 4 beta maps"
                )
            arrays = []
            for path in paths:
                image = nib.load(path)
                validate_nifti_geometry(
                    reference,
                    image,
                    context=f"repeated-measures beta {path}",
                )
                arrays.append(np.asarray(image.dataobj, dtype=np.float32))
            condition_mean = np.mean(arrays, axis=0)
            data[..., row_index] = condition_mean
            rows.append(
                {
                    "row": row_index + 1,
                    "subject_id": subject,
                    "condition": condition,
                    "n_session_betas": 4,
                }
            )
            row_index += 1
        print(f"Prepared subject {subject_index + 1}/{len(subjects)}", flush=True)
    nib.save(nib.Nifti1Image(data, reference.affine, reference.header), merged_path)
    pd.DataFrame(rows).to_csv(label_path, index=False)

    n_subjects = len(subjects)
    design = np.zeros((n_subjects * 3, 2 + n_subjects), dtype=float)
    groups = np.zeros((n_subjects * 3, 1), dtype=float)
    condition_codes = {
        "Simple": (1.0, 1.0),
        "Complex": (-1.0, 1.0),
        "Control": (0.0, -2.0),
    }
    for row, record in enumerate(rows):
        subject_index = subjects.index(record["subject_id"])
        design[row, :2] = condition_codes[record["condition"]]
        design[row, 2 + subject_index] = 1.0
        groups[row, 0] = subject_index + 1
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError("Repeated-measures design matrix is rank deficient")
    t_contrasts = np.zeros((2, design.shape[1]), dtype=float)
    t_contrasts[0, 0] = 1.0
    t_contrasts[1, 1] = 1.0
    f_contrast = np.array([[1.0, 1.0]])
    write_vest(design_path, design, "mat")
    write_vest(contrast_path, t_contrasts, "con")
    write_vest(ftest_path, f_contrast, "fts")
    write_vest(group_path, groups, "grp")
    atomic_write_json(manifest_path, expected_manifest)
    return merged_path, design_path, contrast_path, ftest_path, group_path


def render(output: Path, fstat_path: Path, corrp_path: Path):
    f_img = nib.load(str(fstat_path))
    corrp = np.asarray(nib.load(str(corrp_path)).dataobj)
    f_values = np.asarray(f_img.dataobj)
    thresholded = np.where(corrp >= 0.95, f_values, 0).astype(np.float32)
    thresholded_path = output / "condition_omnibus_tfce_fwer05_fstat.nii.gz"
    nib.save(nib.Nifti1Image(thresholded, f_img.affine, f_img.header), thresholded_path)
    cluster_csv = output / "condition_omnibus_clusters_tfce_fwer05.csv"
    cluster_table("condition_omnibus", fstat_path, corrp_path, cluster_csv)
    if cluster_csv.exists() and cluster_csv.stat().st_size:
        clusters = pd.read_csv(cluster_csv)
        if "peak_t" in clusters:
            clusters = clusters.rename(columns={"peak_t": "peak_F"})
            clusters.to_csv(cluster_csv, index=False)

    display = plotting.plot_glass_brain(
        str(thresholded_path),
        colorbar=True,
        cmap="inferno",
        plot_abs=False,
        title="Three-condition repeated-measures omnibus F | TFCE-FWER p < .05",
    )
    display.savefig(output / "condition_omnibus_glass.png", dpi=300)
    display.savefig(output / "condition_omnibus_glass.pdf")
    display.close()
    display = plotting.plot_stat_map(
        str(thresholded_path),
        display_mode="z",
        cut_coords=8,
        colorbar=True,
        cmap="inferno",
        threshold=1e-8,
        title="Simple / Complex / Control omnibus F",
    )
    display.savefig(output / "condition_omnibus_slices.png", dpi=300)
    display.savefig(output / "condition_omnibus_slices.pdf")
    display.close()
    return thresholded_path, cluster_csv


def build_randomise_command(
    executable: str,
    *,
    merged: Path,
    prefix: Path,
    design: Path,
    contrast: Path,
    ftest: Path,
    groups: Path,
    mask: Path,
    n_permutations: int,
    seed: int,
) -> list[str]:
    """Build the omnibus-F-only randomise command."""

    return [
        executable,
        "-i",
        str(merged),
        "-o",
        str(prefix),
        "-d",
        str(design),
        "-t",
        str(contrast),
        "-f",
        str(ftest),
        "--fonly",
        "-e",
        str(groups),
        "-m",
        str(mask),
        "-n",
        str(n_permutations),
        "-T",
        "--uncorrp",
        f"--seed={seed}",
    ]


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged, design, contrast, ftest, groups = prepare_data(args, output)
    prefix = output / "condition_omnibus"
    fstat = Path(f"{prefix}_fstat1.nii.gz")
    corrp = Path(f"{prefix}_tfce_corrp_fstat1.nii.gz")
    mask = nib.load(str(args.mask))
    merged_image = nib.load(str(merged))
    validate_nifti_geometry(
        merged_image,
        mask,
        context="repeated-measures randomise mask",
    )
    expected_manifest = cache_manifest(
        stage="activation.repeated_measures.randomise",
        parameters={
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "tfce": True,
            "uncorrected_p_output": True,
            "f_only": True,
        },
        inputs={
            name: file_fingerprint(path, full_hash=True)
            for name, path in {
                "merged": merged,
                "design": design,
                "contrast": contrast,
                "ftest": ftest,
                "groups": groups,
                "mask": Path(args.mask),
            }.items()
        },
    )
    randomise_manifest_path = output / "randomise_cache_manifest.json"
    reuse = require_matching_cache(
        randomise_manifest_path,
        expected_manifest,
        artifacts_exist=fstat.exists() or corrp.exists(),
        force=args.force,
    )
    if not reuse:
        randomise = shutil.which("randomise") or "/usr/local/fsl/bin/randomise"
        command = build_randomise_command(
            randomise,
            merged=merged,
            prefix=prefix,
            design=design,
            contrast=contrast,
            ftest=ftest,
            groups=groups,
            mask=Path(args.mask),
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        with (output / "randomise.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not fstat.exists() or not corrp.exists():
            raise RuntimeError("randomise did not create omnibus F outputs")
        atomic_write_json(randomise_manifest_path, expected_manifest)
    if not fstat.exists() or not corrp.exists():
        raise RuntimeError("randomise did not create omnibus F outputs")
    thresholded, clusters = render(output, fstat, corrp)
    labels = pd.read_csv(output / "condition_mean_labels.csv")
    summary = pd.DataFrame(
        [
            {
                "analysis": "three_condition_repeated_measures_F",
                "n_subjects": int(labels.subject_id.nunique()),
                "n_conditions": 3,
                "sessions_averaged_per_condition": 4,
                "n_permutations": args.n_permutations,
                "tfce_fwer_threshold": 0.05,
                "seed": args.seed,
                "n_significant_clusters": len(pd.read_csv(clusters)),
                "thresholded_map": str(thresholded),
            }
        ]
    )
    summary.to_csv(output / "omnibus_summary.csv", index=False)
    print(summary.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "activation/repeated_measures")
    )
    p.add_argument(
        "--mask",
        default=str(
            RESULTS_DIR / "activation/group_contrasts/group_intersection_mask.nii.gz"
        ),
    )
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=20260731)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
