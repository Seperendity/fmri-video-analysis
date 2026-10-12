#!/usr/bin/env python3
"""Four-session group activation with sign-flip TFCE-FWER inference."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.image import resample_to_img
from scipy import ndimage

from fmri_video_analysis.core.config import (
    ATLAS_DIR,
    CONTRAST_FILES,
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    audit_contrasts,
)
from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    require_matching_cache,
)
from fmri_video_analysis.core.validation import validate_nifti_geometry

MAIN_CONTRASTS = ("simple", "complex", "control", "complex_minus_simple")


def merge_contrast(table: pd.DataFrame, contrast: str, output_dir: Path) -> Path:
    subset = table.loc[table.contrast == contrast].sort_values("subject_id")
    paths = [Path(p) for p in subset.contrast_file]
    sign = float(subset.sign.iloc[0])
    reference = nib.load(str(paths[0]))
    arrays = []
    for path in paths:
        img = nib.load(str(path))
        validate_nifti_geometry(
            reference,
            img,
            context=f"group contrast {contrast}: {path}",
        )
        arrays.append(np.asarray(img.dataobj, dtype=np.float32) * sign)
    data = np.stack(arrays, axis=3)
    merged = output_dir / f"{contrast}_subjects_4d.nii.gz"
    nib.save(nib.Nifti1Image(data, reference.affine, reference.header), merged)
    return merged


def intersection_mask(merged_paths: list[Path], output_path: Path) -> Path:
    mask = None
    reference = None
    for path in merged_paths:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
        current = np.all(np.isfinite(data), axis=3) & np.any(data != 0, axis=3)
        mask = current if mask is None else mask & current
        reference = img
    if not np.any(mask):
        raise RuntimeError("Intersection mask is empty.")
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header),
        output_path,
    )
    return output_path


def descriptive_maps(merged_path: Path, output_dir: Path, contrast: str):
    img = nib.load(str(merged_path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    mean = data.mean(axis=3)
    sd = data.std(axis=3, ddof=1)
    tmap = np.divide(
        mean,
        sd / np.sqrt(data.shape[3]),
        out=np.zeros_like(mean),
        where=sd > 0,
    )
    dz = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    for suffix, array in (
        ("mean_effect", mean),
        ("tstat_parametric", tmap),
        ("cohens_dz", dz),
    ):
        nib.save(
            nib.Nifti1Image(array.astype(np.float32), img.affine, img.header),
            output_dir / f"{contrast}_{suffix}.nii.gz",
        )


def run_randomise(
    merged_path: Path,
    mask_path: Path,
    output_dir: Path,
    contrast: str,
    n_permutations: int,
    n_jobs: int,
    seed: int,
    force: bool,
) -> Path:
    randomise = shutil.which("randomise") or "/usr/local/fsl/bin/randomise"
    prefix = output_dir / contrast
    corrected = output_dir / f"{contrast}_tfce_corrp_tstat1.nii.gz"
    manifest_path = output_dir / f"{contrast}_randomise_cache_manifest.json"
    expected_manifest = cache_manifest(
        stage=f"activation.group_contrasts.randomise.{contrast}",
        parameters={
            "n_permutations": n_permutations,
            "seed": seed,
            "one_sample": True,
            "tfce": True,
        },
        inputs={
            "merged": file_fingerprint(merged_path, full_hash=True),
            "mask": file_fingerprint(mask_path, full_hash=True),
        },
    )
    if require_matching_cache(
        manifest_path,
        expected_manifest,
        artifacts_exist=corrected.exists(),
        force=force,
    ):
        return corrected
    command = [
        randomise,
        "-i",
        str(merged_path),
        "-o",
        str(prefix),
        "-m",
        str(mask_path),
        "-1",
        "-T",
        "-n",
        str(n_permutations),
        f"--seed={seed}",
    ]
    if n_jobs > 1:
        print(
            "Note: local randomise is single-process; --n-jobs is recorded but "
            "parallelise is not available.",
            flush=True,
        )
    subprocess.run(command, check=True)
    if not corrected.exists():
        raise RuntimeError(f"randomise did not create {corrected}")
    atomic_write_json(manifest_path, expected_manifest)
    return corrected


def schaefer_labels() -> tuple[Path, dict[int, str]]:
    root = ATLAS_DIR / "schaefer_2018"
    atlas = root / "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    order = root / "Schaefer2018_200Parcels_17Networks_order.txt"
    labels = {}
    for line in order.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            labels[int(fields[0])] = fields[1]
    return atlas, labels


def parse_fsl_atlas_xml(xml_path: Path) -> tuple[Path | None, dict[int, str]]:
    root = ET.parse(xml_path).getroot()
    labels = {}
    for node in root.findall(".//label"):
        index = int(node.attrib["index"]) + 1
        labels[index] = (node.text or "").strip()
    image_path = None
    images = root.findall(".//summaryimagefile")
    images = sorted(images, key=lambda node: "2mm" not in (node.text or ""))
    for image in images:
        value = (image.text or "").strip()
        candidate = Path("/usr/local/fsl/data/atlases") / value.lstrip("/")
        for suffix in ("", ".nii.gz"):
            p = Path(str(candidate) + suffix)
            if p.exists():
                image_path = p
                break
        if image_path:
            break
    return image_path, labels


def label_coordinate(
    voxel: tuple[int, int, int],
    reference_img: nib.Nifti1Image,
    atlases: list[tuple[np.ndarray, dict[int, str], str]],
) -> str:
    world = nib.affines.apply_affine(reference_img.affine, voxel)
    for data, labels, prefix in atlases:
        atlas_voxel = np.rint(
            nib.affines.apply_affine(np.linalg.inv(reference_img.affine), world)
        ).astype(int)
        if np.all(atlas_voxel >= 0) and np.all(atlas_voxel < data.shape):
            value = int(data[tuple(atlas_voxel)])
            if value and value in labels:
                return f"{prefix}:{labels[value]}"
    return "Unlabelled"


def resampled_label_atlases(reference_img):
    resources = []
    schaefer_path, schaefer_names = schaefer_labels()
    items = [(schaefer_path, schaefer_names, "Schaefer200")]
    for xml_name, prefix in (
        ("HarvardOxford-Subcortical.xml", "HarvardOxford"),
        ("Cerebellum_MNIfnirt.xml", "Cerebellum"),
    ):
        path, labels = parse_fsl_atlas_xml(
            Path("/usr/local/fsl/data/atlases") / xml_name
        )
        if path:
            items.append((path, labels, prefix))
    for path, labels, prefix in items:
        resampled = resample_to_img(
            nib.load(str(path)), reference_img, interpolation="nearest"
        )
        resources.append(
            (
                np.rint(np.asarray(resampled.dataobj)).astype(int),
                labels,
                prefix,
            )
        )
    return resources


def cluster_table(
    contrast: str,
    t_path: Path,
    corrected_path: Path,
    output_path: Path,
    threshold: float = 0.95,
):
    t_img = nib.load(str(t_path))
    t = np.asarray(t_img.dataobj)
    corrp = np.asarray(nib.load(str(corrected_path)).dataobj)
    significant = corrp >= threshold
    labelled, n_clusters = ndimage.label(significant, structure=np.ones((3, 3, 3)))
    atlases = resampled_label_atlases(t_img)
    rows = []
    voxel_volume = abs(np.linalg.det(t_img.affine[:3, :3]))
    for cluster_id in range(1, n_clusters + 1):
        voxels = np.argwhere(labelled == cluster_id)
        if not len(voxels):
            continue
        peak_voxel = tuple(voxels[np.argmax(t[tuple(voxels.T)])])
        xyz = nib.affines.apply_affine(t_img.affine, peak_voxel)
        rows.append(
            {
                "contrast": contrast,
                "cluster_id": cluster_id,
                "n_voxels": len(voxels),
                "volume_mm3": len(voxels) * voxel_volume,
                "peak_t": float(t[peak_voxel]),
                "peak_corrp_1_minus_p": float(corrp[peak_voxel]),
                "mni_x": xyz[0],
                "mni_y": xyz[1],
                "mni_z": xyz[2],
                "atlas_label": label_coordinate(peak_voxel, t_img, atlases),
            }
        )
    columns = [
        "contrast",
        "cluster_id",
        "n_voxels",
        "volume_mm3",
        "peak_t",
        "peak_corrp_1_minus_p",
        "mni_x",
        "mni_y",
        "mni_z",
        "atlas_label",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values("peak_t", ascending=False)
    frame.to_csv(output_path, index=False)


def save_figure(contrast: str, t_path: Path, corrected_path: Path, output_dir: Path):
    t_img = nib.load(str(t_path))
    t = np.asarray(t_img.dataobj)
    corrp = np.asarray(nib.load(str(corrected_path)).dataobj)
    thresholded = np.where(corrp >= 0.95, t, 0)
    out_img = nib.Nifti1Image(
        thresholded.astype(np.float32), t_img.affine, t_img.header
    )
    thresholded_path = output_dir / f"{contrast}_tfce_fwer05_tstat.nii.gz"
    nib.save(out_img, thresholded_path)
    title = contrast.replace("_", " ").title() + " | TFCE-FWER p < .05"
    for display, suffix in (("ortho", "glass"), ("z", "slices")):
        fig = plt.figure(figsize=(12, 5))
        if display == "ortho":
            plotting.plot_glass_brain(
                out_img,
                display_mode="lyrz",
                colorbar=True,
                plot_abs=False,
                cmap="cold_hot",
                title=title,
                figure=fig,
            )
        else:
            plotting.plot_stat_map(
                out_img,
                display_mode="z",
                cut_coords=9,
                colorbar=True,
                cmap="cold_hot",
                title=title,
                figure=fig,
            )
        fig.savefig(
            output_dir / f"{contrast}_{suffix}.png", dpi=300, bbox_inches="tight"
        )
        fig.savefig(output_dir / f"{contrast}_{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
    return thresholded_path


def run(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = audit_contrasts(Path(args.first_level_dir))
    table.to_csv(output_dir / "contrast_audit.csv", index=False)
    contrasts = tuple(args.contrasts)
    merged = {c: merge_contrast(table, c, output_dir) for c in contrasts}
    n_subjects = int(table.subject_id.nunique())
    mask = intersection_mask(
        list(merged.values()), output_dir / "group_intersection_mask.nii.gz"
    )
    summary_rows = []
    for contrast, merged_path in merged.items():
        descriptive_maps(merged_path, output_dir, contrast)
        t_path = output_dir / f"{contrast}_tstat_parametric.nii.gz"
        corrected = output_dir / f"{contrast}_tfce_corrp_tstat1.nii.gz"
        if args.skip_randomise:
            summary_rows.append(
                {
                    "contrast": contrast,
                    "status": "descriptive_only",
                    "n_subjects": n_subjects,
                }
            )
            continue
        corrected = run_randomise(
            merged_path,
            mask,
            output_dir,
            contrast,
            args.n_permutations,
            args.n_jobs,
            args.seed,
            args.force,
        )
        thresholded = save_figure(contrast, t_path, corrected, output_dir)
        cluster_csv = output_dir / f"{contrast}_clusters_tfce_fwer05.csv"
        cluster_table(contrast, t_path, corrected, cluster_csv)
        n_clusters = len(pd.read_csv(cluster_csv)) if cluster_csv.stat().st_size else 0
        summary_rows.append(
            {
                "contrast": contrast,
                "status": "complete",
                "n_subjects": n_subjects,
                "n_permutations": args.n_permutations,
                "n_significant_clusters": n_clusters,
                "thresholded_map": str(thresholded),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "activation_summary.csv", index=False
    )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "activation/group_contrasts")
    )
    p.add_argument(
        "--contrasts",
        nargs="+",
        choices=list(CONTRAST_FILES),
        default=list(MAIN_CONTRASTS),
    )
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--skip-randomise", action="store_true")
    p.add_argument("--force", action="store_true")
    return p

if __name__ == "__main__":
    run(parser().parse_args())
