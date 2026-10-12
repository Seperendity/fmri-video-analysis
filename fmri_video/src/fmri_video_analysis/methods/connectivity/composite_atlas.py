"""Build the frozen Schaefer-200 + Tian S2 + Buckner-17 atlas."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

YEO17_NETWORKS = (
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schaefer_labels(order_path: Path) -> dict[int, dict[str, str]]:
    labels: dict[int, dict[str, str]] = {}
    for line in order_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        label_id, name = int(fields[0]), fields[1]
        match = re.match(r"17Networks_([LR])H_([^_]+)", name)
        if match is None:
            raise ValueError(f"Cannot parse Schaefer label name: {name}")
        labels[label_id] = {
            "name": name,
            "hemisphere": {"L": "Left", "R": "Right"}[match.group(1)],
            "network": match.group(2),
        }
    if set(labels) != set(range(1, 201)):
        raise ValueError("Schaefer order file must define labels 1 through 200.")
    return labels


def _tian_labels(label_path: Path) -> dict[int, dict[str, str]]:
    names = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != 32:
        raise ValueError(f"Tian S2 must contain 32 labels, found {len(names)}.")
    return {
        index: {
            "name": name,
            "hemisphere": (
                "Right" if name.endswith("-rh") else "Left" if name.endswith("-lh") else "Unknown"
            ),
            "network": "Subcortical",
        }
        for index, name in enumerate(names, 1)
    }


def _buckner_labels(table_path: Path) -> dict[int, dict[str, str]]:
    table = pd.read_csv(table_path, sep="\t")
    if set(table["index"].astype(int)) != set(range(1, 18)):
        raise ValueError("Buckner table must define labels 1 through 17.")
    return {
        int(row["index"]): {
            "name": str(row["name"]),
            "hemisphere": "Bilateral",
            "network": YEO17_NETWORKS[int(row["index"]) - 1],
        }
        for _, row in table.iterrows()
    }


def source_resources(atlas_root: Path) -> list[dict]:
    schaefer_root = atlas_root / "schaefer_2018"
    tian_root = atlas_root / "tian_2020_s2"
    buckner_root = atlas_root / "buckner_2011"
    return [
        {
            "key": "schaefer",
            "source_atlas": "Schaefer2018_200Parcels_17Networks",
            "domain": "Cortex",
            "image": schaefer_root
            / "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
            "labels_file": schaefer_root
            / "Schaefer2018_200Parcels_17Networks_order.txt",
            "labels": _schaefer_labels(
                schaefer_root
                / "Schaefer2018_200Parcels_17Networks_order.txt"
            ),
        },
        {
            "key": "tian",
            "source_atlas": "Tian2020MSA_v1.4_ScaleII_3T",
            "domain": "Subcortex",
            "image": tian_root / "Tian_Subcortex_S2_3T.nii.gz",
            "labels_file": tian_root / "Tian_Subcortex_S2_3T_label.txt",
            "labels": _tian_labels(
                tian_root / "Tian_Subcortex_S2_3T_label.txt"
            ),
        },
        {
            "key": "buckner",
            "source_atlas": "Buckner2011_17Networks",
            "domain": "Cerebellum",
            "image": buckner_root / "atl-Buckner17_space-MNI_dseg.nii",
            "labels_file": buckner_root / "atl-Buckner17.tsv",
            "labels": _buckner_labels(buckner_root / "atl-Buckner17.tsv"),
        },
    ]


def _reference_3d(reference_path: Path) -> nib.Nifti1Image:
    image = nib.load(str(reference_path))
    if len(image.shape) not in {3, 4}:
        raise ValueError(f"Reference image must be 3D or 4D, got {image.shape}.")
    header = image.header.copy()
    header.set_data_shape(image.shape[:3])
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(
        np.zeros(image.shape[:3], dtype=np.uint8), image.affine, header
    )


def _resampled_labels(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    resampled = resample_to_img(
        nib.load(str(path)),
        reference,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.rint(np.asarray(resampled.dataobj)).astype(np.int16)
    if np.any(data < 0):
        raise ValueError(f"Negative labels found in {path}.")
    return data


def _centroid(mask: np.ndarray, affine: np.ndarray) -> tuple[float, float, float]:
    ijk = np.argwhere(mask).mean(axis=0)
    xyz = nib.affines.apply_affine(affine, ijk)
    return tuple(float(value) for value in xyz)


def build_composite_atlas(
    reference_path: Path,
    atlas_root: Path,
    output_dir: Path,
    small_roi_warning_voxels: int = 10,
) -> tuple[Path, pd.DataFrame, dict]:
    """Build and validate a mutually exclusive 249-node atlas at reference grid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _reference_3d(reference_path)
    resources = source_resources(atlas_root)
    label_arrays = {
        resource["key"]: _resampled_labels(resource["image"], reference)
        for resource in resources
    }

    expected = {"schaefer": 200, "tian": 32, "buckner": 17}
    for key, n_labels in expected.items():
        observed = set(np.unique(label_arrays[key]).tolist()) - {0}
        wanted = set(range(1, n_labels + 1))
        if observed != wanted:
            raise RuntimeError(
                f"{key} resampling lost labels: missing={sorted(wanted - observed)}, "
                f"unexpected={sorted(observed - wanted)}"
            )

    occupied = {key: values > 0 for key, values in label_arrays.items()}
    raw_overlap = {
        "schaefer_tian": int(np.sum(occupied["schaefer"] & occupied["tian"])),
        "schaefer_buckner": int(
            np.sum(occupied["schaefer"] & occupied["buckner"])
        ),
        "tian_buckner": int(np.sum(occupied["tian"] & occupied["buckner"])),
    }

    # Anatomical-domain priority is explicit: subcortex, then cerebellum, then
    # cortex. This removes boundary collisions while preserving every source ROI.
    domain_masks = {
        "tian": occupied["tian"],
        "buckner": occupied["buckner"] & ~occupied["tian"],
        "schaefer": occupied["schaefer"]
        & ~occupied["tian"]
        & ~occupied["buckner"],
    }

    composite = np.zeros(reference.shape, dtype=np.int16)
    rows: list[dict] = []
    next_node_id = 1
    for resource in resources:
        key = resource["key"]
        source_labels = label_arrays[key]
        for source_label_id in sorted(resource["labels"]):
            raw_mask = source_labels == source_label_id
            final_mask = raw_mask & domain_masks[key]
            raw_n_voxels = int(raw_mask.sum())
            n_voxels = int(final_mask.sum())
            if n_voxels == 0:
                raise RuntimeError(
                    f"{resource['source_atlas']} label {source_label_id} is empty "
                    "after overlap resolution."
                )
            if np.any(composite[final_mask] != 0):
                raise RuntimeError("Composite atlas contains an unresolved overlap.")
            composite[final_mask] = next_node_id
            label = resource["labels"][source_label_id]
            centroid_x, centroid_y, centroid_z = _centroid(
                final_mask, reference.affine
            )
            rows.append(
                {
                    "node_id": next_node_id,
                    "source_atlas": resource["source_atlas"],
                    "source_label_id": source_label_id,
                    "node_name": label["name"],
                    "hemisphere": label["hemisphere"],
                    "network": label["network"],
                    "domain": resource["domain"],
                    "raw_n_voxels": raw_n_voxels,
                    "overlap_voxels_removed": raw_n_voxels - n_voxels,
                    "n_voxels": n_voxels,
                    "centroid_x_mni": centroid_x,
                    "centroid_y_mni": centroid_y,
                    "centroid_z_mni": centroid_z,
                    "small_roi_warning": n_voxels < small_roi_warning_voxels,
                }
            )
            next_node_id += 1

    nodes = pd.DataFrame(rows)
    observed_nodes = set(np.unique(composite).tolist()) - {0}
    if len(nodes) != 249 or observed_nodes != set(range(1, 250)):
        raise RuntimeError(
            f"Composite atlas must have labels 1-249; got {len(observed_nodes)}."
        )

    atlas_path = output_dir / "composite_249_3mm.nii.gz"
    atlas_image = nib.Nifti1Image(composite, reference.affine, reference.header)
    atlas_image.set_data_dtype(np.int16)
    nib.save(atlas_image, str(atlas_path))
    nodes_path = output_dir / "composite_249_nodes.tsv"
    nodes.to_csv(nodes_path, sep="\t", index=False)

    source_checksums = {
        str(path.relative_to(atlas_root)): sha256_file(path)
        for resource in resources
        for path in (resource["image"], resource["labels_file"])
    }
    qc = {
        "reference_image": str(reference_path),
        "reference_shape": list(reference.shape),
        "reference_zooms_mm": [
            float(value) for value in reference.header.get_zooms()[:3]
        ],
        "node_count": len(nodes),
        "edge_count": len(nodes) * (len(nodes) - 1) // 2,
        "source_node_counts": {
            str(key): int(value)
            for key, value in nodes.groupby("source_atlas").size().items()
        },
        "raw_overlap_voxels": raw_overlap,
        "final_overlap_voxels": 0,
        "small_roi_warning_threshold_voxels": small_roi_warning_voxels,
        "small_roi_node_ids": nodes.loc[
            nodes["small_roi_warning"], "node_id"
        ].astype(int).tolist(),
        "source_sha256": source_checksums,
        "composite_sha256": sha256_file(atlas_path),
        "nodes_sha256": sha256_file(nodes_path),
    }
    (output_dir / "composite_249_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return atlas_path, nodes, qc


def load_composite_atlas(output_dir: Path) -> tuple[Path, pd.DataFrame]:
    atlas_path = output_dir / "composite_249_3mm.nii.gz"
    nodes_path = output_dir / "composite_249_nodes.tsv"
    if not atlas_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(
            f"Composite atlas is incomplete under {output_dir}; run atlas stage."
        )
    nodes = pd.read_csv(nodes_path, sep="\t")
    if len(nodes) != 249 or nodes["node_id"].tolist() != list(range(1, 250)):
        raise RuntimeError("Stored composite atlas node table is invalid.")
    return atlas_path, nodes
