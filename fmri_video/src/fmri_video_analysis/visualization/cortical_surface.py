#!/usr/bin/env python3
"""Render the four-session group activation map on a bilateral pycortex flatmap."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, surface


from fmri_video_analysis.core.config import RESULTS_DIR

DEFAULT_ACTIVATION_DIR = RESULTS_DIR / "activation/group_contrasts"
DEFAULT_MAP = DEFAULT_ACTIVATION_DIR / "complex_minus_simple_tfce_fwer05_tstat.nii.gz"
DEFAULT_OUTPUT = (
    DEFAULT_ACTIVATION_DIR / "complex_minus_simple_tfce_fwer05_pycortex_flatmap"
)


def load_cortex(subject: str):
    try:
        import cortex
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装 pycortex；请先激活包含 pycortex 的环境。"
        ) from exc
    cortex.db.get_surf(subject, "flat", merge=True)
    return cortex


def project_to_fsaverage5(volume_path: Path) -> np.ndarray:
    """Project an MNI statistical volume to fsaverage5 (LH then RH)."""
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    image = nib.load(str(volume_path))
    hemispheres = []
    for hemisphere in ("left", "right"):
        values = surface.vol_to_surf(
            image,
            fsaverage5[f"pial_{hemisphere}"],
            inner_mesh=fsaverage5[f"white_{hemisphere}"],
            kind="depth",
            n_samples=5,
            interpolation="linear",
        )
        hemispheres.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(hemispheres)


def run(args: argparse.Namespace):
    cortex = load_cortex(args.pycortex_subject)
    source_subject = Path(args.subjects_dir) / args.freesurfer_subject
    if not source_subject.is_dir():
        raise FileNotFoundError(
            f"FreeSurfer source subject not found: {source_subject}"
        )

    values = project_to_fsaverage5(Path(args.input_map))
    # The input is already TFCE-FWER thresholded. Zeros are background rather
    # than t=0 observations and must remain transparent on the surface.
    values[values < args.threshold] = np.nan
    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError("The thresholded activation map contains no surface values.")
    vmax = args.vmax or float(np.ceil(np.nanpercentile(values[finite], 99)))
    vmax = max(vmax, float(np.nanmax(values[finite])) * 0.5)

    full_values = cortex.freesurfer.upsample_to_fsaverage(
        values,
        args.freesurfer_subject,
        freesurfer_subjects_dir=str(Path(args.subjects_dir).resolve()),
    )
    vertex = cortex.Vertex(
        full_values,
        args.pycortex_subject,
        vmin=float(args.vmin),
        vmax=float(vmax),
        cmap=args.cmap,
    )
    inkscape_available = shutil.which("inkscape") is not None
    figure = cortex.quickflat.make_figure(
        vertex,
        with_curvature=True,
        with_colorbar=False,
        with_rois=bool(args.with_rois and inkscape_available),
        recache=bool(args.recache),
        with_sulci=inkscape_available,
        with_labels=False,
        with_borders=inkscape_available,
        with_dropout=False,
        extra_disp=None,
        with_connected_vertices=False,
        overlay_file=None,
    )
    figure.set_size_inches(14, 5.8)
    figure.axes[0].set_position([0.015, 0.19, 0.97, 0.66])
    figure.suptitle(args.title, fontsize=20, fontweight="bold", y=0.965)

    colorbar_axis = figure.add_axes([0.34, 0.075, 0.32, 0.032])
    scalar_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=float(args.vmin), vmax=float(vmax)),
        cmap=plt.get_cmap(args.cmap),
    )
    scalar_map.set_array([])
    colorbar = figure.colorbar(
        scalar_map,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=np.linspace(float(args.vmin), float(vmax), 6),
    )
    colorbar.set_label(args.colorbar_label, fontsize=14, labelpad=3)
    colorbar.ax.tick_params(labelsize=12, length=3, pad=2)

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in args.formats:
        path = output_stem.with_suffix(f".{suffix}")
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            options["dpi"] = args.dpi
        figure.savefig(path, **options)
        print(f"Saved: {path}")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-map", default=str(DEFAULT_MAP))
    p.add_argument("--output-stem", default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--title",
        default="Complex minus Simple | four-session group activation",
    )
    p.add_argument("--pycortex-subject", default="fsaverage")
    p.add_argument("--freesurfer-subject", default="fsaverage5")
    p.add_argument("--subjects-dir", default="/usr/local/freesurfer/subjects")
    p.add_argument("--vmin", type=float, default=2.5)
    p.add_argument("--vmax", type=float)
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Hide projected surface values below this threshold.",
    )
    p.add_argument("--cmap", default="inferno")
    p.add_argument(
        "--colorbar-label",
        default="t value (TFCE-FWER p < .05)",
    )
    p.add_argument(
        "--with-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--recache", action="store_true")
    p.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"]
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
