"""Single command-line entry point for all manuscript-revision analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    ORIGINAL_MVPA_RESULTS_DIR,
    RESULTS_DIR,
    STIMULUS_FRAMES_DIR,
    audit_contrasts,
    build_four_session_manifest,
    write_analysis_config,
)


def audit(
    first_level_dir: Path,
    output_dir: Path,
    *,
    stimulus_frames_dir: Path,
    timeseries_root: Path | None,
):
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_four_session_manifest(first_level_dir)
    contrasts = audit_contrasts(first_level_dir)
    manifest.to_csv(metadata_dir / "four_session_beta_manifest.csv", index=False)
    contrasts.to_csv(metadata_dir / "four_session_contrast_manifest.csv", index=False)
    write_analysis_config(
        metadata_dir,
        first_level_dir=first_level_dir,
        stimulus_frames_dir=stimulus_frames_dir,
        timeseries_root=timeseries_root,
        subject_ids=tuple(sorted(manifest.subject_id.unique())),
    )
    print(
        f"Audit passed: {manifest.subject_id.nunique()} subjects, "
        f"{len(manifest)} beta maps, {len(contrasts)} contrast maps."
    )


def run(args):
    output_dir = Path(args.output_dir)
    first_level_dir = Path(args.first_level_dir).resolve()
    stimulus_frames_dir = Path(args.stimulus_frames_dir).resolve()
    timeseries_root = (
        Path(args.timeseries_root).resolve() if args.timeseries_root else None
    )
    audit(
        first_level_dir,
        output_dir,
        stimulus_frames_dir=stimulus_frames_dir,
        timeseries_root=timeseries_root,
    )
    analysis_mask = (
        output_dir / "activation/group_contrasts/group_intersection_mask.nii.gz"
    )
    if (
        args.analysis in {"omnibus", "searchlight", "expanded"}
        and not analysis_mask.exists()
    ):
        raise FileNotFoundError(
            f"Required analysis mask is missing from this run: {analysis_mask}. "
            "Run --analysis activation with the same --output-dir first."
        )
    if args.analysis in {"stimulus", "all-available", "all"}:
        from fmri_video_analysis.methods.stimulus import (
            characterization as stimulus_analysis,
        )

        stimulus_analysis.run(
            argparse.Namespace(
                stimulus_frames_dir=args.stimulus_frames_dir,
                output_dir=str(output_dir / "stimulus/characterization"),
                skip_detection=args.skip_detection,
                detection_every=args.detection_every,
                detection_threshold=args.detection_threshold,
                force=args.force,
            )
        )
    if args.analysis in {"activation", "all-available", "all"}:
        from fmri_video_analysis.methods.activation import (
            group_contrasts as group_activation,
        )

        group_activation.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "activation/group_contrasts"),
                contrasts=args.activation_contrasts,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                seed=args.seed,
                skip_randomise=args.skip_randomise,
                force=args.force,
            )
        )
    if args.analysis in {"mvpa-validation", "all-available", "all"}:
        from fmri_video_analysis.methods.decoding import cross_video as mvpa_cross_video

        mvpa_cross_video.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "decoding/cross_video"),
                subject_folds=args.subject_folds,
                bootstrap=args.bootstrap,
                n_permutations=args.n_permutations,
                structured_permutations=args.structured_permutations,
                permutation_jobs=args.permutation_jobs,
                structured_permutation_batch_size=args.structured_permutation_batch_size,
                seed=args.seed,
                include_mlp=args.include_mlp,
            )
        )
    if args.analysis in {"omnibus", "expanded", "all"}:
        if not analysis_mask.exists():
            raise FileNotFoundError(
                f"Required analysis mask is missing from this run: {analysis_mask}. "
                "Run --analysis activation with the same --output-dir first."
            )
        from fmri_video_analysis.methods.activation import (
            repeated_measures as repeated_measures_f,
        )

        repeated_measures_f.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "activation/repeated_measures"),
                mask=str(analysis_mask),
                n_permutations=args.n_permutations,
                force=args.force,
                seed=args.seed,
            )
        )
    if args.analysis in {"mvpa-threeclass", "expanded", "all"}:
        from fmri_video_analysis.methods.decoding import (
            three_class_networks as mvpa_threeclass_networks,
        )

        mvpa_threeclass_networks.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "decoding/three_class_networks"),
                subject_folds=args.subject_folds,
                bootstrap=args.bootstrap,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
        )
    if args.analysis in {"rsa", "expanded", "all"}:
        from fmri_video_analysis.methods.rsa import crossnobis as crossnobis_rsa

        crossnobis_rsa.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "rsa/crossnobis"),
                diagonal_shrinkage=args.diagonal_shrinkage,
                spatial_mean_center=False,
                bootstrap=args.bootstrap,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
        )
    if args.analysis in {"searchlight", "expanded", "all"}:
        from fmri_video_analysis.methods.rsa import (
            searchlight as searchlight_crossnobis,
        )

        if not analysis_mask.exists():
            raise FileNotFoundError(
                f"Required analysis mask is missing from this run: {analysis_mask}. "
                "Run --analysis activation with the same --output-dir first."
            )
        searchlight_crossnobis.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "rsa/searchlight"),
                mask=str(analysis_mask),
                radius_mm=args.searchlight_radius_mm,
                diagonal_shrinkage=args.diagonal_shrinkage,
                n_permutations=args.n_permutations,
                force=args.force,
                seed=args.seed,
            )
        )
    if args.analysis in {"embedding-sensitivity", "supplementary", "all"}:
        from fmri_video_analysis.methods.embedding import (
            sensitivity as embedding_sensitivity,
        )

        embedding_sensitivity.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "embedding/sensitivity"),
                pca_components=50,
                tsne_perplexities=[15, 30, 50],
                umap_neighbors=[10, 30, 50],
                umap_min_dist=[0.05, 0.3, 0.8],
                seed=args.seed,
            )
        )
    if args.analysis in {"classifier-benchmark", "supplementary", "all"}:
        from fmri_video_analysis.methods.decoding import classifier_benchmark

        classifier_benchmark.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "decoding/classifier_benchmark"),
                k_features=300,
                subject_folds=args.subject_folds,
                bootstrap=args.bootstrap,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
        )
    if args.analysis in {"exploratory-voxels", "supplementary", "all"}:
        from fmri_video_analysis.methods.decoding import exploratory_voxel_weights

        exploratory_voxel_weights.run(
            argparse.Namespace(
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "decoding/exploratory_voxel_weights"),
                candidate_voxels=100,
                weight_voxels=300,
                minimum_improvement=1e-4,
                n_jobs=args.n_jobs,
            )
        )
    if args.analysis in {"beta-connectivity", "supplementary", "all"}:
        from fmri_video_analysis.methods.connectivity import (
            beta_series as beta_series_connectivity,
        )

        beta_series_connectivity.run(
            argparse.Namespace(
                parcel_beta_values=str(
                    Path(args.original_mvpa_results_dir)
                    / "descriptive_responses/parcel_beta_values.csv"
                ),
                output_dir=str(output_dir / "connectivity/beta_series"),
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
        )
    if args.analysis in {"posthoc-roi", "supplementary", "all"}:
        from fmri_video_analysis.methods.activation import (
            roi_screening as posthoc_roi_screening,
        )

        posthoc_roi_screening.run(
            argparse.Namespace(
                descriptive_root=str(
                    Path(args.original_mvpa_results_dir) / "descriptive_responses"
                ),
                output_dir=str(output_dir / "activation/roi_screening"),
                top_n=30,
            )
        )
    if args.analysis in {"connectivity", "topology", "all"}:
        from fmri_video_analysis.methods.connectivity import (
            timeseries_topology as connectivity_topology,
        )

        connectivity_topology.run(
            argparse.Namespace(
                analysis=("all" if args.analysis == "all" else args.analysis),
                timeseries_root=args.timeseries_root,
                first_level_dir=args.first_level_dir,
                output_dir=str(output_dir / "connectivity/timeseries"),
                hrf_lag_trs=args.hrf_lag_trs,
                min_roi_voxels=args.min_roi_voxels,
                n_permutations=args.n_permutations,
                equivalence_bound_dz=args.equivalence_bound_dz,
            )
        )


def parser():
    p = argparse.ArgumentParser(
        description=(
            "Four-session stimulus, activation, MVPA, connectivity and topology "
            "analyses for the manuscript revision."
        )
    )
    p.add_argument(
        "--analysis",
        choices=[
            "audit",
            "stimulus",
            "activation",
            "mvpa-validation",
            "omnibus",
            "mvpa-threeclass",
            "rsa",
            "searchlight",
            "expanded",
            "embedding-sensitivity",
            "classifier-benchmark",
            "exploratory-voxels",
            "beta-connectivity",
            "posthoc-roi",
            "supplementary",
            "connectivity",
            "topology",
            "all-available",
            "all",
        ],
        default="all-available",
    )
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--stimulus-frames-dir",
        "--pic-dir",
        dest="stimulus_frames_dir",
        default=str(STIMULUS_FRAMES_DIR),
    )
    p.add_argument(
        "--original-mvpa-results-dir", default=str(ORIGINAL_MVPA_RESULTS_DIR)
    )
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--timeseries-root")
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--structured-permutations", type=int, default=0)
    p.add_argument("--permutation-jobs", type=int, default=1)
    p.add_argument("--structured-permutation-batch-size", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--subject-folds", type=int, default=5)
    p.add_argument(
        "--activation-contrasts",
        nargs="+",
        default=["simple", "complex", "control", "complex_minus_simple"],
    )
    p.add_argument("--skip-randomise", action="store_true")
    p.add_argument("--skip-detection", action="store_true")
    p.add_argument("--detection-every", type=int, default=25)
    p.add_argument("--detection-threshold", type=float, default=0.35)
    p.add_argument("--include-mlp", action="store_true")
    p.add_argument("--hrf-lag-trs", type=int, default=0)
    p.add_argument("--min-roi-voxels", type=int, default=20)
    p.add_argument("--equivalence-bound-dz", type=float, default=0.2)
    p.add_argument("--diagonal-shrinkage", type=float, default=0.2)
    p.add_argument("--searchlight-radius-mm", type=float, default=9.0)
    p.add_argument("--force", action="store_true")
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
