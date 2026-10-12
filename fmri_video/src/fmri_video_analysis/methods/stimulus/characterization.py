"""Quantify low-level and semantic properties of the 12 analysed videos."""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from fmri_video_analysis.core.config import (
    EXPECTED_FRAMES_PER_VIDEO,
    RESULTS_DIR,
    SESSION_VIDEOS,
    STIMULUS_FRAMES_DIR,
)
from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    file_fingerprint,
    manifest_differences,
)
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    exact_label_permutation_anova,
    exact_two_group_permutation,
)

CONDITION_COLORS = {
    "Simple": "#4C78A8",
    "Complex": "#E45756",
    "Control": "#72B7B2",
}
CONDITION_ORDER = ("Simple", "Complex", "Control")
SEMANTIC_SCHEMA_VERSION = 2
TARGET_LABELS = {"car", "motorcycle", "bus", "truck", "bicycle", "person"}
VEHICLE_LABELS = TARGET_LABELS - {"person"}


def selected_videos() -> pd.DataFrame:
    rows = []
    for session, condition_map in SESSION_VIDEOS.items():
        for condition, video_id in condition_map.items():
            rows.append(
                {
                    "session": session,
                    "condition": condition,
                    "video_id": video_id,
                }
            )
    return pd.DataFrame(rows).sort_values(["session", "condition"])


def frame_paths(pic_dir: Path, video_id: str) -> list[Path]:
    paths = list(pic_dir.glob(f"{video_id}_*.jpg"))
    parsed = []
    for path in paths:
        try:
            index = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        parsed.append((index, path))
    parsed.sort()
    indices = [i for i, _ in parsed]
    expected = list(range(EXPECTED_FRAMES_PER_VIDEO))
    if indices != expected:
        raise RuntimeError(
            f"Video {video_id}: expected frames 0..{EXPECTED_FRAMES_PER_VIDEO - 1}, "
            f"found {len(indices)} frames with range "
            f"{indices[:1]}..{indices[-1:] if indices else []}."
        )
    return [p for _, p in parsed]


def grayscale_small(path: Path, size=(256, 192)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def entropy_8bit(gray: np.ndarray) -> float:
    hist = np.bincount(
        np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8).ravel(),
        minlength=256,
    ).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def edge_density(gray: np.ndarray) -> float:
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    magnitude = np.hypot(gx, gy)
    return float(np.mean(magnitude > 0.08))


def motion_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    flow_size: tuple[int, int] = (128, 96),
) -> dict[str, float]:
    """Estimate dense flow and a robust global-translation motion proxy."""

    previous_small = np.clip(
        np.rint(cv2.resize(previous, flow_size) * 255), 0, 255
    ).astype(np.uint8)
    current_small = np.clip(
        np.rint(cv2.resize(current, flow_size) * 255), 0, 255
    ).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        previous_small,
        current_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_vectors = flow.reshape(-1, 2)
    global_vector = np.median(flow_vectors, axis=0)
    residual = flow_vectors - global_vector
    return {
        "optical_flow_mean": float(
            np.linalg.norm(flow_vectors, axis=1).mean()
        ),
        "global_motion_median": float(np.linalg.norm(global_vector)),
        "residual_flow_mean": float(np.linalg.norm(residual, axis=1).mean()),
    }


def compute_low_level(pic_dir: Path, output_dir: Path) -> pd.DataFrame:
    frame_rows = []
    video_rows = []
    videos = selected_videos()
    for row in videos.itertuples(index=False):
        paths = frame_paths(pic_dir, row.video_id)
        previous = None
        per_video = []
        for frame_index, path in enumerate(paths):
            gray = grayscale_small(path)
            motion = (
                {
                    "optical_flow_mean": np.nan,
                    "global_motion_median": np.nan,
                    "residual_flow_mean": np.nan,
                }
                if previous is None
                else motion_metrics(previous, gray)
            )
            metrics = {
                "luminance": float(gray.mean()),
                "rms_contrast": float(gray.std()),
                "entropy": entropy_8bit(gray),
                "edge_density": edge_density(gray),
                "frame_difference": (
                    np.nan
                    if previous is None
                    else float(np.mean(np.abs(gray - previous)))
                ),
                **motion,
            }
            previous = gray
            per_video.append(metrics)
            frame_rows.append(
                {
                    "session": row.session,
                    "condition": row.condition,
                    "video_id": row.video_id,
                    "frame_index": frame_index,
                    "time_seconds": frame_index / 25.0,
                    **metrics,
                }
            )
        frame_df = pd.DataFrame(per_video)
        summary = {
            "session": row.session,
            "condition": row.condition,
            "video_id": row.video_id,
            "n_frames": len(paths),
        }
        for metric in (
            "luminance",
            "rms_contrast",
            "entropy",
            "edge_density",
            "frame_difference",
            "optical_flow_mean",
            "global_motion_median",
            "residual_flow_mean",
        ):
            summary[f"{metric}_mean"] = float(frame_df[metric].mean())
            summary[f"{metric}_sd"] = float(frame_df[metric].std(ddof=1))
        video_rows.append(summary)
        print(f"Low-level metrics complete: video {row.video_id}", flush=True)

    frame_table = pd.DataFrame(frame_rows)
    video_table = pd.DataFrame(video_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_table.to_csv(output_dir / "frame_level_metrics.csv.gz", index=False)
    video_table.to_csv(output_dir / "video_level_metrics.csv", index=False)
    return video_table


def load_detector():
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(device)
    categories = weights.meta["categories"]
    return torch, model, categories, device


def tiled_boxes(image: Image.Image, tile=512, overlap=96):
    width, height = image.size
    stride = tile - overlap
    xs = sorted(
        set(list(range(0, max(1, width - tile + 1), stride)) + [max(0, width - tile)])
    )
    ys = sorted(
        set(list(range(0, max(1, height - tile + 1), stride)) + [max(0, height - tile)])
    )
    for x in xs:
        for y in ys:
            yield x, y, image.crop((x, y, min(width, x + tile), min(height, y + tile)))


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45):
    if not len(boxes):
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][iou <= threshold]
    return np.asarray(keep, dtype=int)


def detect_frame(image: Image.Image, detector, score_threshold: float):
    torch, model, categories, device = detector
    boxes, scores, labels = [], [], []
    tiles = list(tiled_boxes(image))
    tensors = []
    for _, _, crop in tiles:
        array = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).to(device))
    with torch.inference_mode():
        predictions = model(tensors)
        for (x, y, _), prediction in zip(tiles, predictions):
            keep = prediction["scores"].cpu().numpy() >= score_threshold
            b = prediction["boxes"].cpu().numpy()[keep]
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            boxes.append(b)
            scores.append(prediction["scores"].cpu().numpy()[keep])
            labels.append(prediction["labels"].cpu().numpy()[keep])
    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    detections = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        for chosen in idx[nms_numpy(boxes[idx], scores[idx])]:
            detections.append(
                {
                    "label_id": int(label),
                    "label": categories[int(label)],
                    "score": float(scores[chosen]),
                    "x1": float(boxes[chosen, 0]),
                    "y1": float(boxes[chosen, 1]),
                    "x2": float(boxes[chosen, 2]),
                    "y2": float(boxes[chosen, 3]),
                }
            )
    return detections

def canonical_semantic_tables(
    sampled: pd.DataFrame, detections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the one canonical frame/video semantic schema from detections."""

    base_columns = ["session", "condition", "video_id", "frame_index"]
    missing = sorted(set(base_columns) - set(sampled.columns))
    if missing:
        raise RuntimeError(f"Semantic sampled-frame table is missing columns: {missing}")
    base = sampled[base_columns].drop_duplicates().copy()
    if detections.empty:
        target = detections.copy()
        counts = base.copy()
        for column in (
            "vehicle_count",
            "person_count",
            "target_count",
            "target_category_count",
        ):
            counts[column] = 0
    else:
        required = set(base_columns) | {"label", "score", "x1", "x2", "y1", "y2"}
        missing = sorted(required - set(detections.columns))
        if missing:
            raise RuntimeError(f"Semantic detection table is missing columns: {missing}")
        detections = detections.copy()
        detections["area_ratio"] = (
            (detections.x2 - detections.x1)
            * (detections.y2 - detections.y1)
            / (1024 * 768)
        )
        target = detections[
            detections.label.isin(TARGET_LABELS)
            & (detections.score >= 0.50)
            & (detections.area_ratio <= 0.10)
        ].copy()
        if target.empty:
            counts = base.copy()
            for column in (
                "vehicle_count",
                "person_count",
                "target_count",
                "target_category_count",
            ):
                counts[column] = 0
        else:
            target["is_vehicle"] = target.label.isin(VEHICLE_LABELS).astype(int)
            target["is_person"] = (target.label == "person").astype(int)
            counts = target.groupby(base_columns, as_index=False).agg(
                vehicle_count=("is_vehicle", "sum"),
                person_count=("is_person", "sum"),
                target_count=("label", "size"),
                target_category_count=("label", "nunique"),
            )
            counts = base.merge(counts, on=base_columns, how="left").fillna(0)
    count_columns = [
        "vehicle_count",
        "person_count",
        "target_count",
        "target_category_count",
    ]
    counts[count_columns] = counts[count_columns].astype(int)
    video = counts.groupby(["session", "condition", "video_id"], as_index=False).agg(
        sampled_frames=("frame_index", "size"),
        mean_vehicle_count=("vehicle_count", "mean"),
        max_vehicle_count=("vehicle_count", "max"),
        mean_person_count=("person_count", "mean"),
        max_person_count=("person_count", "max"),
        mean_target_count=("target_count", "mean"),
        mean_target_category_count=("target_category_count", "mean"),
    )
    richness = (
        target.groupby(["session", "condition", "video_id"])["label"]
        .nunique()
        .rename("target_category_richness")
        .reset_index()
        if not target.empty
        else pd.DataFrame(
            columns=["session", "condition", "video_id", "target_category_richness"]
        )
    )
    video = video.merge(
        richness, on=["session", "condition", "video_id"], how="left"
    ).fillna({"target_category_richness": 0})
    video["target_category_richness"] = video.target_category_richness.astype(int)
    return counts, video


def run_semantic_detection(
    pic_dir: Path,
    output_dir: Path,
    sample_every: int = 25,
    score_threshold: float = 0.35,
) -> pd.DataFrame:
    detector = load_detector()
    sampled_rows, detection_rows, qc_items = [], [], []
    for video in selected_videos().itertuples(index=False):
        paths = frame_paths(pic_dir, video.video_id)
        sampled = paths[::sample_every]
        for sample_number, path in enumerate(sampled):
            frame_index = int(path.stem.split("_", 1)[1])
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            detections = detect_frame(image, detector, score_threshold)
            sampled_rows.append(
                {
                    "session": video.session,
                    "condition": video.condition,
                    "video_id": video.video_id,
                    "frame_index": frame_index,
                }
            )
            for detection in detections:
                detection_rows.append(
                    {
                        "session": video.session,
                        "condition": video.condition,
                        "video_id": video.video_id,
                        "frame_index": frame_index,
                        **detection,
                    }
                )
            if sample_number in {0, len(sampled) // 2, len(sampled) - 1}:
                qc_items.append((video, frame_index, image, detections))
        print(f"Semantic detection complete: video {video.video_id}", flush=True)

    sampled_df = pd.DataFrame(sampled_rows)
    detection_df = pd.DataFrame(
        detection_rows,
        columns=[
            "session",
            "condition",
            "video_id",
            "frame_index",
            "label_id",
            "label",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_df.to_csv(output_dir / "semantic_detections.csv", index=False)
    sampled_df, video_semantic = canonical_semantic_tables(
        sampled_df, detection_df
    )
    sampled_df.to_csv(output_dir / "semantic_sampled_frames.csv", index=False)
    video_semantic.to_csv(output_dir / "video_semantic_metrics.csv", index=False)
    save_qc_montage(qc_items, output_dir / "semantic_detection_qc.png")
    pd.DataFrame(
        [
            {
                "video_id": item[0].video_id,
                "condition": item[0].condition,
                "frame_index": item[1],
                "review_status": "PENDING_MANUAL_REVIEW",
                "review_notes": "",
            }
            for item in qc_items
        ]
    ).to_csv(output_dir / "semantic_qc_review.csv", index=False)
    return video_semantic


def save_qc_montage(items, output_path: Path):
    if not items:
        return
    columns = 3
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (video, frame_index, image, detections) in zip(axes, items):
        draw = ImageDraw.Draw(image)
        for d in detections:
            draw.rectangle((d["x1"], d["y1"], d["x2"], d["y2"]), outline="red", width=2)
            draw.text((d["x1"], d["y1"]), d["label"], fill="yellow")
        ax.imshow(image)
        ax.set_title(f"{video.condition} {video.video_id}, frame {frame_index}")
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def rebuild_semantic_tables(output_dir: Path) -> pd.DataFrame:
    """Reapply manuscript-facing target filters without rerunning inference."""

    sampled_path = output_dir / "semantic_sampled_frames.csv"
    detections_path = output_dir / "semantic_detections.csv"
    sampled = pd.read_csv(sampled_path, dtype={"video_id": str})
    sampled["video_id"] = sampled.video_id.str.zfill(3)
    detections = pd.read_csv(detections_path, dtype={"video_id": str})
    detections["video_id"] = detections.video_id.str.zfill(3)
    counts, video = canonical_semantic_tables(sampled, detections)
    counts.to_csv(sampled_path, index=False)
    video.to_csv(output_dir / "video_semantic_metrics.csv", index=False)
    return video


def statistical_summary(combined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    id_columns = {"session", "condition", "video_id", "n_frames", "sampled_frames"}
    metrics = [
        c
        for c in combined.columns
        if c not in id_columns and pd.api.types.is_numeric_dtype(combined[c])
    ]
    omnibus_rows, pairwise_rows = [], []
    for metric in metrics:
        values = combined[metric].to_numpy(float)
        labels = combined["condition"].to_numpy()
        omnibus_rows.append(
            {
                "metric": metric,
                "p_exact_permutation": exact_label_permutation_anova(values, labels),
            }
        )
        for a, b in (
            ("Complex", "Simple"),
            ("Simple", "Control"),
            ("Complex", "Control"),
        ):
            x = combined.loc[combined.condition == a, metric].to_numpy(float)
            y = combined.loc[combined.condition == b, metric].to_numpy(float)
            p, g = exact_two_group_permutation(x, y)
            bootstrap_seed = int(
                (20260731 + zlib.crc32(f"{metric}:{a}:{b}".encode()))
                % (2**32 - 1)
            )
            rng = np.random.default_rng(bootstrap_seed)
            sampled_x = x[rng.integers(0, len(x), size=(5000, len(x)))]
            sampled_y = y[rng.integers(0, len(y), size=(5000, len(y)))]
            bootstrap_difference = sampled_x.mean(axis=1) - sampled_y.mean(
                axis=1
            )
            pooled_sd = np.sqrt(
                (
                    (len(x) - 1) * sampled_x.var(axis=1, ddof=1)
                    + (len(y) - 1) * sampled_y.var(axis=1, ddof=1)
                )
                / (len(x) + len(y) - 2)
            )
            correction = 1 - 3 / (4 * (len(x) + len(y)) - 9)
            bootstrap_g = np.divide(
                bootstrap_difference,
                pooled_sd,
                out=np.full(5000, np.nan),
                where=pooled_sd > 0,
            ) * correction
            difference_ci = np.quantile(
                bootstrap_difference, [0.025, 0.975]
            )
            g_ci = np.nanquantile(bootstrap_g, [0.025, 0.975])
            pairwise_rows.append(
                {
                    "metric": metric,
                    "contrast": f"{a.lower()}_minus_{b.lower()}",
                    "condition_a": a,
                    "condition_b": b,
                    "mean_a": x.mean(),
                    "mean_b": y.mean(),
                    "mean_difference": x.mean() - y.mean(),
                    "mean_difference_bootstrap_95_ci_low": difference_ci[0],
                    "mean_difference_bootstrap_95_ci_high": difference_ci[1],
                    "hedges_g": g,
                    "hedges_g_bootstrap_95_ci_low": g_ci[0],
                    "hedges_g_bootstrap_95_ci_high": g_ci[1],
                    "p_exact_permutation": p,
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["q_fdr"] = bh_fdr(omnibus["p_exact_permutation"])
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["q_fdr_all"] = bh_fdr(pairwise["p_exact_permutation"])
    omnibus.to_csv(output_dir / "stimulus_omnibus_tests.csv", index=False)
    pairwise.to_csv(output_dir / "stimulus_pairwise_tests.csv", index=False)
    return pairwise


def save_summary_figure(combined: pd.DataFrame, output_path: Path):
    metrics = [
        "luminance_mean",
        "rms_contrast_mean",
        "entropy_mean",
        "edge_density_mean",
        "frame_difference_mean",
        "optical_flow_mean_mean",
        "global_motion_median_mean",
        "residual_flow_mean_mean",
    ]
    semantic = [
        c
        for c in ("mean_vehicle_count", "mean_person_count", "target_category_richness")
        if c in combined
    ]
    metrics.extend(semantic)
    fig, axes = plt.subplots(2, math.ceil(len(metrics) / 2), figsize=(16, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        for i, condition in enumerate(CONDITION_ORDER):
            values = combined.loc[combined.condition == condition, metric]
            ax.scatter(
                np.full(len(values), i),
                values,
                color=CONDITION_COLORS[condition],
                s=55,
                zorder=3,
            )
            ax.hlines(
                values.mean(),
                i - 0.25,
                i + 0.25,
                color="black",
                linewidth=2,
            )
        ax.set_xticks(range(3), CONDITION_ORDER)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    fig.suptitle("Stimulus properties for the 12 analysed videos (4 per condition)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_path = output_dir / "video_level_metrics.csv"
    semantic_metadata_path = output_dir / "semantic_cache_metadata.json"
    semantic_metadata = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "stimulus_frames_dir": str(Path(args.stimulus_frames_dir).resolve()),
        "sample_every": args.detection_every,
        "detection_threshold": args.detection_threshold,
        "skip_detection": bool(args.skip_detection),
        "code": file_fingerprint(Path(__file__), full_hash=True),
    }
    cache_artifacts = (
        low_path,
        output_dir / "semantic_detections.csv",
        output_dir / "semantic_sampled_frames.csv",
        output_dir / "video_semantic_metrics.csv",
        output_dir / "stimulus_metrics_combined.csv",
    )
    if getattr(args, "rebuild_semantic_only", False):
        required = cache_artifacts[:3]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot rebuild semantic summaries; missing raw artifacts: {missing}"
            )
        low = pd.read_csv(low_path, dtype={"video_id": str})
        low["video_id"] = low.video_id.str.zfill(3)
        semantic = rebuild_semantic_tables(output_dir)
        combined = low.merge(
            semantic, on=["session", "condition", "video_id"], how="left"
        )
        combined.to_csv(output_dir / "stimulus_metrics_combined.csv", index=False)
        statistical_summary(combined, output_dir)
        atomic_write_json(semantic_metadata_path, semantic_metadata)
        save_summary_figure(combined, output_dir / "stimulus_summary.png")
        return combined
    if not semantic_metadata_path.is_file() and any(
        path.exists() for path in cache_artifacts
    ) and not args.force:
        raise RuntimeError(
            "Existing stimulus artifacts have no semantic cache metadata. "
            "Use --force or a new output directory."
        )
    if semantic_metadata_path.is_file() and not args.force:
        observed = json.loads(semantic_metadata_path.read_text(encoding="utf-8"))
        differences = manifest_differences(observed, semantic_metadata)
        if differences:
            preview = "\n".join(f"- {item}" for item in differences)
            raise RuntimeError(
                f"Semantic cache configuration mismatch:\n{preview}\n"
                "Use --force or a new output directory."
            )
    if low_path.exists() and not args.force:
        low = pd.read_csv(low_path, dtype={"video_id": str})
        low["video_id"] = low["video_id"].str.zfill(3)
    else:
        low = compute_low_level(Path(args.stimulus_frames_dir), output_dir)
    combined = low
    if not args.skip_detection:
        semantic_path = output_dir / "video_semantic_metrics.csv"
        if semantic_path.exists() and not args.force:
            semantic = rebuild_semantic_tables(output_dir)
        else:
            semantic = run_semantic_detection(
                Path(args.stimulus_frames_dir),
                output_dir,
                sample_every=args.detection_every,
                score_threshold=args.detection_threshold,
            )
        combined = low.merge(
            semantic, on=["session", "condition", "video_id"], how="left"
        )
    combined.to_csv(output_dir / "stimulus_metrics_combined.csv", index=False)
    statistical_summary(combined, output_dir)
    atomic_write_json(semantic_metadata_path, semantic_metadata)
    save_summary_figure(combined, output_dir / "stimulus_summary.png")
    return combined


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stimulus-frames-dir",
        "--pic-dir",
        dest="stimulus_frames_dir",
        default=str(STIMULUS_FRAMES_DIR),
    )
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "stimulus/characterization")
    )
    p.add_argument("--skip-detection", action="store_true")
    p.add_argument("--detection-every", type=int, default=25)
    p.add_argument("--detection-threshold", type=float, default=0.35)
    p.add_argument(
        "--rebuild-semantic-only",
        action="store_true",
        help="Rebuild canonical semantic summaries from existing raw detections.",
    )
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
