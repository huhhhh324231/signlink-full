# -*- coding: utf-8 -*-
"""Debug the VSL Transformer inference pipeline on one video.

Usage:
  venv\Scripts\python.exe debug_vsl_pipeline.py 000000.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vsl_transformer_model import load_vsl_transformer
from vsl_word_video_recognition import (
    HAND_END,
    HAND_START,
    LABELS_PATH,
    POSE_FEATURES,
    STATE_PATH,
    TARGET_SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
    extract_feature_timeline_from_video,
    load_labels,
    normalize_sequence,
    predict_features,
)


def normalize_landmark_variant(raw: np.ndarray, variant: str) -> np.ndarray:
    data = raw.astype(np.float32).copy()

    if variant == "raw":
        return data

    if variant == "zero_z":
        data[:, :, 2] = 0.0
        return data

    if variant == "center_pose0":
        center = data[:, 0:1, :].copy()
        data = data - center
        return data

    if variant == "center_shoulders":
        center = ((data[:, 11:12, :] + data[:, 12:13, :]) / 2.0).copy()
        data = data - center
        return data

    if variant == "center_shoulders_scale":
        center = ((data[:, 11:12, :] + data[:, 12:13, :]) / 2.0).copy()
        scale = np.linalg.norm(data[:, 11, :2] - data[:, 12, :2], axis=1)
        scale = np.where(scale > 1e-6, scale, 1.0).reshape(-1, 1, 1)
        data = (data - center) / scale
        return data

    if variant == "center_hips_scale":
        center = ((data[:, 23:24, :] + data[:, 24:25, :]) / 2.0).copy()
        scale = np.linalg.norm(data[:, 11, :2] - data[:, 12, :2], axis=1)
        scale = np.where(scale > 1e-6, scale, 1.0).reshape(-1, 1, 1)
        data = (data - center) / scale
        return data

    raise ValueError(f"Unknown variant: {variant}")


def flatten_sequence_variant(raw: np.ndarray, variant: str) -> np.ndarray:
    if variant == "landmark_xyz":
        return normalize_sequence(list(raw))
    if variant == "xyz_landmark":
        sequence = raw.astype(np.float32)
        if len(sequence) > TARGET_SEQUENCE_LENGTH:
            indices = np.linspace(0, len(sequence) - 1, TARGET_SEQUENCE_LENGTH).astype(np.int32)
            sequence = sequence[indices]
        elif len(sequence) < TARGET_SEQUENCE_LENGTH:
            pad = np.repeat(sequence[-1:], TARGET_SEQUENCE_LENGTH - len(sequence), axis=0)
            sequence = np.concatenate([sequence, pad], axis=0)
        return np.transpose(sequence, (0, 2, 1)).reshape(TARGET_SEQUENCE_LENGTH, -1)
    if variant == "pose_hands_xy_only_z_last":
        sequence = raw.astype(np.float32)
        if len(sequence) > TARGET_SEQUENCE_LENGTH:
            indices = np.linspace(0, len(sequence) - 1, TARGET_SEQUENCE_LENGTH).astype(np.int32)
            sequence = sequence[indices]
        elif len(sequence) < TARGET_SEQUENCE_LENGTH:
            pad = np.repeat(sequence[-1:], TARGET_SEQUENCE_LENGTH - len(sequence), axis=0)
            sequence = np.concatenate([sequence, pad], axis=0)
        xy = sequence[:, :, :2].reshape(TARGET_SEQUENCE_LENGTH, -1)
        z = sequence[:, :, 2:3].reshape(TARGET_SEQUENCE_LENGTH, -1)
        return np.concatenate([xy, z], axis=1)
    raise ValueError(f"Unknown flatten variant: {variant}")


def landmark_order_variant(raw: np.ndarray, variant: str) -> np.ndarray:
    if variant == "pose_left_right_extra":
        return raw
    pose = raw[:, :33]
    left = raw[:, 33:54]
    right = raw[:, 54:75]
    extra = raw[:, 75:76]
    if variant == "pose_right_left_extra":
        return np.concatenate([pose, right, left, extra], axis=1)
    if variant == "left_right_pose_extra":
        return np.concatenate([left, right, pose, extra], axis=1)
    if variant == "right_left_pose_extra":
        return np.concatenate([right, left, pose, extra], axis=1)
    raise ValueError(f"Unknown landmark order variant: {variant}")


def print_feature_stats(features: list[np.ndarray]) -> None:
    print("raw_frame_count:", len(features))
    if not features:
        return

    raw = np.asarray(features, dtype=np.float32)
    flat = raw.reshape(raw.shape[0], -1)
    pose = flat[:, :POSE_FEATURES]
    hands = flat[:, HAND_START:HAND_END]

    print("features[0].shape:", features[0].shape)
    print("features array shape:", raw.shape)
    print("expected raw shape:", f"(T, {TOTAL_LANDMARKS}, 3)")
    print("normalized shape:", normalize_sequence(features).shape)
    print("target length:", TARGET_SEQUENCE_LENGTH)
    print("pose_nonzero_frames:", int(np.count_nonzero(np.any(pose, axis=1))))
    print("hand_nonzero_frames:", int(np.count_nonzero(np.any(hands, axis=1))))
    print("first 10 landmarks of frame 0:")
    print(features[0][:10])
    print("feature min/max/mean:", float(raw.min()), float(raw.max()), float(raw.mean()))


def run_predictions(video_path: Path, mirror: bool) -> None:
    print("\n" + "=" * 72)
    print("mirror:", mirror)
    features = extract_feature_timeline_from_video(video_path, mirror=mirror, show_preview=False)
    print_feature_stats(features)

    labels = load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vsl_transformer(STATE_PATH, num_classes=len(labels), device=device)

    print("labels file:", LABELS_PATH)
    print("label_count:", len(labels))
    raw = np.asarray(features, dtype=np.float32)
    target_label = "Anh"
    target_idx = labels.index(target_label) if target_label in labels else None
    variants = [
        "raw",
        "zero_z",
        "center_pose0",
        "center_shoulders",
        "center_shoulders_scale",
        "center_hips_scale",
    ]
    flatten_variants = ["landmark_xyz", "xyz_landmark", "pose_hands_xy_only_z_last"]
    order_variants = [
        "pose_left_right_extra",
        "pose_right_left_extra",
        "left_right_pose_extra",
        "right_left_pose_extra",
    ]

    for variant in variants:
        variant_raw = normalize_landmark_variant(raw, variant)

        print("\nvariant:", variant)
        for order_name in order_variants:
            ordered_raw = landmark_order_variant(variant_raw, order_name)
            for flatten_name in flatten_variants:
                sequence = flatten_sequence_variant(ordered_raw, flatten_name)
                with torch.no_grad():
                    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                    embedded = model.embedding(tensor)
                    encoded = model.transformer(model.position(embedded))
                    pooled = encoded.mean(dim=1)
                    logits = model.classifier(pooled)
                    probs = torch.softmax(logits, dim=1)[0]
                    values, indices = torch.topk(probs, k=min(5, len(labels)))
                    order = torch.argsort(probs, descending=True)

                print("order:", order_name, "flatten:", flatten_name)
                if target_idx is not None:
                    target_rank = int((order == target_idx).nonzero(as_tuple=True)[0].item() + 1)
                    print(f"  {target_label} rank:", target_rank, "prob:", f"{float(probs[target_idx]):.4%}")
                for rank, (value, index) in enumerate(zip(values.cpu(), indices.cpu()), start=1):
                    print(f"  {rank}. {labels[int(index)]}: {float(value):.2%}")

                if target_idx is not None and target_rank <= 3:
                    print("  *** candidate:", variant, order_name, flatten_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug VSL Holistic inference.")
    parser.add_argument("video", nargs="?", default="000000.mp4", help="Video path to inspect.")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("video:", video_path)
    run_predictions(video_path, mirror=False)
    run_predictions(video_path, mirror=True)


if __name__ == "__main__":
    main()
