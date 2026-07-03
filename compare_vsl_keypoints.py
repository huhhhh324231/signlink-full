# -*- coding: utf-8 -*-
"""Compare extracted keypoints from a video with a training .npy file.

Usage:
  venv\Scripts\python.exe compare_vsl_keypoints.py 000000.mp4 000000.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vsl_word_video_recognition import extract_feature_timeline_from_video, normalize_sequence


def summarize(name: str, array: np.ndarray) -> None:
    print(f"\n{name}")
    print("shape:", array.shape)
    print("dtype:", array.dtype)
    print("min/max/mean:", float(array.min()), float(array.max()), float(array.mean()))
    print("first frame first 10 landmarks:")
    print(array[0, :10] if array.ndim == 3 else array[0, :30])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VSL video-extracted keypoints to training npy.")
    parser.add_argument("video", help="Input video path, e.g. 000000.mp4")
    parser.add_argument("npy", help="Training keypoint .npy path, e.g. 000000.npy")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    npy_path = Path(args.npy).expanduser().resolve()

    extracted = np.asarray(
        extract_feature_timeline_from_video(video_path, mirror=False, show_preview=False),
        dtype=np.float32,
    )
    training = np.load(npy_path).astype(np.float32)

    summarize("extracted_from_video", extracted)
    summarize("training_npy", training)

    extracted_norm = normalize_sequence(list(extracted))
    training_norm = normalize_sequence(list(training))
    summarize("extracted_normalized", extracted_norm)
    summarize("training_normalized", training_norm)

    diff = extracted_norm - training_norm
    print("\nnormalized absolute diff")
    print("mean:", float(np.mean(np.abs(diff))))
    print("max:", float(np.max(np.abs(diff))))
    print("same shape:", extracted_norm.shape == training_norm.shape)


if __name__ == "__main__":
    main()
