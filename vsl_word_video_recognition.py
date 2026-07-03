# -*- coding: utf-8 -*-
"""Recognize Vietnamese sign-language words from recorded or uploaded videos.

Usage:
  python vsl_word_video_recognition.py --camera
  python vsl_word_video_recognition.py --video path/to/video.mp4
  python vsl_word_video_recognition.py --upload
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import mediapipe as mp
import numpy as np
import torch

from vsl_transformer_model import FEATURE_DIM, load_vsl_transformer


BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "vsl_transformer_state.pth"
LABELS_PATH = BASE_DIR / "vsl_word_labels.json"
RECORDINGS_DIR = BASE_DIR / "recordings"

TOP_K = 5
TARGET_SEQUENCE_LENGTH = 64
POSE_LANDMARKS = 33
BODY_LANDMARKS = 34
HAND_LANDMARKS = 21
TOTAL_LANDMARKS = BODY_LANDMARKS + HAND_LANDMARKS + HAND_LANDMARKS
POSE_FEATURES = BODY_LANDMARKS * 3
HAND_FEATURES = HAND_LANDMARKS * 3 * 2
HAND_START = POSE_FEATURES
HAND_END = POSE_FEATURES + HAND_FEATURES
DEFAULT_PAUSE_FRAMES = 12
DEFAULT_MIN_SEGMENT_FRAMES = 8
DEFAULT_MOTION_THRESHOLD = 0.006
DEFAULT_PRE_ROLL_FRAMES = 3
DEFAULT_POST_ROLL_FRAMES = 3

BODY_NAMES = [
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner", "rightEye", "rightEyeOuter",
    "leftEar", "rightEar", "mouthLeft", "mouthRight", "leftShoulder", "rightShoulder",
    "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftPinky", "rightPinky",
    "leftIndex", "rightIndex", "leftThumb", "rightThumb",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
    "leftHeel", "rightHeel", "leftFootIndex", "rightFootIndex", "neck",
]
HAND_NAMES = [
    "wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP",
    "middleTip", "middleDIP", "middlePIP", "middleMCP",
    "ringTip", "ringDIP", "ringPIP", "ringMCP",
    "littleTip", "littleDIP", "littlePIP", "littleMCP",
    "thumbTip", "thumbIP", "thumbMP", "thumbCMC",
]
POSE_MAP = {
    "nose": 0, "leftEyeInner": 1, "leftEye": 2, "leftEyeOuter": 3, "rightEyeInner": 4,
    "rightEye": 5, "rightEyeOuter": 6, "leftEar": 7, "rightEar": 8, "mouthLeft": 9,
    "mouthRight": 10, "leftShoulder": 11, "rightShoulder": 12, "leftElbow": 13,
    "rightElbow": 14, "leftWrist": 15, "rightWrist": 16, "leftPinky": 17,
    "rightPinky": 18, "leftIndex": 19, "rightIndex": 20, "leftThumb": 21,
    "rightThumb": 22, "leftHip": 23, "rightHip": 24, "leftKnee": 25,
    "rightKnee": 26, "leftAnkle": 27, "rightAnkle": 28, "leftHeel": 29,
    "rightHeel": 30, "leftFootIndex": 31, "rightFootIndex": 32,
}
HAND_MAP = {
    "wrist": 0, "thumbCMC": 1, "thumbMP": 2, "thumbIP": 3, "thumbTip": 4,
    "indexMCP": 5, "indexPIP": 6, "indexDIP": 7, "indexTip": 8,
    "middleMCP": 9, "middlePIP": 10, "middleDIP": 11, "middleTip": 12,
    "ringMCP": 13, "ringPIP": 14, "ringDIP": 15, "ringTip": 16,
    "littleMCP": 17, "littlePIP": 18, "littleDIP": 19, "littleTip": 20,
}
BODY_ANCHORS = ["nose", "leftShoulder", "rightShoulder", "leftHip", "rightHip", "neck"]
ARM_BODY_INDICES = [
    BODY_NAMES.index("leftShoulder"),
    BODY_NAMES.index("rightShoulder"),
    BODY_NAMES.index("leftElbow"),
    BODY_NAMES.index("rightElbow"),
    BODY_NAMES.index("leftWrist"),
    BODY_NAMES.index("rightWrist"),
    BODY_NAMES.index("leftIndex"),
    BODY_NAMES.index("rightIndex"),
    BODY_NAMES.index("leftThumb"),
    BODY_NAMES.index("rightThumb"),
]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid labels file: {LABELS_PATH}")
    return [str(item) for item in data]


def extract_landmark_features(results) -> np.ndarray:
    """Convert Holistic landmarks into the training-time `(76, 3)` frame.

    Training layout:
    - 34 body landmarks: MediaPipe pose 33 plus an added neck point
    - 21 left-hand landmarks in the custom dataset order
    - 21 right-hand landmarks in the custom dataset order

    Body and each hand are normalized independently, matching the Kaggle
    extraction notebook used to create the `.npy` files. The two hands are
    stored interleaved by landmark name: wrist_0, wrist_1, indexTip_0,
    indexTip_1, ...
    """
    pose_data = {name: (0.0, 0.0, 0.0) for name in BODY_NAMES}
    if results.pose_landmarks:
        for name, index in POSE_MAP.items():
            lm = results.pose_landmarks.landmark[index]
            pose_data[name] = (lm.x, lm.y, lm.z)
        left_shoulder = pose_data["leftShoulder"]
        right_shoulder = pose_data["rightShoulder"]
        pose_data["neck"] = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_shoulder[2] + right_shoulder[2]) / 2,
        )

    body_points = normalize_body_points(pose_data)
    left_hand_points = extract_hand_points(results.left_hand_landmarks)
    right_hand_points = extract_hand_points(results.right_hand_landmarks)
    hand_points = []
    for left_point, right_point in zip(left_hand_points, right_hand_points):
        hand_points.append(left_point)
        hand_points.append(right_point)

    return np.asarray(body_points + hand_points, dtype=np.float32)


def normalize_body_points(pose_data: dict[str, tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    normalized = dict(pose_data)
    x_coords = [pose_data[name][0] for name in BODY_ANCHORS if pose_data[name][0] != 0]
    y_coords = [pose_data[name][1] for name in BODY_ANCHORS if pose_data[name][1] != 0]
    if x_coords and y_coords:
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        dx = (max_x - min_x) * 1.6
        dy = (max_y - min_y) * 1.6
        if dx > 0 and dy > 0:
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            box_min_x = center_x - dx / 2
            box_min_y = center_y - dy / 2
            for key in BODY_NAMES:
                x, y, z = normalized[key]
                if x == 0 and y == 0:
                    continue
                normalized[key] = ((x - box_min_x) / dx - 0.5, (y - box_min_y) / dy - 0.5, z)
    return [normalized[name] for name in BODY_NAMES]


def extract_hand_points(hand_landmarks) -> list[tuple[float, float, float]]:
    hand_data = {name: (0.0, 0.0, 0.0) for name in HAND_NAMES}
    if hand_landmarks:
        for name, index in HAND_MAP.items():
            lm = hand_landmarks.landmark[index]
            hand_data[name] = (lm.x, lm.y, lm.z)
    return normalize_hand_points(hand_data)


def normalize_hand_points(hand_data: dict[str, tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    normalized = dict(hand_data)
    x_coords = [hand_data[name][0] for name in HAND_NAMES if hand_data[name][0] != 0]
    y_coords = [hand_data[name][1] for name in HAND_NAMES if hand_data[name][1] != 0]
    if x_coords and y_coords:
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        dx = max_x - min_x
        dy = max_y - min_y
        if dx > 0 and dy > 0:
            for name in HAND_NAMES:
                x, y, z = normalized[name]
                if x == 0 and y == 0:
                    continue
                normalized[name] = ((x - min_x) / dx - 0.5, (y - min_y) / dy - 0.5, z)
    return [normalized[name] for name in HAND_NAMES]


def draw_hands(frame, results) -> None:
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 180, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 180, 255), thickness=1),
        )
    for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        if not hand_landmarks:
            continue
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )


def normalize_sequence(features: list[np.ndarray]) -> np.ndarray:
    """Downsample/pad and flatten to the transformer's 64x228 input.

    `extract_feature_timeline_from_video` returns raw `(76, 3)` frames so the
    intermediate shape remains inspectable. Flattening happens only here, just
    before the tensor is passed into the Transformer.
    """
    if not features:
        return np.zeros((TARGET_SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    sequence = np.asarray(features, dtype=np.float32)
    if len(sequence) > TARGET_SEQUENCE_LENGTH:
        indices = np.linspace(0, len(sequence) - 1, TARGET_SEQUENCE_LENGTH).astype(np.int32)
        sequence = sequence[indices]
    elif len(sequence) < TARGET_SEQUENCE_LENGTH:
        pad = np.repeat(sequence[-1:], TARGET_SEQUENCE_LENGTH - len(sequence), axis=0)
        sequence = np.concatenate([sequence, pad], axis=0)

    return sequence.reshape(TARGET_SEQUENCE_LENGTH, FEATURE_DIM)


def flatten_frame(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def predict_features(model, labels: list[str], features: list[np.ndarray], device: torch.device) -> list[tuple[str, float]]:
    sequence = normalize_sequence(features)
    with torch.no_grad():
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(TOP_K, len(labels)))

    return [(labels[int(idx)], float(value)) for value, idx in zip(values.cpu(), indices.cpu())]


def extract_feature_timeline_from_video(video_path: Path, mirror: bool = False, show_preview: bool = True) -> list[np.ndarray]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    features: list[np.ndarray] = []
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
    )

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            vector = extract_landmark_features(results)
            features.append(vector)

            if show_preview:
                draw_hands(frame, results)
                cv2.putText(
                    frame,
                    f"Reading video... valid frames: {len(features)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("VSL video recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        holistic.close()
        if show_preview:
            cv2.destroyWindow("VSL video recognition")

    return features


def extract_features_from_video(video_path: Path, mirror: bool = False, show_preview: bool = True) -> list[np.ndarray]:
    return extract_feature_timeline_from_video(video_path, mirror=mirror, show_preview=show_preview)


def hand_motion(prev_vector: np.ndarray | None, vector: np.ndarray) -> float:
    flat_vector = flatten_frame(vector)
    hand_vector = flat_vector[HAND_START:HAND_END]
    if not np.any(hand_vector):
        return 0.0
    if prev_vector is None:
        return 1.0
    prev_flat_vector = flatten_frame(prev_vector)
    if not np.any(prev_flat_vector[HAND_START:HAND_END]):
        return 1.0
    return float(np.mean(np.abs(hand_vector - prev_flat_vector[HAND_START:HAND_END])))


def has_hand_landmarks(vector: np.ndarray) -> bool:
    flat_vector = flatten_frame(vector)
    return bool(np.any(flat_vector[HAND_START:HAND_END]))


def motion_signal(prev_vector: np.ndarray | None, vector: np.ndarray) -> float:
    """Estimate signing motion from arms plus hand shapes.

    Hand coordinates are normalized per hand to match training, so translation
    of the whole hand is weak there. Body arm landmarks keep wrist/elbow motion,
    while hand landmarks keep finger-shape motion. Combining both gives a more
    useful pause detector for sentence-level demos.
    """
    if prev_vector is None:
        return 0.0

    frame = np.asarray(vector, dtype=np.float32)
    prev_frame = np.asarray(prev_vector, dtype=np.float32)
    arm_delta = np.mean(np.abs(frame[ARM_BODY_INDICES] - prev_frame[ARM_BODY_INDICES]))
    hand_delta = np.mean(np.abs(frame[BODY_LANDMARKS:] - prev_frame[BODY_LANDMARKS:]))
    return float((arm_delta * 0.65) + (hand_delta * 0.35))


def smooth_values(values: list[float], radius: int = 2) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return smoothed


def adaptive_motion_threshold(motions: list[float], fallback: float) -> float:
    positive = np.asarray([value for value in motions if value > 0], dtype=np.float32)
    if positive.size < 4:
        return fallback
    percentile_value = float(np.percentile(positive, 35))
    return max(fallback, percentile_value * 0.65)


def segment_sentence_timeline(
    timeline: list[np.ndarray],
    pause_frames: int = DEFAULT_PAUSE_FRAMES,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    pre_roll_frames: int = DEFAULT_PRE_ROLL_FRAMES,
    post_roll_frames: int = DEFAULT_POST_ROLL_FRAMES,
) -> list[dict[str, object]]:
    """Split a full video timeline into word-like segments.

    The method is deliberately simple for local demos: the signer leaves a
    short rest between words, then this detects sustained low motion/no hand
    visibility as a boundary.
    """
    if not timeline:
        return []

    motions: list[float] = []
    prev_vector: np.ndarray | None = None
    for vector in timeline:
        motions.append(motion_signal(prev_vector, vector))
        prev_vector = vector

    smoothed = smooth_values(motions)
    threshold = adaptive_motion_threshold(smoothed, motion_threshold)
    hand_flags = [has_hand_landmarks(vector) for vector in timeline]
    active_flags = [has_hand and motion >= threshold for has_hand, motion in zip(hand_flags, smoothed)]

    segments: list[dict[str, object]] = []
    current_start: int | None = None
    last_active = -1
    pause_count = 0

    for index, is_active in enumerate(active_flags):
        has_hand = hand_flags[index]

        if is_active and current_start is None:
            current_start = max(0, index - pre_roll_frames)

        if current_start is None:
            continue

        if is_active:
            last_active = index
            pause_count = 0
            continue

        pause_count = pause_count + 1 if not has_hand or smoothed[index] < threshold else 0
        if pause_count >= pause_frames and last_active >= current_start:
            end = min(len(timeline), last_active + post_roll_frames + 1)
            if end - current_start >= min_segment_frames:
                segments.append(
                    {
                        "startFrame": current_start,
                        "endFrame": end - 1,
                        "features": timeline[current_start:end],
                        "threshold": threshold,
                    }
                )
            current_start = None
            last_active = -1
            pause_count = 0

    if current_start is not None:
        end = len(timeline)
        if last_active >= current_start:
            end = min(len(timeline), last_active + post_roll_frames + 1)
        if end - current_start >= min_segment_frames:
            segments.append(
                {
                    "startFrame": current_start,
                    "endFrame": end - 1,
                    "features": timeline[current_start:end],
                    "threshold": threshold,
                }
            )

    if not segments and sum(hand_flags) >= min_segment_frames:
        first = next(index for index, flag in enumerate(hand_flags) if flag)
        last = len(hand_flags) - 1 - next(index for index, flag in enumerate(reversed(hand_flags)) if flag)
        if last - first + 1 >= min_segment_frames:
            segments.append(
                {
                    "startFrame": first,
                    "endFrame": last,
                    "features": timeline[first : last + 1],
                    "threshold": threshold,
                }
            )

    return segments


def segment_sentence_features(
    timeline: list[np.ndarray],
    pause_frames: int = DEFAULT_PAUSE_FRAMES,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
) -> list[list[np.ndarray]]:
    return [
        segment["features"]
        for segment in segment_sentence_timeline(
            timeline,
            pause_frames=pause_frames,
            min_segment_frames=min_segment_frames,
            motion_threshold=motion_threshold,
        )
    ]


def predict_sentence_features(
    model,
    labels: list[str],
    timeline: list[np.ndarray],
    device: torch.device,
    pause_frames: int = DEFAULT_PAUSE_FRAMES,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
) -> tuple[str, list[dict[str, object]]]:
    segments = segment_sentence_timeline(
        timeline,
        pause_frames=pause_frames,
        min_segment_frames=min_segment_frames,
        motion_threshold=motion_threshold,
    )
    words: list[dict[str, object]] = []
    for index, segment_info in enumerate(segments, start=1):
        segment = segment_info["features"]
        predictions = predict_features(model, labels, segment, device)
        if not predictions:
            continue
        label, confidence = predictions[0]
        words.append(
            {
                "index": index,
                "label": label,
                "confidence": confidence,
                "frames": len(segment),
                "startFrame": segment_info["startFrame"],
                "endFrame": segment_info["endFrame"],
                "segmentThreshold": segment_info["threshold"],
                "predictions": [
                    {"label": item_label, "confidence": item_confidence}
                    for item_label, item_confidence in predictions
                ],
            }
        )

    sentence = " ".join(str(item["label"]) for item in words)
    return sentence, words


def record_from_camera(camera_id: int = 0) -> tuple[Path, list[np.ndarray]]:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    output_path = RECORDINGS_DIR / f"vsl_recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(10.0, min(fps, 30.0)),
        (width, height),
    )

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
    )

    features: list[np.ndarray] = []
    is_recording = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            vector = extract_landmark_features(results)

            if is_recording:
                writer.write(frame)
                features.append(vector)

            draw_hands(frame, results)
            status = "RECORDING" if is_recording else "READY"
            color = (0, 0, 255) if is_recording else (0, 255, 255)
            cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(
                frame,
                "R: start/stop recording | Q: finish",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Valid sequence frames: {len(features)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("VSL record word", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                is_recording = not is_recording
                print(f"[*] Recording {'started' if is_recording else 'stopped'}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        holistic.close()
        cv2.destroyWindow("VSL record word")

    return output_path, features


def print_predictions(title: str, predictions: list[tuple[str, float]], valid_frames: int) -> None:
    print("\n" + "=" * 60)
    print(title)
    print(f"Valid landmark frames: {valid_frames}")
    if not predictions:
        print("No prediction available.")
    else:
        print(f"Best word: {predictions[0][0]} ({predictions[0][1]:.2%})")
        print("\nTop predictions:")
        for rank, (label, confidence) in enumerate(predictions, start=1):
            print(f"{rank}. {label}: {confidence:.2%}")
    print("=" * 60)


def choose_video_file() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Chọn video ký hiệu để nhận diện",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese sign-language word recognition from video.")
    parser.add_argument("--camera", action="store_true", help="Record a clip from webcam, then recognize the word.")
    parser.add_argument("--video", type=str, help="Recognize a word from an existing video file.")
    parser.add_argument("--upload", action="store_true", help="Open a file picker and recognize the selected video.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--mirror-video", action="store_true", help="Mirror uploaded video before feature extraction.")
    parser.add_argument("--no-preview", action="store_true", help="Do not show OpenCV preview while reading video.")
    args = parser.parse_args()

    if not args.camera and not args.video and not args.upload:
        parser.error("Choose one mode: --camera, --video path/to/video.mp4, or --upload")

    labels = load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vsl_transformer(STATE_PATH, num_classes=len(labels), device=device)

    if args.camera:
        saved_path, features = record_from_camera(args.camera_id)
        predictions = predict_features(model, labels, features, device)
        print_predictions(f"Recorded video: {saved_path}", predictions, len(features))

    selected_video = Path(args.video).expanduser().resolve() if args.video else None
    if args.upload:
        selected_video = choose_video_file()
        if selected_video is None:
            print("[!] No video selected.")
            return

    if selected_video:
        features = extract_features_from_video(
            selected_video,
            mirror=args.mirror_video,
            show_preview=not args.no_preview,
        )
        predictions = predict_features(model, labels, features, device)
        print_predictions(f"Uploaded video: {selected_video}", predictions, len(features))


if __name__ == "__main__":
    main()
