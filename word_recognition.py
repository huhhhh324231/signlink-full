# -*- coding: utf-8 -*-
"""
Real-time word recognition for GRU/LSTM sequence models.

This file is intended for models trained on hand-landmark sequences, not on
single RGB frames. The integrated model `sign_language_gru.h5` expects:

- input shape:  (batch, 30, 126)
- 30 timesteps
- 126 features per frame = 2 hands * 21 landmarks * 3 values (x, y, z)
- output: 21 classes
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_language_gru.h5"
LABELS_PATH = BASE_DIR / "word_labels.json"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126
MAX_HANDS = 2
CONFIDENCE_THRESHOLD = 0.55
PREDICTION_WINDOW = 5
MIN_STABLE_VOTES = 3
DISPLAY_HOLD_SECONDS = 1.5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def default_labels(num_classes: int) -> list[str]:
    return [f"CLASS_{idx:02d}" for idx in range(num_classes)]


def load_labels(num_classes: int) -> list[str]:
    """Load class labels from word_labels.json if available."""
    if LABELS_PATH.exists():
        try:
            data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) == num_classes:
                return [str(label) for label in data]
            print(f"[!] Ignoring {LABELS_PATH.name}: expected {num_classes} labels, got {len(data)}")
        except Exception as exc:
            print(f"[!] Failed to read {LABELS_PATH.name}: {exc}")

    labels = default_labels(num_classes)
    print(f"[!] No valid label map found. Falling back to placeholder labels: {labels}")
    return labels


def load_sequence_model() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Real-time word recognition is disabled until a completed GRU model is added."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape
    output_shape = model.output_shape

    if len(input_shape) != 3 or input_shape[1] != SEQUENCE_LENGTH or input_shape[2] != FEATURE_DIM:
        raise ValueError(
            f"Unexpected model input shape {input_shape}. Expected (None, {SEQUENCE_LENGTH}, {FEATURE_DIM})."
        )

    if len(output_shape) != 2:
        raise ValueError(f"Unexpected model output shape {output_shape}.")

    num_classes = int(output_shape[-1])
    labels = load_labels(num_classes)

    print(f"[+] Loaded model: {MODEL_PATH.name}")
    print(f"[+] Input shape: {input_shape}")
    print(f"[+] Output shape: {output_shape}")
    print(f"[+] Labels: {labels}")
    return model, labels


def extract_hand_features(results) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks into a fixed 126-length feature vector.

    Layout:
    - first hand:  21 * (x, y, z) = 63
    - second hand: 21 * (x, y, z) = 63
    """
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return vector

    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
        offset = hand_idx * 63
        for lm_idx, lm in enumerate(hand_landmarks.landmark):
            base = offset + lm_idx * 3
            vector[base] = lm.x
            vector[base + 1] = lm.y
            vector[base + 2] = lm.z

    return vector


def draw_hands(frame, results) -> None:
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )


def predict_sequence(model: tf.keras.Model, sequence_buffer: deque[np.ndarray], labels: list[str]):
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return None, 0.0

    sequence = np.array(sequence_buffer, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=0)

    preds = model.predict(sequence, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    confidence = float(preds[best_idx])
    label = labels[best_idx]
    return label, confidence


def get_stable_prediction(prediction_buffer: deque[tuple[str, float]]):
    if not prediction_buffer:
        return None, 0.0, 0

    labels = [item[0] for item in prediction_buffer]
    counter = Counter(labels)
    stable_label, votes = counter.most_common(1)[0]

    if votes < MIN_STABLE_VOTES:
        return None, 0.0, votes

    avg_conf = float(
        np.mean([confidence for label, confidence in prediction_buffer if label == stable_label])
    )
    return stable_label, avg_conf, votes


def main():
    print("=" * 60)
    print("REAL-TIME WORD RECOGNITION (GRU SEQUENCE MODEL)")
    print("=" * 60)

    try:
        model, labels = load_sequence_model()
    except Exception as exc:
        print(f"[-] Cannot load sequence model: {exc}")
        print("[*] Add a completed sign_language_gru.h5 to re-enable this feature.")
        print("[*] If you know the real class names, also create word_labels.json as a 21-item JSON list.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Cannot open camera")
        return

    print("[*] Press Q to quit")
    print("[*] Keep one signed word visible for about 1 second to fill the 30-frame buffer")

    feature_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer: deque[tuple[str, float]] = deque(maxlen=PREDICTION_WINDOW)
    fps_buffer: deque[float] = deque(maxlen=30)

    prev_time = time.time()
    locked_label = None
    locked_confidence = 0.0
    locked_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[-] Cannot read frame")
                break

            now = time.time()
            fps = 1 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            fps_buffer.append(fps)
            avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            draw_hands(frame, results)

            feature_vector = extract_hand_features(results)
            feature_buffer.append(feature_vector)

            live_label = None
            live_confidence = 0.0

            if np.any(feature_vector) and len(feature_buffer) == SEQUENCE_LENGTH and now >= locked_until:
                live_label, live_confidence = predict_sequence(model, feature_buffer, labels)
                if live_label is not None and live_confidence >= CONFIDENCE_THRESHOLD:
                    prediction_buffer.append((live_label, live_confidence))
                    stable_label, stable_confidence, vote_count = get_stable_prediction(prediction_buffer)
                    if stable_label is not None:
                        locked_label = stable_label
                        locked_confidence = stable_confidence
                        locked_until = now + DISPLAY_HOLD_SECONDS
                        print(f"[+] Stable word: {stable_label} ({stable_confidence:.1%}, votes={vote_count})")
                elif live_label is not None:
                    prediction_buffer.clear()

            h, w, _ = frame.shape

            cv2.rectangle(frame, (10, 10), (w - 10, 155), (0, 0, 0), -1)
            cv2.putText(frame, "Word Recognition (GRU)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Buffer: {len(feature_buffer)}/{SEQUENCE_LENGTH}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if live_label is not None:
                cv2.putText(
                    frame,
                    f"Live: {live_label} ({live_confidence:.1%})",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Live: waiting for full valid sequence...",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 255),
                    2,
                )

            if locked_label is not None and now < locked_until:
                cv2.putText(
                    frame,
                    f"Stable: {locked_label} ({locked_confidence:.1%})",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Stable: none",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (180, 180, 180),
                    2,
                )

            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Word Recognition (GRU Sequence)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[+] Camera closed")


if __name__ == "__main__":
    main()
