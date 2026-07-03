# -*- coding: utf-8 -*-
import os
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from image_processing import get_hand_roi_params, process_roi_for_model

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# A-Z gesture classes from sign_language_data
SIGN_LABELS = {i: chr(ord("a") + i) for i in range(26)}
NUM_CLASSES = 26
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_language_model.h5"

# Accuracy-focused inference settings
INPUT_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.45
PREDICTION_BUFFER_SIZE = 7
MIN_CONSENSUS_FRAMES = 4
HAND_PADDING = 30
PREDICTION_HOLD_SECONDS = 2.0
PREDICTION_UPDATE_INTERVAL = 0.5


def open_camera():
    """Try common Windows camera backends and return the first working capture."""
    camera_attempts = [
        ("default", lambda: cv2.VideoCapture(0)),
        ("dshow", lambda: cv2.VideoCapture(0, cv2.CAP_DSHOW)),
        ("msmf", lambda: cv2.VideoCapture(0, cv2.CAP_MSMF)),
    ]

    for backend_name, factory in camera_attempts:
        cap = factory()
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok, _ = cap.read()
        if ok:
            print(f"[+] Camera opened with backend: {backend_name}")
            return cap

        cap.release()

    return None


def preprocess_frame(frame):
    """Keep inference preprocessing aligned with image_processing.py."""
    return process_roi_for_model(frame, INPUT_SIZE)


def draw_hands(frame, results):
    """Draw detected hand landmarks."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )
    return frame


def extract_hand_roi(frame, hand_landmarks, padding=HAND_PADDING):
    """Crop a tighter hand box using centralized logic."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = get_hand_roi_params(hand_landmarks, w, h, padding)
    
    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def create_transfer_learning_model():
    """Load the trained model or return None if it is missing."""
    if MODEL_PATH.exists():
        print(f"[+] Loading model from: {MODEL_PATH}")
        return tf.keras.models.load_model(str(MODEL_PATH))

    print(f"[!] Model not found: {MODEL_PATH}")
    print("[*] This feature is disabled until a trained sign_language_model.h5 is added back.")
    print("[*] Add a completed model or retrain before starting webcam recognition.")
    return None


def predict_sign_language(frame, model):
    """Predict a gesture from a cropped hand frame."""
    try:
        processed = preprocess_frame(frame)
        if processed is None:
            return None, 0.0

        # Expand dims for batch size 1
        processed_batch = np.expand_dims(processed, axis=0)
        predictions = model.predict(processed_batch, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        return predicted_class, confidence
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return None, 0.0


def get_stable_prediction(prediction_buffer):
    """Return the dominant class only after enough stable votes."""
    if not prediction_buffer:
        return None, 0.0, 0

    class_votes = [pred[0] for pred in prediction_buffer]
    most_common_class = max(set(class_votes), key=class_votes.count)
    vote_count = class_votes.count(most_common_class)

    if vote_count < MIN_CONSENSUS_FRAMES:
        return None, 0.0, vote_count

    avg_confidence = float(
        np.mean([pred[1] for pred in prediction_buffer if pred[0] == most_common_class])
    )
    return most_common_class, avg_confidence, vote_count


def main():
    """Main program for real-time hand gesture recognition."""
    print("=" * 60)
    print("HAND GESTURE RECOGNITION - TRANSFER LEARNING")
    print("=" * 60)

    print("\n[*] Loading transfer learning model (MobileNetV2)...")
    model = create_transfer_learning_model()
    if model is None:
        print("[-] Recognition cannot start because no completed model is available.")
        return
    print("[+] Model loaded successfully!")
    print(f"[+] Number of gesture classes: {NUM_CLASSES}")
    print(f"[+] Supported labels: {list(SIGN_LABELS.values())}\n")

    cap = open_camera()
    if cap is None:
        print("[-] Cannot open camera!")
        print("[*] Try closing Zoom/Meet/Camera app, then run again.")
        return

    print("[*] Press 'q' to quit, 's' to save the current frame\n")

    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    fps_buffer = deque(maxlen=30)
    frame_count = 0
    prev_time = time.time()
    last_prediction_time = 0.0
    locked_prediction = None
    locked_confidence = 0.0
    locked_until = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Cannot read frame from camera")
                break

            frame_count += 1

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            fps_buffer.append(fps)
            avg_fps = np.mean(fps_buffer)

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            frame = draw_hands(frame, results)

            hand_detected = results.multi_hand_landmarks is not None
            best_prediction = None
            best_bbox = None
            now = time.time()

            should_refresh_prediction = now >= locked_until and (now - last_prediction_time) >= PREDICTION_UPDATE_INTERVAL

            if hand_detected and should_refresh_prediction:
                for hand_landmarks in results.multi_hand_landmarks:
                    roi, bbox = extract_hand_roi(clean_frame, hand_landmarks)
                    if roi is None:
                        continue

                    predicted_class, confidence = predict_sign_language(roi, model)
                    if predicted_class is None:
                        continue

                    if best_prediction is None or confidence > best_prediction[1]:
                        best_prediction = (predicted_class, confidence)
                        best_bbox = bbox

            if best_bbox is not None:
                x1, y1, x2, y2 = best_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Tracked hand",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if best_prediction is not None and best_prediction[1] >= CONFIDENCE_THRESHOLD:
                last_prediction_time = now
                prediction_buffer.append(best_prediction)
                stable_class, stable_confidence, vote_count = get_stable_prediction(prediction_buffer)

                if stable_class is not None:
                    locked_prediction = stable_class
                    locked_confidence = stable_confidence
                    locked_until = now + PREDICTION_HOLD_SECONDS
                    sign_name = SIGN_LABELS.get(locked_prediction, "Unknown").upper()
                    cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"Sign: {sign_name}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Confidence: {locked_confidence:.2%}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Stable frames: {vote_count}/{PREDICTION_BUFFER_SIZE}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        1,
                    )
            elif locked_prediction is not None and now < locked_until:
                sign_name = SIGN_LABELS.get(locked_prediction, "Unknown").upper()
                remaining = max(0.0, locked_until - now)
                cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Sign: {sign_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {locked_confidence:.2%}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Holding result: {remaining:.1f}s",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )
            else:
                if hand_detected:
                    if best_prediction is not None:
                        live_sign = SIGN_LABELS.get(best_prediction[0], "Unknown").upper()
                        cv2.putText(
                            frame,
                            f"Live guess: {live_sign} ({best_prediction[1]:.2%})",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 200, 255),
                            2,
                        )
                    cv2.putText(
                        frame,
                        "Hold steady for a clearer prediction...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                else:
                    prediction_buffer.clear()
                    locked_prediction = None
                    locked_confidence = 0.0
                    locked_until = 0.0
                    cv2.putText(
                        frame,
                        "Waiting for hand...",
                        (w // 2 - 100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

            info_text = f"FPS: {avg_fps:.1f} | Frame: {frame_count}"
            cv2.putText(
                frame,
                info_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Hand Gesture Recognition - Transfer Learning", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[*] Exiting...")
                break
            if key == ord("s"):
                filename = f"gesture_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[+] Frame saved: {filename}")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    except Exception as exc:
        print(f"[-] Error: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera closed")


if __name__ == "__main__":
    main()
