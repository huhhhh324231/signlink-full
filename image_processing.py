import cv2
import numpy as np


def get_hand_roi_params(hand_landmarks, frame_w, frame_h, padding=30):
    """Return a clamped hand ROI box from MediaPipe landmarks."""
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]

    x1 = max(int(min(xs)) - padding, 0)
    y1 = max(int(min(ys)) - padding, 0)
    x2 = min(int(max(xs)) + padding, frame_w)
    y2 = min(int(max(ys)) + padding, frame_h)

    return x1, y1, x2, y2


def process_roi_for_model(roi, target_size=(224, 224)):
    """Resize ROI and convert OpenCV BGR input into RGB model input."""
    if roi is None or roi.size == 0:
        return None

    try:
        resized = cv2.resize(roi, target_size)
        # Training images are decoded as RGB by Keras. Camera frames come from
        # OpenCV in BGR order, so inference must convert here to stay aligned.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return normalized
    except Exception as exc:
        print(f"Error processing ROI: {exc}")
        return None
