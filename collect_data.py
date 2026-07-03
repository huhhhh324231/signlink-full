import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from image_processing import get_hand_roi_params

SIGN_LABELS = {i: chr(ord("a") + i) for i in range(26)}
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sign_language_data"
SAMPLES_PER_SIGN = 120
IMG_SIZE = (224, 224)
HAND_PADDING = 30
AUTO_CAPTURE_EVERY = 0.35

SESSION_STEPS = [
    "center / normal distance",
    "slightly closer to camera",
    "slightly farther from camera",
    "tilt hand a little left/right",
    "move to a different background spot",
    "change brightness a little if possible",
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_drawing = mp.solutions.drawing_utils


def create_directories():
    """Create one folder per letter."""
    for label in SIGN_LABELS.values():
        (DATA_DIR / label).mkdir(parents=True, exist_ok=True)
    print(f"[+] Data folders created in: {DATA_DIR}")


def next_image_index(save_dir):
    """Continue numbering without overwriting previous sessions."""
    existing = []
    for filename in os.listdir(save_dir):
        stem = Path(filename).stem
        try:
            existing.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return max(existing, default=0) + 1


def extract_hand_roi(frame, hand_landmarks, padding=HAND_PADDING):
    """Crop a tight hand region using centralized logic."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = get_hand_roi_params(hand_landmarks, w, h, padding)
    
    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_preview(frame, sign_name, count, target_count, mode_text, tip_text):
    """Overlay capture instructions."""
    h, _, _ = frame.shape
    cv2.putText(frame, f"Letter: {sign_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Saved: {count}/{target_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Mode: {mode_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame, f"Tip: {tip_text}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, "A: auto capture | SPACE: single capture | N: next tip | ESC: finish", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)


def collect_data():
    """Collect letter gesture samples from webcam with more diversity."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session_tag = input("Session tag (example: room1_daylight_user1): ").strip().replace(" ", "_")
    if not session_tag:
        session_tag = time.strftime("session_%Y%m%d_%H%M%S")

    print("[+] Camera opened")
    print("\n" + "=" * 60)
    print("HAND GESTURE DATA COLLECTION")
    print("=" * 60)
    print("[!] Goal: collect many natural variations, not one perfect pose only.")

    for _, sign_name in sorted(SIGN_LABELS.items()):
        save_dir = DATA_DIR / sign_name
        image_index = next_image_index(save_dir)
        target_count = SAMPLES_PER_SIGN
        tip_index = 0
        auto_capture = False
        last_capture_time = 0.0

        print(f"\n[*] Prepare gesture: {sign_name.upper()}")
        print(f"[*] Save folder: {save_dir}")
        print(f"[*] Target samples: {target_count}")
        print("[!] Press SPACE to start, ESC to skip this letter")

        start_key = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Cannot read frame!")
                cap.release()
                cv2.destroyAllWindows()
                return

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )
                cv2.putText(frame, "Hand detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Letter: {sign_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "SPACE: start | ESC: skip", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Hand Gesture Data Collection", frame)

            start_key = cv2.waitKey(1) & 0xFF
            if start_key in (32, 27):
                break

        if start_key == 27:
            print(f"[!] Skipped {sign_name.upper()}")
            continue

        count = 0
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                print("[-] Frame read error!")
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            results = hands.process(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))

            roi = None
            bbox = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                roi, bbox = extract_hand_roi(clean_frame, hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Tracked hand", (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            mode_text = "auto" if auto_capture else "manual"
            tip_text = SESSION_STEPS[tip_index % len(SESSION_STEPS)]
            draw_preview(frame, sign_name, count, target_count, mode_text, tip_text)
            cv2.imshow("Hand Gesture Data Collection", frame)

            current_time = time.time()
            should_auto_capture = (
                auto_capture
                and roi is not None
                and (current_time - last_capture_time) >= AUTO_CAPTURE_EVERY
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                auto_capture = not auto_capture
                print(f"[*] Auto capture {'enabled' if auto_capture else 'disabled'} for {sign_name.upper()}")
            elif key == ord("n"):
                tip_index += 1
                print(f"[*] Next capture tip for {sign_name.upper()}: {SESSION_STEPS[tip_index % len(SESSION_STEPS)]}")
            elif key == 27:
                print(f"[!] Stopped early, collected {count} images")
                break

            manual_capture = key == 32 and roi is not None

            if manual_capture or should_auto_capture:
                resized = cv2.resize(roi, IMG_SIZE)
                filename = f"{sign_name}_{session_tag}_{image_index:04d}.jpg"
                img_path = save_dir / filename
                cv2.imwrite(str(img_path), resized)
                count += 1
                image_index += 1
                last_capture_time = current_time
                print(f"  [{count}/{target_count}] Saved: {img_path}")
            elif key == 32 and roi is None:
                print("  [!] No hand detected, try again")

        print(f"[+] Done: {sign_name.upper()} ({count} images)")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[+] Data collection finished!")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA COLLECTION TOOL")
    print("=" * 60)
    create_directories()
    print("\n[!] Tips:")
    print("  - Keep the hand large and clear in frame")
    print("  - Collect the same letter under several lighting/background conditions")
    print("  - Use auto capture, then change angle/pose slightly every few seconds")
    print("  - Recollect letters that are often confused")
    input("\nPress Enter to start...")
    collect_data()
