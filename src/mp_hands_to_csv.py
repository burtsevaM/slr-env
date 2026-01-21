# src/mp_hands_to_csv.py
"""
MediaPipe Hands -> CSV (1 hand, 21 keypoints)

- Opens webcam
- Detects one hand (max_num_hands=1)
- Saves 21 landmarks (x,y,z) per frame into a CSV

Controls:
  SPACE  - start/stop recording
  q      - quit
  f      - toggle flip (selfie)
  d      - toggle drawing

Run:
  python src/mp_hands_to_csv.py --out data/hand_001.csv
  python src/mp_hands_to_csv.py --camera 1 --out data/hand_002.csv
"""
"""
Проверить размерность через pandas

python - <<'PY'
import pandas as pd
df = pd.read_csv("data/hand_001.csv")
print("shape:", df.shape)
print("columns:", len(df.columns))
print(df.head(2).T.head(20))
PY

"""

"""
Проверка, что CSV реально записался

head -n 2 data/hand_001.csv
wc -l data/hand_001.csv
"""


import argparse
import csv
import os
import time

import cv2
import mediapipe as mp


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                 # palm base
]


def draw_hand(frame_bgr, landmarks_norm, color=(0, 255, 0)):
    h, w = frame_bgr.shape[:2]
    pts = []
    for lm in landmarks_norm:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        pts.append((x_px, y_px))

    # lines
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color, 2)

    # points
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to output CSV, e.g. data/hand_001.csv")
    parser.add_argument("--model", default="models/hand_landmarker.task", help="Path to .task model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (usually 0)")
    parser.add_argument("--max-frames", type=int, default=500, help="How many frames to record (with hand)")
    parser.add_argument("--min-det", type=float, default=0.5, help="min_hand_detection_confidence")
    parser.add_argument("--min-pres", type=float, default=0.5, help="min_hand_presence_confidence")
    parser.add_argument("--min-track", type=float, default=0.5, help="min_tracking_confidence")
    parser.add_argument("--hand", choices=["Any", "Left", "Right"], default="Any", help="Filter by handedness")
    parser.add_argument("--no-preview", action="store_true", help="Do not show OpenCV window")
    parser.add_argument("--flip", action="store_true", help="Flip image horizontally (mirror)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            f"Download it into models/hand_landmarker.task (см. шаг 2)."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # MediaPipe Tasks objects (официальный Python API)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(args.model)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_det,
        min_hand_presence_confidence=args.min_pres,
        min_tracking_confidence=args.min_track,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "Не удалось открыть камеру.\n"
            "Проверь: 1) камера не занята Zoom/Meet, 2) разрешения на камеру для VS Code/Terminal."
        )

    header = ["timestamp_ms", "handedness"] + [
        f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")
    ]

    frames_written = 0
    start_ms = int(time.time() * 1000)

    with open(args.out, "w", newline="") as f, HandLandmarker.create_from_options(options) as landmarker:
        writer = csv.writer(f)
        writer.writerow(header)

        while frames_written < args.max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                print("⚠️ frame read failed, stopping")
                break

            if args.flip:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # timestamp MUST be monotonically increasing in VIDEO mode
            ts_ms = int(time.time() * 1000) - start_ms

            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                # handedness
                handed = "Unknown"
                try:
                    handed = result.handedness[0][0].category_name  # обычно "Left"/"Right"
                except Exception:
                    pass

                if args.hand != "Any" and handed != args.hand:
                    # если хотим только Left/Right — пропускаем
                    pass
                else:
                    lms = result.hand_landmarks[0]  # 21 landmark
                    row = [ts_ms, handed]
                    for lm in lms:
                        row.extend([lm.x, lm.y, lm.z])
                    writer.writerow(row)
                    frames_written += 1

                    if not args.no_preview:
                        draw_hand(frame_bgr, lms)
                        cv2.putText(
                            frame_bgr,
                            f"written: {frames_written}/{args.max_frames} hand: {handed}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

            if not args.no_preview:
                cv2.imshow("MediaPipe HandLandmarker -> CSV (press q to stop)", frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved: {args.out} (rows with hand = {frames_written})")


if __name__ == "__main__":
    main()
