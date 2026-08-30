"""
Accident Detector - Inference / Deployment Script
===================================================
Runs your trained YOLOv8 accident-detection model on a video file or live
webcam/camera feed, draws boxes on detected accidents, and saves clips +
a log whenever an accident is detected with high confidence.

Prerequisites:
    pip install ultralytics opencv-python

Usage:
    python detect_accidents.py --source test_video.mp4 --weights best.pt
    python detect_accidents.py --source 0 --weights best.pt   # webcam
"""

import argparse
import os
import time
from collections import deque

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run accident detection on video/webcam")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to trained model weights (best.pt)")
    parser.add_argument("--source", type=str, default="0",
                         help="Video file path, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Confidence threshold for a detection to count")
    parser.add_argument("--consecutive-frames", type=int, default=5,
                         help="Number of consecutive confident frames needed before alerting")
    parser.add_argument("--output-dir", type=str, default="accident_alerts",
                         help="Folder to save alert clips and logs")
    parser.add_argument("--display", action="store_true",
                         help="Show live video window while processing")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "alerts_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,frame_number,confidence\n")

    # Source can be a webcam index (int) or a video file path (str)
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20  # webcams often report 0/invalid FPS; fall back to a sane default
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Rolling buffer to save a few seconds of video before/after an accident
    buffer_seconds = 3
    frame_buffer = deque(maxlen=int(fps * buffer_seconds))

    frame_number = 0
    consecutive_hits = 0
    cooldown_frames = int(fps * 10)  # don't re-alert for 10s after an alert
    cooldown_counter = 0

    print("Starting detection... press 'q' to quit if --display is on.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        frame_buffer.append(frame.copy())

        # Run detection on this frame
        results = model.predict(frame, conf=args.conf, verbose=False)
        result = results[0]
        annotated_frame = result.plot()  # frame with boxes drawn

        # Check if any detected box is our "accident" class with high confidence
        # NOTE: adjust the string match below to your dataset's actual class name
        # (check data.yaml from your Roboflow dataset for the exact class name)
        accident_detected_this_frame = False
        best_conf = 0.0
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if "accident" in cls_name.lower() and conf >= args.conf:
                accident_detected_this_frame = True
                best_conf = max(best_conf, conf)

        if accident_detected_this_frame:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        if cooldown_counter > 0:
            cooldown_counter -= 1

        # Trigger an alert only after several consecutive confident detections
        # (this filters out single-frame false positives)
        if consecutive_hits >= args.consecutive_frames and cooldown_counter == 0:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            print(f"[ALERT] Accident detected at frame {frame_number}, "
                  f"confidence {best_conf:.2f}, time {timestamp}")

            # Log it
            with open(log_path, "a") as f:
                f.write(f"{timestamp},{frame_number},{best_conf:.3f}\n")

            # Save the buffered clip (before + a bit after)
            save_clip(frame_buffer, args.output_dir, timestamp, fps, width, height)

            cooldown_counter = cooldown_frames
            consecutive_hits = 0

        if args.display:
            cv2.imshow("Accident Detector", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Alerts logged to {log_path}")


def save_clip(frame_buffer, output_dir, timestamp, fps, width, height):
    """Save the buffered frames (last few seconds) as a video clip."""
    clip_path = os.path.join(output_dir, f"accident_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for f in frame_buffer:
        writer.write(f)
    writer.release()
    print(f"  Saved clip: {clip_path}")


if __name__ == "__main__":
    main()