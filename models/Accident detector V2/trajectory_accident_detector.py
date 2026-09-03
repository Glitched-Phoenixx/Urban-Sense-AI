"""
Accident Detector - Trajectory-Based Motion Analysis
======================================================
Tracks vehicles across frames using YOLOv8's built-in ByteTrack tracker,
then analyzes each vehicle's motion pattern over time to detect accidents:
    - Sudden direction change (smoothed, to avoid detector-noise false alarms)
    - Sudden deceleration
    - Bounding box overlap (IoU) spikes between two vehicles
    - Sudden bounding box size/shape change (spin/flip/crumple proxy)

These signals are combined into a single anomaly score per vehicle per frame.
Optionally, your trained Accident/Non-Accident classifier can be used as a
confirmation step to reduce false positives further.

Prerequisites:
    pip install ultralytics opencv-python numpy

Usage:
    python trajectory_accident_detector.py --source test_video.mp4 --detector-weights yolov8n.pt
    python trajectory_accident_detector.py --source 0 --detector-weights yolov8n.pt --display

    # With your trained Accident/Non-Accident classifier as a confirmation filter:
    python trajectory_accident_detector.py --source test_video.mp4 --detector-weights yolov8n.pt \
        --classifier-weights runs/detect/accident_detection_model-5/weights/best.pt --display
"""

import argparse
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- CLI ARGUMENTS -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory-based accident detection")
    parser.add_argument("--detector-weights", type=str, default="yolov8n.pt",
                         help="Vehicle-detection model. A general COCO-pretrained model "
                              "(default) already detects car/truck/bus/motorcycle.")
    parser.add_argument("--classifier-weights", type=str, default=None,
                         help="Optional: path to your trained Accident/Non-Accident model, "
                              "used as a confirmation filter on flagged frames.")
    parser.add_argument("--source", type=str, default="0",
                         help="Video file path, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.35,
                         help="Confidence threshold for vehicle detections")
    parser.add_argument("--history-len", type=int, default=15,
                         help="Number of past frames to keep per vehicle for trajectory analysis")
    parser.add_argument("--direction-change-thresh", type=float, default=60.0,
                         help="Degrees of direction change (over the smoothing window) to flag as abnormal")
    parser.add_argument("--decel-thresh", type=float, default=0.6,
                         help="Fractional speed drop (0-1) to flag as sudden deceleration")
    parser.add_argument("--iou-spike-thresh", type=float, default=0.15,
                         help="IoU between two vehicle boxes above which we consider it a possible collision")
    parser.add_argument("--anomaly-score-thresh", type=float, default=0.6,
                         help="Combined anomaly score (0-1) needed to flag a frame as a likely accident")
    parser.add_argument("--consecutive-frames", type=int, default=3,
                         help="Consecutive anomalous frames needed before alerting")
    parser.add_argument("--output-dir", type=str, default="accident_alerts",
                         help="Folder to save alert clips and logs")
    parser.add_argument("--display", action="store_true",
                         help="Show live annotated video window")
    return parser.parse_args()


VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}


# ----------------------------- TRACK STATE -----------------------------

class VehicleTrack:
    """Keeps a short rolling history of one tracked vehicle's state."""

    def __init__(self, history_len):
        self.centroids = deque(maxlen=history_len)   # (x, y)
        self.box_sizes = deque(maxlen=history_len)    # (w, h)
        self.boxes = deque(maxlen=history_len)        # (x1,y1,x2,y2) for IoU checks
        self.timestamps = deque(maxlen=history_len)   # frame indices

    def update(self, box, frame_idx):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.centroids.append((cx, cy))
        self.box_sizes.append((w, h))
        self.boxes.append((x1, y1, x2, y2))
        self.timestamps.append(frame_idx)

    def smoothed_centroids(self, window=3):
        """Simple moving average to reduce detector-noise jitter."""
        pts = list(self.centroids)
        if len(pts) < window:
            return pts
        smoothed = []
        for i in range(len(pts)):
            lo = max(0, i - window + 1)
            chunk = pts[lo:i + 1]
            avg_x = sum(p[0] for p in chunk) / len(chunk)
            avg_y = sum(p[1] for p in chunk) / len(chunk)
            smoothed.append((avg_x, avg_y))
        return smoothed

    def direction_change_degrees(self):
        """Angle (degrees) between the earliest and latest movement vectors in history."""
        pts = self.smoothed_centroids()
        if len(pts) < 5:
            return 0.0
        early_vec = np.array(pts[len(pts) // 2]) - np.array(pts[0])
        late_vec = np.array(pts[-1]) - np.array(pts[len(pts) // 2])
        if np.linalg.norm(early_vec) < 1e-3 or np.linalg.norm(late_vec) < 1e-3:
            return 0.0
        cos_angle = np.dot(early_vec, late_vec) / (
            np.linalg.norm(early_vec) * np.linalg.norm(late_vec)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def speed_series(self):
        """Per-step speed (pixels/frame) from smoothed centroids."""
        pts = self.smoothed_centroids()
        speeds = []
        for i in range(1, len(pts)):
            d = np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1]))
            speeds.append(d)
        return speeds

    def deceleration_fraction(self):
        """Fractional drop from average early speed to latest speed. 0 = no change, 1 = stopped."""
        speeds = self.speed_series()
        if len(speeds) < 5:
            return 0.0
        early_avg = np.mean(speeds[:len(speeds) // 2]) if len(speeds) // 2 > 0 else 0.0
        late_speed = np.mean(speeds[-2:])
        if early_avg < 1.0:  # was already near-stationary; ignore to avoid noise
            return 0.0
        drop = (early_avg - late_speed) / early_avg
        return float(np.clip(drop, 0.0, 1.0))

    def size_change_fraction(self):
        """Sudden bounding-box area change, a proxy for spinning/flipping/crumpling."""
        sizes = list(self.box_sizes)
        if len(sizes) < 5:
            return 0.0
        areas = [w * h for w, h in sizes]
        early_avg = np.mean(areas[:len(areas) // 2])
        late_avg = np.mean(areas[-2:])
        if early_avg < 1.0:
            return 0.0
        change = abs(late_avg - early_avg) / early_avg
        return float(np.clip(change, 0.0, 1.0))


def compute_iou(box_a, box_b):
    """Standard IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# ----------------------------- ANOMALY SCORING -----------------------------

def compute_anomaly_score(track, args, all_current_boxes, this_id):
    """
    Combine direction change, deceleration, size change, and IoU overlap with
    other vehicles into one weighted anomaly score in [0, 1].
    """
    dir_change = track.direction_change_degrees()
    decel = track.deceleration_fraction()
    size_change = track.size_change_fraction()

    # Normalize direction change against threshold (cap at 1.0)
    dir_score = min(dir_change / args.direction_change_thresh, 1.0) if args.direction_change_thresh > 0 else 0.0
    decel_score = min(decel / args.decel_thresh, 1.0) if args.decel_thresh > 0 else 0.0
    size_score = min(size_change / 0.5, 1.0)  # size doubling/halving -> max score

    # IoU with any other currently-tracked vehicle
    max_iou = 0.0
    if track.boxes:
        this_box = track.boxes[-1]
        for other_id, other_box in all_current_boxes.items():
            if other_id == this_id:
                continue
            max_iou = max(max_iou, compute_iou(this_box, other_box))
    iou_score = min(max_iou / args.iou_spike_thresh, 1.0) if args.iou_spike_thresh > 0 else 0.0

    # Weighted combination - IoU overlap and direction change are the strongest
    # standalone signals; deceleration and size change support but don't dominate.
    score = (
        0.35 * dir_score +
        0.20 * decel_score +
        0.15 * size_score +
        0.30 * iou_score
    )
    return float(np.clip(score, 0.0, 1.0)), {
        "direction_change_deg": dir_change,
        "deceleration_frac": decel,
        "size_change_frac": size_change,
        "max_iou": max_iou,
    }


# ----------------------------- MAIN LOOP -----------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "trajectory_alerts_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,frame_number,track_id,anomaly_score,direction_change_deg,"
                     "deceleration_frac,size_change_frac,max_iou,classifier_confirmed\n")

    detector = YOLO(args.detector_weights)
    classifier = YOLO(args.classifier_weights) if args.classifier_weights else None

    source = int(args.source) if args.source.isdigit() else args.source

    tracks = defaultdict(lambda: VehicleTrack(args.history_len))
    consecutive_hits = defaultdict(int)

    fps_for_buffer = 20
    buffer_seconds = 3
    frame_buffer = deque(maxlen=int(fps_for_buffer * buffer_seconds))

    cooldown_counter = 0
    cooldown_frames = int(fps_for_buffer * 10)
    frame_number = 0

    print("Starting trajectory-based detection... press 'q' to quit if --display is on.")

    # model.track() gives us a generator of results with persistent IDs across frames
    results_stream = detector.track(
        source=source,
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
        classes=None,  # we'll filter by name below since class indices vary by model
    )

    for result in results_stream:
        frame_number += 1
        frame = result.orig_img
        frame_buffer.append(frame.copy())

        current_boxes = {}  # track_id -> (x1,y1,x2,y2) for this frame, used for IoU checks

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, cls_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                cls_name = detector.names[int(cls_id)]
                if cls_name.lower() not in VEHICLE_CLASS_NAMES:
                    continue
                tid = int(track_id)
                tracks[tid].update(tuple(box), frame_number)
                current_boxes[tid] = tuple(box)

        annotated_frame = result.plot()

        # Score every currently-visible vehicle
        any_alert_this_frame = False
        for tid, box in current_boxes.items():
            track = tracks[tid]
            score, details = compute_anomaly_score(track, args, current_boxes, tid)

            if score >= args.anomaly_score_thresh:
                consecutive_hits[tid] += 1
            else:
                consecutive_hits[tid] = 0

            # Draw the anomaly score above the vehicle for visual debugging
            x1, y1, x2, y2 = [int(v) for v in box]
            color = (0, 0, 255) if score >= args.anomaly_score_thresh else (0, 255, 0)
            cv2.putText(annotated_frame, f"ID{tid} score:{score:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if consecutive_hits[tid] >= args.consecutive_frames and cooldown_counter == 0:
                classifier_confirmed = "n/a"
                proceed_with_alert = True

                # Optional confirmation step using your trained classifier
                if classifier is not None:
                    crop = crop_with_margin(frame, box, margin=0.3)
                    if crop is not None and crop.size > 0:
                        cls_results = classifier.predict(crop, conf=0.25, verbose=False)
                        confirmed = any(
                            "accident" in classifier.names[int(c)].lower() and "non" not in classifier.names[int(c)].lower()
                            for c in cls_results[0].boxes.cls.cpu().numpy()
                        ) if cls_results[0].boxes is not None else False
                        classifier_confirmed = str(confirmed)
                        proceed_with_alert = confirmed

                if proceed_with_alert:
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    print(f"[ALERT] Possible accident: track ID {tid}, frame {frame_number}, "
                          f"score {score:.2f}, details {details}, classifier_confirmed={classifier_confirmed}")

                    with open(log_path, "a") as f:
                        f.write(f"{timestamp},{frame_number},{tid},{score:.3f},"
                                f"{details['direction_change_deg']:.1f},{details['deceleration_frac']:.3f},"
                                f"{details['size_change_frac']:.3f},{details['max_iou']:.3f},{classifier_confirmed}\n")

                    save_clip(frame_buffer, args.output_dir, timestamp, fps_for_buffer,
                              frame.shape[1], frame.shape[0])

                    cooldown_counter = cooldown_frames
                    consecutive_hits[tid] = 0
                    any_alert_this_frame = True

        if cooldown_counter > 0:
            cooldown_counter -= 1

        if args.display:
            cv2.imshow("Trajectory-Based Accident Detector", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"Done. Alerts logged to {log_path}")


def crop_with_margin(frame, box, margin=0.3):
    """Crop a region around a box with extra margin, for classifier confirmation."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_clip(frame_buffer, output_dir, timestamp, fps, width, height):
    clip_path = os.path.join(output_dir, f"accident_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for f in frame_buffer:
        writer.write(f)
    writer.release()
    print(f"  Saved clip: {clip_path}")


if __name__ == "__main__":
    main()
