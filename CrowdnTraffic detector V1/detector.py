#!/usr/bin/env python3
"""
Crowd / Traffic Detector using YOLO (Ultralytics YOLOv8/v11)
==============================================================

Detects and counts people (crowd) and vehicles (traffic) in an image,
video file, or live webcam/RTSP stream, and classifies the scene into
congestion levels (e.g. LOW / MODERATE / HIGH / CRITICAL) based on
configurable thresholds.

Usage
-----
    # Single image
    python detector.py --source path/to/image.jpg --mode crowd

    # Video file
    python detector.py --source path/to/video.mp4 --mode traffic

    # Webcam (device 0)
    python detector.py --source 0 --mode both

    # RTSP / IP camera stream
    python detector.py --source rtsp://user:pass@ip:port/stream --mode both

    # Custom thresholds + save annotated output
    python detector.py --source video.mp4 --mode crowd \
        --thresholds 5 15 30 --save --output out.mp4

Modes
-----
    crowd    -> counts only "person"
    traffic  -> counts vehicle classes (car, motorcycle, bus, truck, bicycle, train)
    both     -> counts and reports both independently

Requires
--------
    pip install -r requirements.txt
    (First run auto-downloads the YOLO weights, e.g. yolov8n.pt, ~6MB)
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: ultralytics is not installed.\n"
        "Install dependencies first:\n\n"
        "    pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Class configuration (COCO class names used by stock YOLOv8/v11 weights)
# --------------------------------------------------------------------------- #

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "train"}

DEFAULT_THRESHOLDS = {
    # (low_max, moderate_max, high_max) -> above high_max is CRITICAL
    "crowd": (10, 25, 50),
    "traffic": (5, 15, 30),
}

LEVEL_COLORS = {
    "LOW": (80, 200, 80),        # green (BGR)
    "MODERATE": (0, 200, 255),   # yellow/orange
    "HIGH": (0, 120, 255),       # orange/red
    "CRITICAL": (0, 0, 255),     # red
}


def classify_level(count: int, thresholds: tuple) -> str:
    low_max, moderate_max, high_max = thresholds
    if count <= low_max:
        return "LOW"
    elif count <= moderate_max:
        return "MODERATE"
    elif count <= high_max:
        return "HIGH"
    return "CRITICAL"


# --------------------------------------------------------------------------- #
# Rolling stats helper (for smoothing counts across video frames)
# --------------------------------------------------------------------------- #

@dataclass
class RollingCounter:
    window: int = 15
    history: deque = field(default_factory=deque)

    def update(self, value: int) -> float:
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.popleft()
        return sum(self.history) / len(self.history)


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #

class CrowdTrafficDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        mode: str = "both",
        thresholds: dict | None = None,
        device: str | None = None,
        smoothing_window: int = 15,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.mode = mode
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.device = device
        self.names = self.model.names  # class id -> name

        self.person_ids = {i for i, n in self.names.items() if n in PERSON_CLASSES}
        self.vehicle_ids = {i for i, n in self.names.items() if n in VEHICLE_CLASSES}

        self.crowd_roll = RollingCounter(window=smoothing_window)
        self.traffic_roll = RollingCounter(window=smoothing_window)

    # ------------------------------------------------------------------ #

    def _target_class_ids(self):
        ids = set()
        if self.mode in ("crowd", "both"):
            ids |= self.person_ids
        if self.mode in ("traffic", "both"):
            ids |= self.vehicle_ids
        return list(ids)

    def infer(self, frame: np.ndarray):
        """Run YOLO on a single frame, return the Results object."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self._target_class_ids(),
            device=self.device,
            verbose=False,
        )[0]
        return results

    def count_by_class(self, results) -> dict:
        counts = defaultdict(int)
        if results.boxes is None:
            return counts
        for cls_id in results.boxes.cls.tolist():
            counts[self.names[int(cls_id)]] += 1
        return counts

    def summarize(self, counts: dict) -> dict:
        person_count = sum(v for k, v in counts.items() if k in PERSON_CLASSES)
        vehicle_count = sum(v for k, v in counts.items() if k in VEHICLE_CLASSES)

        summary = {}
        if self.mode in ("crowd", "both"):
            smoothed = self.crowd_roll.update(person_count)
            summary["crowd"] = {
                "raw_count": person_count,
                "smoothed_count": round(smoothed, 1),
                "level": classify_level(round(smoothed), self.thresholds["crowd"]),
            }
        if self.mode in ("traffic", "both"):
            smoothed = self.traffic_roll.update(vehicle_count)
            summary["traffic"] = {
                "raw_count": vehicle_count,
                "smoothed_count": round(smoothed, 1),
                "level": classify_level(round(smoothed), self.thresholds["traffic"]),
                "by_type": {k: v for k, v in counts.items() if k in VEHICLE_CLASSES},
            }
        return summary

    # ------------------------------------------------------------------ #

    def annotate(self, frame: np.ndarray, results, summary: dict) -> np.ndarray:
        annotated = results.plot()  # draws boxes + labels using ultralytics' own renderer

        # Overlay a status panel (top-left) with counts + congestion level
        pad, line_h = 10, 26
        lines = []
        if "crowd" in summary:
            c = summary["crowd"]
            lines.append((f"People: {c['raw_count']} (avg {c['smoothed_count']})  [{c['level']}]",
                           LEVEL_COLORS[c["level"]]))
        if "traffic" in summary:
            t = summary["traffic"]
            lines.append((f"Vehicles: {t['raw_count']} (avg {t['smoothed_count']})  [{t['level']}]",
                           LEVEL_COLORS[t["level"]]))

        panel_h = pad * 2 + line_h * len(lines)
        panel_w = 430
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
        annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

        for i, (text, color) in enumerate(lines):
            y = pad + line_h * (i + 1) - 6
            cv2.putText(annotated, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)

        return annotated

    # ------------------------------------------------------------------ #

    def process_image(self, path: str, save_path: str | None = None, show: bool = True):
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        results = self.infer(frame)
        counts = self.count_by_class(results)
        summary = self.summarize(counts)
        annotated = self.annotate(frame, results, summary)

        self._print_summary(summary, frame_id=None)

        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"Saved annotated image to {save_path}")

        if show:
            cv2.imshow("Crowd / Traffic Detector", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return summary

    def process_stream(
        self,
        source,
        save_path: str | None = None,
        show: bool = True,
        frame_skip: int = 0,
        max_frames: int | None = None,
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_id = 0
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1

                if frame_skip and frame_id % (frame_skip + 1) != 0:
                    continue

                results = self.infer(frame)
                counts = self.count_by_class(results)
                summary = self.summarize(counts)
                annotated = self.annotate(frame, results, summary)

                if frame_id % int(max(fps, 1)) == 0:  # print roughly once per second
                    self._print_summary(summary, frame_id=frame_id)

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("Crowd / Traffic Detector", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames and frame_id >= max_frames:
                    break
        finally:
            elapsed = time.time() - t0
            print(f"\nProcessed {frame_id} frames in {elapsed:.1f}s "
                  f"({frame_id / max(elapsed, 1e-6):.1f} FPS)")
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_summary(summary: dict, frame_id: int | None):
        prefix = f"[frame {frame_id}] " if frame_id is not None else ""
        parts = []
        if "crowd" in summary:
            c = summary["crowd"]
            parts.append(f"People={c['raw_count']} ({c['level']})")
        if "traffic" in summary:
            t = summary["traffic"]
            parts.append(f"Vehicles={t['raw_count']} ({t['level']})")
        print(prefix + " | ".join(parts))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Crowd / Traffic Detector using YOLO")
    p.add_argument("--source", required=True,
                   help="Path to image/video file, webcam index (e.g. 0), or RTSP URL")
    p.add_argument("--mode", choices=["crowd", "traffic", "both"], default="both",
                   help="What to detect/count (default: both)")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="YOLO weights file (default: yolov8n.pt, auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default=None,
                   help="Inference device: 'cpu', 'cuda:0', or None for auto")
    p.add_argument("--thresholds", nargs=3, type=int, metavar=("LOW", "MODERATE", "HIGH"),
                   help="Custom count thresholds for the *primary* mode "
                        "(applies to crowd if mode=crowd/both, else traffic)")
    p.add_argument("--save", action="store_true", help="Save annotated output")
    p.add_argument("--output", default=None, help="Output file path (image or video)")
    p.add_argument("--no-show", action="store_true", help="Do not open a display window")
    p.add_argument("--frame-skip", type=int, default=0,
                   help="Process every Nth frame only (video/stream), e.g. 1 = skip every other frame")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (video/stream)")
    p.add_argument("--smoothing-window", type=int, default=15,
                   help="Rolling average window size (frames) for smoothing counts")
    return p.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    args = parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        key = "crowd" if args.mode in ("crowd", "both") else "traffic"
        thresholds[key] = tuple(args.thresholds)

    detector = CrowdTrafficDetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        mode=args.mode,
        thresholds=thresholds,
        device=args.device,
        smoothing_window=args.smoothing_window,
    )

    source = args.source
    # webcam index passed as a plain integer string
    if source.isdigit():
        source = int(source)

    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS

    save_path = args.output
    if args.save and not save_path:
        save_path = "output.jpg" if is_image else "output.mp4"

    show = not args.no_show

    if is_image:
        detector.process_image(source, save_path=save_path, show=show)
    else:
        detector.process_stream(
            source,
            save_path=save_path,
            show=show,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
