#!/usr/bin/env python3
"""
Density Heatmap Detector using CSRNet
=======================================

Estimates crowd/traffic DENSITY (not discrete bounding boxes) and classifies
congestion by how much of the frame area is covered by dense regions.

Unlike the YOLO-based detector.py (which counts individual boxes), this is
built for scenes where individuals are heavily occluded/overlapping and
box-detection undercounts badly (packed crowds, dense traffic jams).

IMPORTANT — pretrained weights required
----------------------------------------
There is no auto-download for CSRNet (unlike YOLO/ultralytics). You must
manually download a pretrained .pth checkpoint and pass it via --weights.

Common source (people/crowd density, trained on ShanghaiTech):
    https://github.com/leeyeehoo/CSRNet-pytorch
    (see its README for Google Drive links to partA/partB pretrained models)

Place the downloaded file in the weights/ folder, e.g.:
    weights/csrnet_shanghaiA.pth

Note: publicly available pretrained CSRNet weights are trained for PEOPLE
density. For vehicle/traffic density you'd need a model trained on a vehicle
density dataset (e.g. TRANCOS) with this same architecture — ask if you want
a training script for that.

Usage
-----
    # Image, default (percentile-based) area thresholding
    python csrnet_detector.py --source crowd.jpg --weights weights/csrnet_shanghaiA.pth --save

    # Video, custom area-% thresholds for LOW/MODERATE/HIGH
    python csrnet_detector.py --source video.mp4 --weights weights/csrnet_shanghaiA.pth \
        --thresholds 10 30 55 --save --output result.mp4

    # Webcam
    python csrnet_detector.py --source 0 --weights weights/csrnet_shanghaiA.pth
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    print(
        "ERROR: torch is not installed.\n"
        "Install dependencies first:\n\n"
        "    pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)

from models.csrnet import load_csrnet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_AREA_THRESHOLDS = (15, 35, 60)  # % of frame area covered by dense region

LEVEL_COLORS = {
    "LOW": (80, 200, 80),
    "MODERATE": (0, 200, 255),
    "HIGH": (0, 120, 255),
    "CRITICAL": (0, 0, 255),
}


def classify_level(area_pct: float, thresholds: tuple) -> str:
    low_max, moderate_max, high_max = thresholds
    if area_pct <= low_max:
        return "LOW"
    elif area_pct <= moderate_max:
        return "MODERATE"
    elif area_pct <= high_max:
        return "HIGH"
    return "CRITICAL"


@dataclass
class RollingCounter:
    window: int = 15
    history: deque = field(default_factory=deque)

    def update(self, value: float) -> float:
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.popleft()
        return sum(self.history) / len(self.history)


class CSRNetDetector:
    def __init__(
        self,
        weights: str,
        device: str | None = None,
        max_dim: int = 1024,
        area_method: str = "percentile",
        percentile: float = 75.0,
        density_threshold: float | None = None,
        thresholds: tuple = DEFAULT_AREA_THRESHOLDS,
        smoothing_window: int = 15,
    ):
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"CSRNet weights not found at '{weights}'.\n"
                "Download a pretrained .pth checkpoint (see script docstring) "
                "and pass its path via --weights."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_csrnet(weights, device=self.device)
        self.max_dim = max_dim
        self.area_method = area_method
        self.percentile = percentile
        self.density_threshold = density_threshold
        self.thresholds = thresholds

        self.count_roll = RollingCounter(window=smoothing_window)
        self.area_roll = RollingCounter(window=smoothing_window)

    # ------------------------------------------------------------------ #

    def _preprocess(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_dim / max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        # round down to nearest multiple of 8 (3 max-pools in frontend)
        new_w -= new_w % 8
        new_h -= new_h % 8
        new_w, new_h = max(new_w, 8), max(new_h, 8)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device), (w, h)

    def infer(self, frame_bgr: np.ndarray):
        tensor, (orig_w, orig_h) = self._preprocess(frame_bgr)
        with torch.no_grad():
            output = self.model(tensor)
        density_map = output.squeeze().cpu().numpy()
        density_map = np.clip(density_map, a_min=0, a_max=None)

        count = float(density_map.sum())  # count from RAW (pre-upsample) map

        # upsample only for visualization / area computation against full frame
        density_resized = cv2.resize(
            density_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
        )
        density_resized = np.clip(density_resized, a_min=0, a_max=None)

        return count, density_resized

    def compute_mask(self, density_resized: np.ndarray):
        positive = density_resized[density_resized > 0]

        if self.density_threshold is not None:
            thresh_val = self.density_threshold
        elif positive.size > 0:
            thresh_val = float(np.percentile(positive, self.percentile))
        else:
            thresh_val = float("inf")  # nothing positive -> empty mask

        mask = density_resized > thresh_val
        area_pct = 100.0 * mask.sum() / mask.size
        return mask, area_pct

    def summarize(self, count: float, density_resized: np.ndarray):
        mask, area_pct = self.compute_mask(density_resized)
        smoothed_count = self.count_roll.update(count)
        smoothed_area = self.area_roll.update(area_pct)
        level = classify_level(smoothed_area, self.thresholds)
        return {
            "raw_count": round(count, 1),
            "smoothed_count": round(smoothed_count, 1),
            "area_pct": round(area_pct, 1),
            "smoothed_area_pct": round(smoothed_area, 1),
            "level": level,
            "mask": mask,
        }

    # ------------------------------------------------------------------ #

    def annotate(self, frame: np.ndarray, density_resized: np.ndarray, summary: dict) -> np.ndarray:
        max_val = density_resized.max()
        norm = density_resized / (max_val + 1e-6)
        heatmap_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.55, heatmap_color, 0.45, 0)

        # outline the thresholded "dense" region used for classification
        mask_u8 = (summary["mask"].astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = LEVEL_COLORS[summary["level"]]
        cv2.drawContours(blended, contours, -1, color, 2)

        # status panel
        pad, line_h = 10, 26
        lines = [
            f"Est. count: {summary['smoothed_count']}",
            f"Dense area: {summary['smoothed_area_pct']}%  [{summary['level']}]",
        ]
        panel_h = pad * 2 + line_h * len(lines)
        panel_w = 380
        overlay = blended.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
        blended = cv2.addWeighted(overlay, 0.55, blended, 0.45, 0)
        for i, text in enumerate(lines):
            y = pad + line_h * (i + 1) - 6
            line_color = color if i == 1 else (255, 255, 255)
            cv2.putText(blended, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, line_color, 2, cv2.LINE_AA)

        return blended

    # ------------------------------------------------------------------ #

    def process_image(self, path: str, save_path: str | None = None, show: bool = True):
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        count, density_resized = self.infer(frame)
        summary = self.summarize(count, density_resized)
        annotated = self.annotate(frame, density_resized, summary)

        self._print_summary(summary, frame_id=None)

        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"Saved annotated image to {save_path}")

        if show:
            cv2.imshow("CSRNet Density Detector", annotated)
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

                count, density_resized = self.infer(frame)
                summary = self.summarize(count, density_resized)
                annotated = self.annotate(frame, density_resized, summary)

                if frame_id % int(max(fps, 1)) == 0:
                    self._print_summary(summary, frame_id=frame_id)

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("CSRNet Density Detector", annotated)
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

    @staticmethod
    def _print_summary(summary: dict, frame_id: int | None):
        prefix = f"[frame {frame_id}] " if frame_id is not None else ""
        print(
            f"{prefix}Est. count={summary['smoothed_count']} "
            f"Dense area={summary['smoothed_area_pct']}% "
            f"Level={summary['level']}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Density Heatmap Detector using CSRNet")
    p.add_argument("--source", required=True,
                   help="Path to image/video file, webcam index (e.g. 0), or RTSP URL")
    p.add_argument("--weights", required=True,
                   help="Path to pretrained CSRNet .pth checkpoint (see script docstring)")
    p.add_argument("--device", default=None, help="'cpu', 'cuda', or None for auto")
    p.add_argument("--max-dim", type=int, default=1024,
                   help="Resize longer side to this before inference (speed/memory tradeoff)")
    p.add_argument("--area-method", choices=["percentile", "absolute"], default="percentile",
                   help="How to threshold the density map into a 'dense region' mask")
    p.add_argument("--percentile", type=float, default=75.0,
                   help="Percentile of positive density values used as the mask threshold "
                        "(only used when --area-method=percentile)")
    p.add_argument("--density-threshold", type=float, default=None,
                   help="Absolute per-pixel density threshold (only used when "
                        "--area-method=absolute; try printing density stats first to tune)")
    p.add_argument("--thresholds", nargs=3, type=float, metavar=("LOW", "MODERATE", "HIGH"),
                   default=DEFAULT_AREA_THRESHOLDS,
                   help="Dense-area %% thresholds for LOW/MODERATE/HIGH "
                        f"(default: {DEFAULT_AREA_THRESHOLDS})")
    p.add_argument("--save", action="store_true", help="Save annotated output")
    p.add_argument("--output", default=None, help="Output file path (image or video)")
    p.add_argument("--no-show", action="store_true", help="Do not open a display window")
    p.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame only")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    p.add_argument("--smoothing-window", type=int, default=15,
                   help="Rolling average window size (frames) for smoothing")
    return p.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    args = parse_args()

    area_method = args.area_method
    density_threshold = args.density_threshold
    if area_method == "absolute" and density_threshold is None:
        print("ERROR: --area-method=absolute requires --density-threshold VALUE", file=sys.stderr)
        sys.exit(1)

    detector = CSRNetDetector(
        weights=args.weights,
        device=args.device,
        max_dim=args.max_dim,
        area_method=area_method,
        percentile=args.percentile,
        density_threshold=density_threshold,
        thresholds=tuple(args.thresholds),
        smoothing_window=args.smoothing_window,
    )

    source = args.source
    if source.isdigit():
        source = int(source)

    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS

    save_path = args.output
    if args.save and not save_path:
        save_path = "output_density.jpg" if is_image else "output_density.mp4"

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
