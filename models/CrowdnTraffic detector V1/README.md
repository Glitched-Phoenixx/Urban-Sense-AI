# Crowd / Traffic Detector (YOLO)

Detects and counts people and/or vehicles in an image, video file, or live
camera/RTSP stream using YOLOv8 (via [Ultralytics](https://github.com/ultralytics/ultralytics)),
and classifies the scene into a congestion level: **LOW / MODERATE / HIGH / CRITICAL**.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will auto-download the default weights (`yolov8n.pt`, ~6 MB) from
Ultralytics. No manual download needed, but you do need an internet connection
the first time.

> **GPU (optional):** if you have an NVIDIA GPU, install a CUDA-enabled build of
> PyTorch first (see https://pytorch.org/get-started/locally/) — Ultralytics will
> then use it automatically.

## 2. Quick start

```bash
# Count people in a photo of a crowd
python detector.py --source crowd.jpg --mode crowd --save

# Count vehicles in a traffic video
python detector.py --source traffic.mp4 --mode traffic --save --output result.mp4

# Count both people and vehicles from your webcam
python detector.py --source 0 --mode both

# RTSP camera
python detector.py --source rtsp://user:pass@192.168.1.10:554/stream --mode both
```

An annotated window pops up showing bounding boxes plus a live counter panel.
Press `q` to quit a video/stream. Use `--no-show` for headless servers.

## 3. How it works

- Runs YOLO inference per frame, restricted to the relevant COCO classes:
  - **crowd** mode → `person`
  - **traffic** mode → `car, motorcycle, bus, truck, bicycle, train`
  - **both** → all of the above, counted separately
- Applies a rolling average (`--smoothing-window`, default 15 frames) so the
  congestion level doesn't flicker on momentary detection noise.
- Compares the (smoothed) count against thresholds to assign a level:

  | Level     | Meaning                          |
  |-----------|-----------------------------------|
  | LOW       | count ≤ low threshold             |
  | MODERATE  | low < count ≤ moderate threshold  |
  | HIGH      | moderate < count ≤ high threshold |
  | CRITICAL  | count > high threshold            |

  Defaults: crowd = `(10, 25, 50)`, traffic = `(5, 15, 30)`.
  Override with `--thresholds LOW MODERATE HIGH`, e.g.:

  ```bash
  python detector.py --source lobby.mp4 --mode crowd --thresholds 5 15 30
  ```

## 4. Useful flags

| Flag | Purpose |
|---|---|
| `--weights` | Swap in a different/custom-trained YOLO model (`yolov8s.pt`, `yolov8m.pt`, or your own `best.pt`) |
| `--conf` | Detection confidence threshold (default `0.35`) |
| `--device` | Force `cpu` or `cuda:0`; auto-detected otherwise |
| `--frame-skip` | Skip frames to speed up processing on long videos |
| `--max-frames` | Stop after N frames (useful for testing) |
| `--save` / `--output` | Save annotated image/video to disk |

## 5. Notes on accuracy & tuning

- `yolov8n.pt` (nano) is fastest but least accurate. For denser crowds or
  small/far vehicles, try `yolov8s.pt` or `yolov8m.pt` for better recall
  (slower).
- Very dense crowds (100+ tightly packed people) are a hard case for general
  object detectors due to occlusion — for serious crowd-counting at that
  density, a dedicated density-estimation model (e.g. CSRNet) generally
  outperforms bounding-box detectors. This script is best suited for
  low-to-moderate density scenes (plazas, entrances, intersections, parking
  lots) and general traffic monitoring.
- If you have your own labeled dataset (e.g. from a specific camera angle),
  fine-tuning YOLO on it (`model.train(...)`) will noticeably improve results
  over the stock COCO weights — ask if you'd like a training script too.

## 6. Files

- `detector.py` — main CLI tool (all logic lives here)
- `requirements.txt` — Python dependencies
