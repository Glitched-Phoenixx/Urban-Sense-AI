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

## 6. CSRNet (Density Heatmap) Engine

A second, swappable engine for scenes where individuals are too densely
packed/occluded for box-based detection (YOLO) to count accurately —
concerts, protests, gridlocked traffic. Instead of drawing boxes, CSRNet
predicts a **density heatmap** over the whole frame, and congestion is
classified by **what % of the frame area is covered by dense regions** —
not by a raw object count.

### You must download weights manually

There's no auto-download for CSRNet like there is for YOLO. Get a pretrained
checkpoint (trained on ShanghaiTech, i.e. **people** density) from:

- https://github.com/leeyeehoo/CSRNet-pytorch (README has Google Drive links
  to partA/partB pretrained `.pth` files)

Drop the file into `weights/`, e.g. `weights/csrnet_shanghaiA.pth`.

> **Vehicles/traffic:** publicly shared pretrained CSRNet weights are for
> people-density, not vehicles. For traffic density you'd need a model
> trained on a vehicle-density dataset (e.g. TRANCOS) using this same
> architecture. Ask if you'd like a training script for that — it's a
> separate, more involved effort since you'd need labeled density-map data.

### Usage

```bash
# Image
python csrnet_detector.py --source crowd.jpg --weights weights/csrnet_shanghaiA.pth --save

# Video with custom area-% thresholds for LOW/MODERATE/HIGH
python csrnet_detector.py --source video.mp4 --weights weights/csrnet_shanghaiA.pth \
    --thresholds 10 30 55 --save --output result.mp4

# Webcam
python csrnet_detector.py --source 0 --weights weights/csrnet_shanghaiA.pth
```

### How the area-based classification works

1. CSRNet outputs a per-pixel density map (upsampled to match the frame).
2. Pixels above a threshold are marked "dense" — by default the threshold is
   the 75th percentile of positive density values in that frame
   (`--percentile`), which adapts per-scene without manual tuning. You can
   instead fix an absolute threshold with `--area-method absolute
   --density-threshold VALUE` once you've seen typical values for your
   camera.
3. `dense_pixel_count / total_pixel_count * 100` = **area %**.
4. Area % is compared against `--thresholds LOW MODERATE HIGH` (default
   `15 35 60`, meaning ≤15% = LOW, ≤35% = MODERATE, ≤60% = HIGH, above = CRITICAL).

The overlay shows the heatmap, outlines the exact "dense" region used for
classification, and also reports an estimated count (sum of the raw density
map) as a secondary number — but the **congestion level itself is driven by
area coverage**, per your idea, since that's what actually reflects how much
of the space is occupied regardless of occlusion.

### Tuning tips

- If everything always reads CRITICAL or always LOW, adjust `--percentile`
  (lower = more pixels count as "dense" = higher area%) or the
  `--thresholds` cutoffs for your specific camera framing.
- `--max-dim` controls the internal inference resolution (default 1024px on
  the longer side) — lower it for speed on CPU, raise it for accuracy on
  dense scenes.

## 7. Switching Between Engines

Use `run.py` to call either engine without remembering two script names:

```bash
python run.py yolo --source video.mp4 --mode both
python run.py csrnet --source video.mp4 --weights weights/csrnet_shanghaiA.pth
```

Everything after `yolo`/`csrnet` passes straight through to that engine's
own CLI, so all the flags documented above still apply. Handy for running
both engines on the same footage back-to-back and comparing results.

## 8. Which engine should I use?

| Scenario | Recommended engine |
|---|---|
| Moderate crowds/traffic, individuals mostly distinguishable | **YOLO** (`detector.py`) — gives exact counts, vehicle type breakdown, per-object boxes |
| Packed/overlapping crowds, gridlock where boxes would undercount | **CSRNet** (`csrnet_detector.py`) — density-based, doesn't rely on separating individuals |
| Not sure yet | Run both (`run.py`) on the same clip and compare — cheap to test since neither requires training for people-counting |

## 9. Files

- `detector.py` — YOLO-based box detector/counter
- `csrnet_detector.py` — CSRNet-based density heatmap detector
- `models/csrnet.py` — CSRNet model architecture definition
- `run.py` — unified CLI to switch between engines
- `weights/` — put your downloaded CSRNet `.pth` checkpoint here
- `requirements.txt` — Python dependencies
