"""
Pothole Detector - YOLOv8 (deep learning)
------------------------------------------
Trains and/or runs a YOLOv8 object detector for pothole detection.
Much more accurate than classical CV on varied lighting, angles,
and road textures, but requires a labeled dataset to train on
(or a pretrained pothole weights file).

Requirements:
    pip install ultralytics opencv-python

--------------------------------------------------------------------
1) GET A DATASET
--------------------------------------------------------------------
You need images with YOLO-format bounding box labels (class, x_center,
y_center, width, height - normalized 0-1). Easiest sources:

  - Roboflow Universe: search "pothole detection" -> many public
    datasets, exportable directly in "YOLOv8" format as a .zip
    containing data.yaml + train/valid/test folders.
    https://universe.roboflow.com

  - Kaggle: search "pothole dataset" (some need conversion to YOLO
    format from COCO/VOC - Roboflow can do this conversion for you
    by importing then re-exporting).

Unzip the dataset so you have a structure like:

    pothole_dataset/
      data.yaml
      train/images, train/labels
      valid/images, valid/labels
      test/images,  test/labels  (optional)

--------------------------------------------------------------------
2) TRAIN
--------------------------------------------------------------------
    python yolo_pothole_detector.py --train --data pothole_dataset/data.yaml --epochs 50

This fine-tunes a pretrained YOLOv8n (nano, fast) checkpoint on your
pothole dataset. Swap --model to yolov8s.pt / yolov8m.pt for higher
accuracy at the cost of speed.

--------------------------------------------------------------------
3) DETECT
--------------------------------------------------------------------
    python yolo_pothole_detector.py --detect --weights runs/detect/train/weights/best.pt --image road.jpg
    python yolo_pothole_detector.py --detect --weights runs/detect/train/weights/best.pt --video road.mp4
    python yolo_pothole_detector.py --detect --weights runs/detect/train/weights/best.pt --webcam
"""

from potholeModel import model


class PotholeDetector:
    def __init__(self, conf = 0.35):
        self.model = model
        self.conf = conf

    def predict(self, frame):
        results = self.model.predict(
            source = frame,
            conf = self.conf,
            verbose = False
        )
        return results[0]

    def get_detections(self, results):
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = [
                round(v, 1) 
                for v in box.xyxy[0].tolist()
            ]

            detections.append({
                "class_name": self.model.names[cls_id],
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    def annotate(self, results):
        return results.plot()
