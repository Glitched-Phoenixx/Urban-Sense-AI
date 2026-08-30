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

import argparse
import os
from ultralytics import YOLO


def train(data_yaml, epochs=50, imgsz=640, base_model="yolov8n.pt"):
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=15,       # early stopping
        batch=16,
        name="pothole_yolov8"
    )
    metrics = model.val()
    print("Validation metrics:", metrics)
    print("Best weights saved under runs/detect/pothole_yolov8*/weights/best.pt")


def detect_image(weights, image_path, out_path="pothole_yolo_output.jpg", conf=0.35):
    model = YOLO(weights)
    results = model.predict(source=image_path, conf=conf, save=False)
    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes drawn
    import cv2
    cv2.imwrite(out_path, annotated)

    print(f"Detected {len(result.boxes)} pothole(s). Saved to {out_path}")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  - {model.names[cls_id]}  conf={confidence:.2f}  box=({x1},{y1},{x2},{y2})")


def detect_video(weights, video_path, out_path="pothole_yolo_output.mp4", conf=0.35):
    import cv2
    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        total += len(results[0].boxes)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Total detections across video: {total}. Saved to {out_path}")


def detect_webcam(weights, conf=0.35):
    import cv2
    model = YOLO(weights)
    cap = cv2.VideoCapture(1)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Pothole Detection (YOLOv8)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/run a YOLOv8 pothole detector.")
    parser.add_argument("--train", action="store_true", help="Train on a custom dataset")
    parser.add_argument("--detect", action="store_true", help="Run inference")
    parser.add_argument("--data", type=str, help="Path to data.yaml (for training)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--weights", type=str, help="Trained weights (.pt) for detection")
    parser.add_argument("--image", type=str, help="Image path for detection")
    parser.add_argument("--video", type=str, help="Video path for detection")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam feed for detection")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    if args.train:
        if not args.data:
            raise SystemExit("--data <path to data.yaml> is required for training")
        train(args.data, epochs=args.epochs, base_model=args.model)

    elif args.detect:
        if not args.weights:
            raise SystemExit("--weights <path to .pt file> is required for detection")
        if args.image:
            detect_image(args.weights, args.image, conf=args.conf)
        elif args.video:
            detect_video(args.weights, args.video, conf=args.conf)
        elif args.webcam:
            detect_webcam(args.weights, conf=args.conf)
        else:
            raise SystemExit("Specify --image, --video, or --webcam with --detect")
    else:
        print("Specify --train or --detect. Run with -h for details.")
