# Urban Sense AI
YOLOv8 baseed pothole detector

## How to use -
1) Clone the repo
2) Open the folder in cmd

### For Pothole detector-
1) paste the following command -    python yolo_pothole_detector.py --detect --weights runs/detect/pothole_yolov8/weights/best.pt --image road.jpg
2) replace --image road.jpg to --webcam to get live feed

### For CrowdnTraffic detector-
1) to use webcam use - python detector.py --source 0 --mode both
2) to use video footage - python detector.py --source myvideo.mp4 --mode both

### For Accident detector (inaccuracies still exist) -
1)to use webcam use - python detect_accidents.py --weights "runs/detect/accident_model_5/weights/best.pt" --source 0 --display
