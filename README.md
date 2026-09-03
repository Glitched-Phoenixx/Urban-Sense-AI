# Urban Sense AI
YOLOv8 baseed detector

## How to use -
1) Clone the repo
2) Open the folder in cmd

### For Pothole detector-
1) paste the following command -    python cli.py  --image road.jpg
2) replace --image road.jpg to --webcam to get live feed

### For CrowdnTraffic detector-
1) upgraded for crowd based heat density
2) contain both yolov8 and csrnet feature
3) to run crowd density, you have to download csrnet weights from the following github repo-
4) Go to https://github.com/leeyeehoo/CSRNet-pytorch and find the Google Drive links in its README for pretrained ShanghaiTech Part A or Part B weights. Download one .pth file.(I have Part A)
5) Put the downloaded content in weights folder
6) in Cmd run - python run.py csrnet --source test.jpg --weights weights/csrnet_shanghaiA.pth --save
7) !!! For now it only checks on video footage and image, live webcam is still unclear

### For Accident detector -
1) Many factors are added to increase accuracy, still inaccuracies exist in some cases.
2) In cmd paste - python trajectory_accident_detector.py --source your_test_video.mp4 --detector-weights best.pt --display.
3) It stores the alert footage seperately in accident_alerts folder and maintain a csv file.
4) Havent explored webcam option, so will update later.


**Sample footage and images added to easily check

