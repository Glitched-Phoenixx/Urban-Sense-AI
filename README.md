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

### For Accident detector (inaccuracies still exist) -
1)to use webcam use - python detect_accidents.py --weights "runs/detect/accident_model_5/weights/best.pt" --source 0 --display
