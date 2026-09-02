import argparse
import cv2
from yolo_pothole_detector import PotholeDetector

def process_image(detector, image_path, output_path):
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    result = detector.predict(frame)

    detections = detector.get_detections(result)

    annotated = detector.annotate(result)

    print(f"\nDetected {len(detections)} pothole(s):")

    for detection in detections:
        print(
            f"  - {detection['class_name']}"
            f"conf - {detection['confidence']}"
            f"bbox - {detection['bbox']}"
        )

    cv2.imwrite(output_path, annotated)

    print(f"Annotate image saved to: {output_path}")

def process_video(detector, video_path, output_path):
    cap = cv2.videoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.videoWriter("*mp4v")
    writer = cv2.videoWriter(
        output_path,
        fourcc,
        fps,
        [width, height]
    )

    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame)
        detections = detector.get_detections(results)
        total_detections += len(detections)
        annotated = detector.annotate(results)

        writer.write(annotated)

        cv2.imshow("Pothole Detection: ", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nTotal Detections across video: {total_detections}")
    print(f"Annotated video saved to: {output_path}")

def process_webcam(detector):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open WebCam.")

    print("Press 'q' to quit.")

    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.predict(frame)
        annotated = detector.annotate(results)

        cv2.imshow("Pothole Detections", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Pothole Detector using YOLOv8"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Add path to image"
    )

    parser.add_argument(
        "--video",
        type= str,
        help= "Add path to video"
    )

    parser.add_argument(
        "--webcam",
        action= "store_true",
        help= "Use webcam"
    )

    parser.add_argument(
        "--conf",
        type = float,
        default= 0.35,
        help= "Confidence Threshold"
    )

    parser.add_argument(
        "--output", 
        type= str,
        default= "output",
        help= "Output file path"
    )

    args = parser.parse_args()

    detector = PotholeDetector(conf = args.conf)

    if args.image:
        output_path = args.output

        if not output_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            output_path += ".jpg"

        process_image(
            detector,
            args.image,
            output_path
        )

    elif args.video:
        output_path = args.output

        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"

        process_video(
            detector,
            args.video,
            output_path
        )

    elif args.webcam:
        process_webcam(detector)

    else: 
        parser.error("Not a valid input. Try --image, --video, --webcam, --output and --conf")

if __name__ == "__main__":
    main()
