import argparse
import cv2

from yolo_pothole_detector import PotholeDetector
from alert_logger import AlertStore, get_ip_location


def process_image(detector, alerts, image_path, output_path, lat, lng, location_source):
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    result = detector.predict(frame)
    detections = detector.get_detections(result)
    annotated = detector.annotate(result)

    print(f"\nDetected {len(detections)} pothole(s):")
    for detection in detections:
        print(
            f"  - {detection['class_name']} "
            f"conf={detection['confidence']} "
            f"bbox={detection['bbox']}"
        )
        alerts.record(detection, frame, lat, lng, location_source, source="image")

    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to: {output_path}")
    print(alerts.summary())


def process_video(detector, alerts, video_path, output_path, lat, lng, location_source):
    cap = cv2.VideoCapture(video_path)  # fixed: was cv2.videoCapture (bad capitalization)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # fixed: was cv2.videoWriter("*mp4v") (wrong call entirely)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # fixed: was cv2.videoWriter(...)

    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame)
        detections = detector.get_detections(results)
        total_detections += len(detections)
        annotated = detector.annotate(results)

        for detection in detections:
            alerts.record(detection, frame, lat, lng, location_source, source="video")

        writer.write(annotated)
        cv2.imshow("Pothole Detection: ", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nTotal detections across video: {total_detections}")
    print(f"Annotated video saved to: {output_path}")
    print(alerts.summary())


def process_webcam(detector, alerts, camera_index, lat, lng, location_source):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            f"Most laptops use index 0 — try --camera-index 0 if this fails."
        )

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame)
        detections = detector.get_detections(results)
        annotated = detector.annotate(results)

        for detection in detections:
            alerts.record(detection, frame, lat, lng, location_source, source="webcam")

        cv2.imshow("Pothole Detections", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(alerts.summary())


def resolve_location(args):
    if args.lat is not None and args.lon is not None:
        return args.lat, args.lon, "manual"
    print("No --lat/--lon given — trying an IP-based location estimate (city-level accuracy only)...")
    lat, lng, source = get_ip_location()
    if source == "unavailable":
        print("Could not determine location. Alerts will be saved with location: null.")
        print("Pass --lat <value> --lon <value> for accurate coordinates.")
    else:
        print(f"Using approximate location from IP: ({lat}, {lng})")
    return lat, lng, source


def main():
    parser = argparse.ArgumentParser(
        description="Pothole Detector using YOLOv8"
    )

    parser.add_argument("--image", type=str, help="Add path to image")
    parser.add_argument("--video", type=str, help="Add path to video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--camera-index", type=int, default=0,
                         help="Webcam device index (default 0 — the built-in camera on most laptops)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output", help="Output file path")
    parser.add_argument("--lat", type=float, default=None, help="Latitude to tag alerts with")
    parser.add_argument("--lon", type=float, default=None, help="Longitude to tag alerts with")

    args = parser.parse_args()

    detector = PotholeDetector(conf=args.conf)
    alerts = AlertStore()
    lat, lng, location_source = resolve_location(args)

    if args.image:
        output_path = args.output
        if not output_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            output_path += ".jpg"
        process_image(detector, alerts, args.image, output_path, lat, lng, location_source)

    elif args.video:
        output_path = args.output
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
        process_video(detector, alerts, args.video, output_path, lat, lng, location_source)

    elif args.webcam:
        process_webcam(detector, alerts, args.camera_index, lat, lng, location_source)

    else:
        parser.error("Not a valid input. Try --image, --video, --webcam, --output, --conf, --lat, --lon")


if __name__ == "__main__":
    main()
