"""
alert_logger.py
----------------
Persists pothole detections as deduplicated JSON alerts, ready to sync
to a server. Used by cli.py during --image / --video / --webcam runs.

Each alert on disk looks like:
{
  "alert_id": "a3f1c2b9e7d1",
  "class_name": "pothole",
  "location": {"lat": 20.2961, "lng": 85.8245},
  "location_source": "manual" | "ip_estimate" | "unavailable",
  "confidence": 0.87,
  "percentage": 87.0,
  "max_confidence": 0.91,
  "max_percentage": 91.0,
  "hit_count": 3,
  "first_seen": "2026-09-07T10:15:32Z",
  "last_seen": "2026-09-07T10:16:04Z",
  "source": "webcam" | "video" | "image",
  "image_path": "images/a3f1c2b9e7d1.jpg",
  "synced": false
}

Deduplication: detections whose GPS coordinates fall inside the same
~40-50m grid cell are folded into ONE alert instead of creating a new
entry per frame. hit_count / confidence are updated in place, and the
saved evidence image is swapped in only when a higher-confidence
detection comes in for that same zone — so you always keep the
clearest shot, not just the first one.

Location: this is a CLI/video pipeline, not a browser, so there's no
built-in GPS. Pass --lat/--lon on the command line for accurate
coordinates. If omitted, get_ip_location() makes a best-effort
IP-based lookup (needs internet + the `requests` package; both
optional — this degrades gracefully to "location_source": "unavailable"
if either is missing).

"synced" is there for the server hand-off: an uploader (see
alert_uploader.py) can walk every alert with "synced": false, POST it,
and flip it to true — so re-running the uploader is always safe and
never double-sends.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import cv2

ALERTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts")
IMAGES_DIR = os.path.join(ALERTS_DIR, "images")
ALERTS_FILE = os.path.join(ALERTS_DIR, "alerts.json")

GRID_SIZE = 0.0004  # ~40-50m per cell — same grouping radius used in the map app

os.makedirs(IMAGES_DIR, exist_ok=True)


def _grid_key(lat, lng):
    """Detections within the same grid cell are treated as the same pothole."""
    if lat is None or lng is None:
        return "unknown_location"
    glat = round(lat / GRID_SIZE) * GRID_SIZE
    glng = round(lng / GRID_SIZE) * GRID_SIZE
    return f"{glat:.5f}_{glng:.5f}"


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_ip_location(timeout=3):
    """
    Best-effort city-level location when --lat/--lon isn't passed.
    Requires internet + `pip install requests`. Fails silently and
    returns (None, None, "unavailable") if either is missing — the
    rest of the pipeline still works, alerts are just unlocated.
    """
    try:
        import requests
        resp = requests.get("http://ip-api.com/json/", timeout=timeout)
        data = resp.json()
        if data.get("status") == "success":
            return data["lat"], data["lon"], "ip_estimate"
    except Exception:
        pass
    return None, None, "unavailable"


class AlertStore:
    """In-memory alert store, backed by alerts/alerts.json."""

    def __init__(self):
        self.alerts = {}
        self._load()

    def _load(self):
        if os.path.exists(ALERTS_FILE):
            try:
                with open(ALERTS_FILE, "r") as f:
                    self.alerts = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.alerts = {}

    def save(self):
        with open(ALERTS_FILE, "w") as f:
            json.dump(self.alerts, f, indent=2)

    def record(self, detection, frame, lat, lng, location_source, source):
        """
        detection: one item from PotholeDetector.get_detections()
                   -> {"class_name", "confidence", "bbox"}
        frame:     the raw BGR frame this detection came from
        lat/lng:   coordinates for this detection, or None
        source:    "image" | "video" | "webcam"

        Returns the (created or updated) alert dict for this zone.
        """
        key = _grid_key(lat, lng)
        confidence = detection["confidence"]
        percentage = round(confidence * 100, 1)
        now = _now_iso()

        if key not in self.alerts:
            alert_id = uuid.uuid4().hex[:12]
            image_path = os.path.join(IMAGES_DIR, f"{alert_id}.jpg")
            cv2.imwrite(image_path, frame)

            self.alerts[key] = {
                "alert_id": alert_id,
                "class_name": detection["class_name"],
                "location": {"lat": lat, "lng": lng},
                "location_source": location_source,
                "confidence": confidence,
                "percentage": percentage,
                "max_confidence": confidence,
                "max_percentage": percentage,
                "hit_count": 1,
                "first_seen": now,
                "last_seen": now,
                "source": source,
                "image_path": os.path.relpath(image_path, ALERTS_DIR),
                "synced": False,
            }
        else:
            alert = self.alerts[key]
            n = alert["hit_count"] + 1
            # running average confidence across every hit folded into this zone
            alert["confidence"] = round((alert["confidence"] * alert["hit_count"] + confidence) / n, 3)
            alert["percentage"] = round(alert["confidence"] * 100, 1)
            alert["hit_count"] = n
            alert["last_seen"] = now
            alert["synced"] = False  # data changed -> needs re-sync

            if confidence > alert["max_confidence"]:
                alert["max_confidence"] = confidence
                alert["max_percentage"] = percentage
                # keep the clearest evidence shot for this pothole
                image_path = os.path.join(ALERTS_DIR, alert["image_path"])
                cv2.imwrite(image_path, frame)

        self.save()
        return self.alerts[key]

    def summary(self):
        total = len(self.alerts)
        unsynced = sum(1 for a in self.alerts.values() if not a["synced"])
        return f"{total} alert(s) tracked, {unsynced} pending sync -> {ALERTS_FILE}"
