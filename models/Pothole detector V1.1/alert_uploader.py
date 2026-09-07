"""
alert_uploader.py
------------------
Walks alerts/alerts.json and POSTs every alert marked "synced": false
to a server, then flips it to true on success. Safe to re-run — already
synced alerts are skipped, and a failed upload just gets retried next run.

Usage:
    python alert_uploader.py --server-url https://your-server.com/api/alerts
    python alert_uploader.py --server-url https://your-server.com/api/alerts --dry-run

Expects the server to accept a multipart/form-data POST with:
    - all alert fields as form fields (alert_id, class_name, confidence,
      percentage, hit_count, first_seen, last_seen, source, lat, lng,
      location_source)
    - the evidence image as a file field named "image"
Adjust `upload_one()` below to match your actual server's API shape.

Requires: pip install requests
"""

import argparse
import json
import os

import requests

from alert_logger import ALERTS_DIR, ALERTS_FILE


def load_alerts():
    if not os.path.exists(ALERTS_FILE):
        return {}
    with open(ALERTS_FILE, "r") as f:
        return json.load(f)


def save_alerts(alerts):
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


def upload_one(server_url, key, alert, dry_run=False):
    image_path = os.path.join(ALERTS_DIR, alert["image_path"])

    payload = {
        "alert_id": alert["alert_id"],
        "class_name": alert["class_name"],
        "lat": alert["location"]["lat"],
        "lng": alert["location"]["lng"],
        "location_source": alert["location_source"],
        "confidence": alert["confidence"],
        "percentage": alert["percentage"],
        "max_confidence": alert["max_confidence"],
        "hit_count": alert["hit_count"],
        "first_seen": alert["first_seen"],
        "last_seen": alert["last_seen"],
        "source": alert["source"],
    }

    if dry_run:
        print(f"[dry-run] would POST {alert['alert_id']} ({alert['percentage']}%) -> {server_url}")
        return True

    try:
        with open(image_path, "rb") as img_file:
            files = {"image": (os.path.basename(image_path), img_file, "image/jpeg")}
            resp = requests.post(server_url, data=payload, files=files, timeout=15)
        if resp.status_code in (200, 201):
            print(f"Synced {alert['alert_id']} ({alert['percentage']}%) -> {server_url}")
            return True
        print(f"Server rejected {alert['alert_id']}: HTTP {resp.status_code} — {resp.text[:200]}")
        return False
    except FileNotFoundError:
        print(f"Skipping {alert['alert_id']}: image not found at {image_path}")
        return False
    except requests.RequestException as e:
        print(f"Upload failed for {alert['alert_id']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sync unsynced pothole alerts to a server.")
    parser.add_argument("--server-url", required=True, help="Endpoint to POST alerts to")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent, don't actually POST")
    args = parser.parse_args()

    alerts = load_alerts()
    pending = {k: a for k, a in alerts.items() if not a.get("synced")}

    if not pending:
        print("Nothing to sync — all alerts are already marked synced.")
        return

    print(f"{len(pending)} alert(s) pending sync...")
    for key, alert in pending.items():
        success = upload_one(args.server_url, key, alert, dry_run=args.dry_run)
        if success and not args.dry_run:
            alerts[key]["synced"] = True
            save_alerts(alerts)  # save after each success so a crash mid-run doesn't lose progress

    remaining = sum(1 for a in alerts.values() if not a.get("synced"))
    print(f"Done. {remaining} alert(s) still unsynced.")


if __name__ == "__main__":
    main()
