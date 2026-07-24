import os
import csv
import time
import datetime
import cv2

class AlertSystem:

    COLORS = {
        "NONE":    "\033[92m",
        "WATCH":   "\033[93m",
        "WARNING": "\033[91m",
        "DANGER":  "\033[95m",
        "RESET":   "\033[0m",
    }

    def __init__(self, output_dir="alerts", cooldown_seconds=5):
        self.output_dir      = output_dir
        self.cooldown        = cooldown_seconds
        self.last_alert_time = {}
        self.log_file        = os.path.join(output_dir, "detection_log.csv")

        os.makedirs(output_dir, exist_ok=True)
        self._init_log()

    def process(self, result, frame):
        level = result["alert_level"]
        now   = time.time()

        if level == "NONE":
            return

        last = self.last_alert_time.get(level, 0)
        if now - last < self.cooldown:
            return

        self.last_alert_time[level] = now
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._print_alert(level, timestamp, result)

        snapshot_path = ""
        if level in ("WARNING", "DANGER"):
            snapshot_path = self._save_snapshot(frame, level, timestamp)

        self._log_event(timestamp, level, result, snapshot_path)

    def print_summary(self, total_frames, detections):
        print("\n" + "="*50)
        print("  DETECTION SUMMARY")
        print("="*50)
        print(f"  Total frames processed : {total_frames}")
        print(f"  Fire detections        : {detections.get('fire', 0)}")
        print(f"  Smoke detections       : {detections.get('smoke', 0)}")
        print(f"  DANGER alerts          : {detections.get('DANGER', 0)}")
        print(f"  WARNING alerts         : {detections.get('WARNING', 0)}")
        print(f"  WATCH alerts           : {detections.get('WATCH', 0)}")
        print(f"  Log saved to           : {self.log_file}")
        print("="*50 + "\n")

    def _print_alert(self, level, timestamp, result):
        c = self.COLORS.get(level, "")
        r = self.COLORS["RESET"]

        icons = {"WATCH": "👁 ", "WARNING": "⚠️ ", "DANGER": "🔥"}
        icon  = icons.get(level, "")

        print(f"\n{c}{'─'*50}")
        print(f"  {icon} [{timestamp}]  ALERT LEVEL: {level}")
        if result["fire_detected"]:
            print(f"     Fire coverage : {result['fire_percent']:.1f}%")
        if result["smoke_detected"]:
            print(f"     Smoke coverage: {result['smoke_percent']:.1f}%")
        print(f"{'─'*50}{r}")

    def _save_snapshot(self, frame, level, timestamp):
        safe_ts  = timestamp.replace(":", "-").replace(" ", "_")
        filename = f"{level}_{safe_ts}.jpg"
        path     = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"  📸 Snapshot saved: {path}")
        return path

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "alert_level",
                    "fire_detected", "fire_percent",
                    "smoke_detected", "smoke_percent",
                    "snapshot_path"
                ])

    def _log_event(self, timestamp, level, result, snapshot_path):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                level,
                result["fire_detected"],
                result["fire_percent"],
                result["smoke_detected"],
                result["smoke_percent"],
                snapshot_path
            ])