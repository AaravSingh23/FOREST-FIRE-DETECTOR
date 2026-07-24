import cv2
import numpy as np

class FireDetector:
    def __init__(self):
        self.fire_lower_1 = np.array([0, 150, 150])
        self.fire_upper_1 = np.array([35, 255, 255])
        self.fire_lower_2 = np.array([160, 150, 150])
        self.fire_upper_2 = np.array([180, 255, 255])

        self.smoke_lower = np.array([0, 0, 80])
        self.smoke_upper = np.array([180, 50, 200])

        self.fire_min_area = 500
        self.smoke_min_area = 2000
        self.flicker_threshold = 8

        self.prev_frame = None
        self.fire_frame_count = 0
        self.motion_history = []

    def detect(self, frame):
        result = {
            "fire_detected": False,
            "smoke_detected": False,
            "alert_level": "NONE",
            "fire_boxes": [],
            "smoke_boxes": [],
            "fire_percent": 0.0,
            "smoke_percent": 0.0,
            "annotated_frame": frame.copy()
        }

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fire_mask = self._get_fire_mask(hsv)
        fire_boxes, fire_percent = self._find_regions(fire_mask, self.fire_min_area)
        if fire_boxes:
            result["fire_detected"] = True
            result["fire_boxes"] = fire_boxes
            result["fire_percent"] = fire_percent

        smoke_mask = self._get_smoke_mask(hsv, frame)
        smoke_boxes, smoke_percent = self._find_regions(smoke_mask, self.smoke_min_area)
        if smoke_boxes:
            result["smoke_detected"] = True
            result["smoke_boxes"] = smoke_boxes
            result["smoke_percent"] = smoke_percent

        flickering = self._check_flickering(frame)

        result["alert_level"] = self._get_alert_level(
            result["fire_detected"],
            result["smoke_detected"],
            result["fire_percent"],
            flickering
        )

        result["annotated_frame"] = self._draw_results(frame.copy(), result, flickering)

        return result

    def _get_fire_mask(self, hsv):
        mask1 = cv2.inRange(hsv, self.fire_lower_1, self.fire_upper_1)
        mask2 = cv2.inRange(hsv, self.fire_lower_2, self.fire_upper_2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _get_smoke_mask(self, hsv, bgr_frame):
        smoke_color = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
        blurred = cv2.GaussianBlur(smoke_color, (21, 21), 0)
        _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _check_flickering(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        diff = cv2.absdiff(gray, self.prev_frame)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh > 0)

        self.prev_frame = gray
        self.motion_history.append(motion_pixels)

        if len(self.motion_history) > 15:
            self.motion_history.pop(0)

        if len(self.motion_history) >= 5:
            variance = np.var(self.motion_history)
            avg_motion = np.mean(self.motion_history)
            return variance > 500000 and avg_motion > 300
        return False

    def _find_regions(self, mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        total_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
                total_area += area

        frame_area = mask.shape[0] * mask.shape[1]
        percent = (total_area / frame_area) * 100 if frame_area > 0 else 0.0
        return boxes, round(percent, 2)

    def _get_alert_level(self, fire, smoke, fire_pct, flickering):
        if fire and fire_pct > 5 and flickering:
            return "DANGER"
        elif fire:
            return "WARNING"
        elif smoke:
            return "WATCH"
        return "NONE"

    def _draw_results(self, frame, result, flickering):
        alert_colors = {
            "NONE": (0, 200, 0),
            "WATCH": (0, 200, 200),
            "WARNING": (0, 100, 255),
            "DANGER": (0, 0, 255),
        }
        level = result["alert_level"]
        color = alert_colors[level]

        for (x, y, w, h) in result["fire_boxes"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"FIRE {result['fire_percent']:.1f}%",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for (x, y, w, h) in result["smoke_boxes"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)
            cv2.putText(frame, f"SMOKE {result['smoke_percent']:.1f}%",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        banner_text = f"  ALERT: {level}"
        if flickering:
            banner_text += "  |  FLICKERING DETECTED"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), color, -1)
        cv2.putText(frame, banner_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        h = frame.shape[0]
        cv2.putText(frame,
                    f"Fire: {'YES' if result['fire_detected'] else 'NO'}  "
                    f"Smoke: {'YES' if result['smoke_detected'] else 'NO'}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return frame