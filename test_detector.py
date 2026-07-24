import cv2
import numpy as np
import time
from fire_detector import FireDetector

def make_forest_background(h=480, w=640):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = (34, 80, 30)
    rng = np.random.default_rng(42)
    for _ in range(30):
        x = rng.integers(0, w)
        y1 = rng.integers(0, h // 2)
        y2 = rng.integers(h // 2, h)
        cv2.line(bg, (x, y1), (x, y2), (20, 50, 15), rng.integers(3, 12))
    return bg


def draw_fire(frame, cx, cy, radius, intensity=1.0, frame_idx=0):
    flicker = 1.0 + 0.15 * np.sin(frame_idx * 0.8)
    r = int(radius * intensity * flicker)
    cv2.circle(frame, (cx, cy), r, (0, 100, 255), -1)
    cv2.circle(frame, (cx, cy), max(1, r // 2), (30, 200, 255), -1)
    cv2.circle(frame, (cx, cy), max(1, r // 4), (200, 240, 255), -1)


def draw_smoke(frame, cx, cy, radius):
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy - radius // 2),
                (radius, radius // 2), 0, 0, 360, (140, 140, 140), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def run_tests():
    detector = FireDetector()
    bg_master = make_forest_background()
    h, w = bg_master.shape[:2]

    tests = [
        {
            "name": "🌲 No fire (baseline forest)",
            "fire": False,
            "smoke": False,
        },
        {
            "name": "💨 Smoke only (early warning)",
            "fire": False,
            "smoke": True,
        },
        {
            "name": "🔥 Small fire + no smoke",
            "fire": True,
            "smoke": False,
            "fire_radius": 40,
        },
        {
            "name": "🔥💨 Large fire + smoke (DANGER)",
            "fire": True,
            "smoke": True,
            "fire_radius": 90,
        },
    ]

    print("\n" + "=" * 52)
    print("  FOREST FIRE DETECTOR — SYNTHETIC TEST SUITE")
    print("=" * 52)
    all_passed = True

    for i, test in enumerate(tests):
        results = []
        for f_idx in range(15):
            frame = bg_master.copy()

            if test.get("smoke"):
                draw_smoke(frame, w // 2, h // 3, 130)

            if test.get("fire"):
                draw_fire(frame, w // 2, h // 2,
                          test.get("fire_radius", 60),
                          frame_idx=f_idx)

            result = detector.detect(frame)
            results.append(result)

        r = results[-1]
        expect_fire = test["fire"]
        expect_smoke = test["smoke"]
        ok_fire = r["fire_detected"] == expect_fire or not expect_fire
        ok_smoke = r["smoke_detected"] == expect_smoke or not expect_smoke
        passed = ok_fire and ok_smoke
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  Test {i+1}: {test['name']}")
        print(f"    Alert level  : {r['alert_level']}")
        print(f"    Fire detected: {r['fire_detected']}  (expected: {expect_fire})")
        print(f"    Smoke detect : {r['smoke_detected']}  (expected: {expect_smoke})")
        print(f"    Fire coverage: {r['fire_percent']:.1f}%")
        print(f"    Result       : {status}")

    print("\n" + "=" * 52)
    print("  VISUAL DEMO  (press Q to stop, or wait 10s)")
    print("=" * 52)

    detector = FireDetector()
    start = time.time()
    f_idx = 0

    while True:
        elapsed = time.time() - start
        phase = int(elapsed / 2.5) % 4

        frame = bg_master.copy()

        if phase == 1 or phase == 3:
            draw_smoke(frame, w // 2, h // 3, 130)
        if phase == 2 or phase == 3:
            draw_fire(frame, w // 2, h // 2, 80, frame_idx=f_idx)

        result = detector.detect(frame)
        display = result["annotated_frame"]

        phase_labels = {0: "BASELINE", 1: "SMOKE ONLY", 2: "FIRE ONLY", 3: "FIRE + SMOKE"}
        cv2.putText(display, f"Phase: {phase_labels[phase]}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Forest Fire Detector — Demo", display)
        f_idx += 1

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or elapsed > 10:
            break

    cv2.destroyAllWindows()

    print(f"\n  Overall result: {'✅ All tests passed!' if all_passed else '❌ Some tests failed.'}\n")
    return all_passed


if __name__ == "__main__":
    run_tests()