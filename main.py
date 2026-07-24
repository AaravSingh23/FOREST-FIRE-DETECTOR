import argparse
import time
import cv2
from collections import defaultdict
from fire_detector import FireDetector
from utils.video_stream import VideoStream
from utils.alert_system import AlertSystem

def parse_args():
    parser = argparse.ArgumentParser(
        description="Forest Fire & Smoke Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source", default="0",
        help=(
            "Video source:\n"
            "  0           → default webcam\n"
            "  1, 2, ...   → other camera indices\n"
            "  path/to/video.mp4   → video file\n"
            "  path/to/image.jpg   → single image"
        )
    )
    parser.add_argument("--width", type=int, default=640,
                        help="Resize frames to this width (default: 640)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without opening a display window (headless mode)")
    parser.add_argument("--output-dir", default="alerts",
                        help="Directory for snapshots and logs (default: alerts/)")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Seconds between repeated alerts (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    print("\n" + "=" * 55)
    print("  🌲 FOREST FIRE DETECTION SYSTEM  🌲")
    print("=" * 55)
    print(f"  Source     : {source}")
    print(f"  Frame width: {args.width}px")
    print(f"  Output dir : {args.output_dir}/")
    print(f"  Display    : {'OFF (headless)' if args.no_display else 'ON'}")
    print("=" * 55)
    print("  Press  Q  to quit\n")

    detector = FireDetector()
    alert_system = AlertSystem(output_dir=args.output_dir, cooldown_seconds=args.cooldown)
    stream = VideoStream(source=source, resize_width=args.width)

    stats = defaultdict(int)
    frame_count = 0
    start_time = time.time()

    try:
        for frame in stream:
            frame_count += 1

            result = detector.detect(frame)

            if result["fire_detected"]:
                stats["fire"] += 1
            if result["smoke_detected"]:
                stats["smoke"] += 1
            if result["alert_level"] != "NONE":
                stats[result["alert_level"]] += 1

            alert_system.process(result, result["annotated_frame"])

            if not args.no_display:
                annotated = result["annotated_frame"]

                elapsed = time.time() - start_time
                fps_live = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated, f"FPS: {fps_live:.1f}",
                            (annotated.shape[1] - 120, annotated.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Forest Fire Detection  |  Press Q to quit", annotated)

                if stream.is_image:
                    print("  [Image mode] Press any key to exit.")
                    cv2.waitKey(0)
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  [INFO] User pressed Q — stopping.")
                    break

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted by user.")

    finally:
        stream.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n  Processed {frame_count} frames in {elapsed:.1f}s")
        alert_system.print_summary(frame_count, stats)


if __name__ == "__main__":
    main()