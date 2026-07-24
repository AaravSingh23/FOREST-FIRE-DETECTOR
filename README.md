# 🌲🔥 Forest Fire Detection System

A Python system that detects **fire and smoke** from live video or recorded footage, using computer vision techniques — no machine learning model or GPU required.

---

## How It Works

The system uses three detection strategies running every frame:

| Strategy | What it checks |
|---|---|
| **Fire color detection** | Pixels in red / orange / yellow HSV ranges |
| **Smoke detection** | Gray, low-saturation regions that move |
| **Flickering analysis** | Rapid frame-to-frame variance (fire flickers) |

These signals are combined into a single **alert level**:

| Level | Meaning |
|---|---|
| `NONE` | Nothing detected |
| `WATCH` | Smoke detected — possible early sign |
| `WARNING` | Fire confirmed |
| `DANGER` | Large fire + flickering confirmed |

---

## Project Structure

```
forest_fire_detection/
│
├── main.py               ← Entry point: run detection on webcam / video
├── fire_detector.py      ← Core detection engine (FireDetector class)
├── test_detector.py      ← Self-contained test with synthetic fire frames
├── requirements.txt      ← Python dependencies
│
└── utils/
    ├── video_stream.py   ← Reads webcam, video files, or images
    └── alert_system.py   ← Logs alerts, saves snapshots, prints warnings
```

---

## Setup

**1. Install Python 3.8+** (if not already installed)

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Run with webcam (default)
```bash
python main.py
```

### Run with a video file
```bash
python main.py --source path/to/video.mp4
```

### Analyze a single image
```bash
python main.py --source path/to/forest_image.jpg
```

### Headless / server mode (no display window)
```bash
python main.py --source video.mp4 --no-display
```

### Run the self-test (no camera needed)
```bash
python test_detector.py
```

---

## Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | `0`=webcam, or path to video/image |
| `--width` | `640` | Resize frames to this width |
| `--no-display` | off | Disable the live display window |
| `--output-dir` | `alerts/` | Where snapshots and logs are saved |
| `--cooldown` | `5` | Seconds between repeated alerts |

---

## Output

Every run creates an `alerts/` folder containing:

- **`detection_log.csv`** — timestamped record of every alert
- **`WARNING_*.jpg`** — snapshot images saved on WARNING/DANGER events

---

## Tuning the Detector

Open `fire_detector.py` and adjust these values at the top of `__init__`:

```python
# Make fire detection more/less sensitive
self.fire_min_area = 500   # Lower → detect smaller fires

# Make smoke detection more/less sensitive
self.smoke_min_area = 2000  # Lower → detect thinner smoke

# Flickering threshold (frames)
self.flicker_threshold = 8  # Lower → trigger faster
```

---

## Limitations

- **Color-based**: sunsets, red/orange clothing, and lights can trigger false positives.
- **No ML model**: for production wildfire monitoring, pair this with a deep learning classifier trained on labeled smoke/fire datasets (e.g. FLAME dataset).
- **Night vision**: requires an IR camera for nighttime detection.

---

## Possible Enhancements

- [ ] Add thermal camera support
- [ ] SMS/email notifications via Twilio or SMTP
- [ ] YOLO-based fire detection model for higher accuracy
- [ ] Multi-camera support (grid view)
- [ ] GPS tagging of detection events
- [ ] Web dashboard (Flask/FastAPI)
