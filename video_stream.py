import cv2
import os

class VideoStream:

    def __init__(self, source=0, resize_width=None):
        self.source       = source
        self.resize_width = resize_width
        self.is_image     = False
        self.cap          = None
        self._image_frame = None

        self._open()

    def _open(self):
        if isinstance(self.source, str):
            ext = os.path.splitext(self.source)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                self.is_image     = True
                self._image_frame = cv2.imread(self.source)
                if self._image_frame is None:
                    raise FileNotFoundError(f"Could not read image: {self.source}")
                print(f"[VideoStream] Opened image: {self.source}")
                return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VideoStream] Cannot open source: {self.source}")

        label = "webcam" if self.source == 0 else f"video: {self.source}"
        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VideoStream] Opened {label} | FPS: {fps:.1f} | Frames: {total or 'live'}")

    def read(self):
        if self.is_image:
            frame = self._image_frame.copy()
        else:
            ok, frame = self.cap.read()
            if not ok:
                return False, None

        if self.resize_width and frame is not None:
            frame = self._resize(frame)

        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()

    def get_fps(self):
        if self.is_image or self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        if self.is_image:
            return 1
        if self.cap is None:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _resize(self, frame):
        h, w       = frame.shape[:2]
        ratio      = self.resize_width / w
        new_height = int(h * ratio)
        return cv2.resize(frame, (self.resize_width, new_height))

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.read()
        if not ok:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()