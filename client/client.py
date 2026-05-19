import io
import requests


class G1Client:
    def __init__(self, host: str, port: int = 5055, timeout: float = 2.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # ---- internals ----
    def _get(self, path, **params):
        r = requests.get(self.base + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body=None):
        r = requests.post(self.base + path, json=body or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- robot state ----
    def get_state(self) -> dict:
        """Latest robot observation: 'q' (joint positions, rad), 'dq' (velocities),
        'floating_base_pose' (xyz + quat), 'torso_quat', etc."""
        return self._get("/obs")

    def get_command(self) -> dict:
        """Currently commanded values (vx, vy, vyaw, height, active, ...)."""
        return self._get("/state")

    def get_joint_names(self) -> list:
        """Ordered joint names; indices align with state['q'] / state['dq']. Cached."""
        if not hasattr(self, "_joint_names_cache"):
            self._joint_names_cache = self._get("/joint_names").get("joint_names", [])
        return self._joint_names_cache

    def get_joints_dict(self) -> dict:
        """Convenience: {joint_name: position_rad} from the latest obs."""
        names = self.get_joint_names()
        q = self.get_state().get("q", [])
        return dict(zip(names, q))

    # ---- upper body command ----
    def get_upper_body_names(self) -> list:
        """Ordered upper-body joint names. Cached."""
        if not hasattr(self, "_ub_names_cache"):
            self._ub_names_cache = self._get("/upper_body_names").get("joint_names", [])
        return self._ub_names_cache

    def set_upper_body(self, target=None, joints: dict | None = None,
                       duration: float = 1.0) -> dict:
        """Command upper-body joint positions (rad) with smooth interpolation.

        Pass either:
          target   -- full vector matching get_upper_body_names() length, OR
          joints   -- dict {joint_name: position_rad} for partial updates;
                      unspecified joints hold their current position.
        duration -- seconds the policy gets to interpolate to the target."""
        body = {"duration": float(duration)}
        if target is not None:
            body["target"] = list(target)
        elif joints is not None:
            body["joints"] = {k: float(v) for k, v in joints.items()}
        else:
            raise ValueError("must pass target=[...] or joints={name: value, ...}")
        return self._post("/upper_body", body)

    # ---- gripper (Unitree Dex1-1) ----
    def get_gripper_state(self) -> dict:
        """Latest gripper state per side, e.g.
            {'right': {'q': 0.5, 'dq': 0.0, 'tau': 0.0, 'ts': ...}, ...}"""
        return self._get("/gripper")

    def set_gripper(self, q: float, side: str = "right") -> dict:
        """Command a single gripper to position q (rad).
        side: 'right' | 'left' | 'both'.
        Typical range ~0 (closed) to ~5.5 (open) — check by calibrating once."""
        return self._post("/gripper", {"side": side, "q": float(q)})

    def open_gripper(self, side: str = "right", q: float = 5.0) -> dict:
        return self.set_gripper(q, side)

    def close_gripper(self, side: str = "right", q: float = 0.5) -> dict:
        return self.set_gripper(q, side)

    # ---- camera ----
    def list_cameras(self) -> list:
        return self._get("/cameras").get("cameras", [])

    def get_rgb(self, camera: str | None = None):
        """RGB frame as numpy uint8 array (H, W, 3). Raises if no image is available.
        Adds full HTTP round-trip latency per call — use open_stream() for live video."""
        params = {"camera": camera} if camera else {}
        r = requests.get(self.base + "/rgb", params=params, timeout=self.timeout)
        if r.status_code == 503:
            raise RuntimeError(f"no image: {r.json()}")
        r.raise_for_status()
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))

    def open_stream(self, camera: str | None = None, open_timeout: float = 10.0):
        """Open the persistent MJPEG stream and return a drop-in cv2.VideoCapture
        replacement: `.read()` returns `(ok, bgr_ndarray)`, `.release()` tears down.
        Pass camera='head_camera' or 'wrist_camera' to pick one (omit for first).

        Unlike cv2.VideoCapture over HTTP, this drains the socket in a background
        thread and always returns the LATEST decoded frame, so latency never
        accumulates over time."""
        url = f"{self.base}/stream.mjpg"
        if camera:
            url += f"?camera={camera}"
        return _LatestMjpegStream(url, open_timeout)

    # ---- policy control ----
    def start_policy(self) -> dict:
        return self._post("/cmd", {"active": True})

    def stop_policy(self) -> dict:
        return self._post("/cmd", {"active": False})

    def step(self, cmd: dict | None = None, **kwargs) -> dict:
        """Absolute setpoints. keys: vx, vy, vyaw, height, freq, roll, pitch, yaw, active.
        roll/pitch/yaw in DEGREES."""
        body = dict(cmd or {})
        body.update(kwargs)
        return self._post("/cmd", body)

    def emergency_stop(self) -> dict:
        """Zero all linear/angular cmd AND deactivate the policy."""
        return self._post("/stop")


class _LatestMjpegStream:
    """Background-thread MJPEG drainer; keeps only the most recent frame.
    Drop-in shape compatible with cv2.VideoCapture: .read() / .isOpened()
    / .release(). Avoids OpenCV's HTTP buffer that causes lag to grow over
    time when the producer is faster than the consumer."""
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, url: str, open_timeout: float = 10.0):
        import threading
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._opened = threading.Event()
        self._err = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._opened.wait(timeout=open_timeout):
            raise RuntimeError(f"timeout opening stream {url}")
        if self._err is not None:
            raise RuntimeError(f"failed to open stream {url}: {self._err}")

    def _run(self):
        import urllib.request
        import cv2
        import numpy as np
        try:
            r = urllib.request.urlopen(self.url, timeout=5.0)
        except Exception as e:
            self._err = e
            self._opened.set()
            return
        self._opened.set()
        buf = b""
        try:
            while not self._stop:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
                last_eoi = buf.rfind(self._EOI)
                if last_eoi >= 0:
                    last_soi = buf.rfind(self._SOI, 0, last_eoi)
                    if last_soi >= 0:
                        jpeg = buf[last_soi : last_eoi + 2]
                        buf = buf[last_eoi + 2 :]
                        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                           cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
                if len(buf) > 5_000_000:
                    buf = buf[-1_000_000:]
        finally:
            try:
                r.close()
            except Exception:
                pass

    def read(self):
        with self._lock:
            f = self._frame
        return (f is not None), f

    def isOpened(self) -> bool:
        return self._opened.is_set() and self._err is None

    def release(self):
        self._stop = True


