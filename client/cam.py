import cv2
import numpy as np

from client import G1Client

HOST = "192.168.1.61"
PORT = 5055

RECORD = False
CAMERAS = ["wrist_camera", "head_camera"]      # add more names to show side by side
OUT_PATH = "cam.mp4"
FPS = 30

env = G1Client(HOST, PORT)
caps = [env.open_stream(camera=c) for c in CAMERAS]


def grab_combined():
    frames = []
    for cap in caps:
        ok, bgr = cap.read()
        if not ok:
            return None
        frames.append(bgr)
    h = frames[0].shape[0]
    frames = [
        f if f.shape[0] == h
        else cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        for f in frames
    ]
    return np.hstack(frames) if len(frames) > 1 else frames[0]


writer = None
try:
    while True:
        bgr = grab_combined()
        if bgr is None:
            continue

        if RECORD and writer is None:
            fh, fw = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (fw, fh))

        if writer is not None:
            writer.write(bgr)

        cv2.imshow("g1", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    if writer is not None:
        writer.release()
        print(f"saved {OUT_PATH}")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
