import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=str)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save-width", type=int, default=1280)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.accel)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(args.warmup):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        accel = [f.as_motion_frame().get_motion_data() for f in frames if f.is_motion_frame()]
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = frames.get_depth_frame()
    finally:
        pipe.stop()

    scale = args.save_width / color.shape[1]
    img = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else color
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = color.shape[:2]
    center = depth.get_distance(w // 2, h // 2)
    pitch = None
    if accel:
        a = accel[0]
        pitch = math.degrees(math.atan2(-a.z, -a.y))
    print(f"saved {args.out}  center_depth_m {center:.2f}  camera_down_deg {pitch if pitch is None else round(pitch, 1)}")


if __name__ == "__main__":
    main()
