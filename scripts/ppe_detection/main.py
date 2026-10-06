"""RealSense D435i + YOLO: per-person hardhat / safety vest check."""
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

HARDHAT, VEST = "Hardhat", "Safety Vest"
# Fractions of person box height where each item must sit (top, bottom).
HEAD_REGION = (0.0, 0.35)
TORSO_REGION = (0.2, 0.75)
OVERLAP_MIN = 0.5  # fraction of the PPE box that must lie inside the region
MODELS = Path(__file__).parent / "models"


def inside_ratio(box, region):
    """Fraction of `box` area that lies inside `region` (both x1,y1,x2,y2)."""
    ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def band(person, frac):
    x1, y1, x2, y2 = person
    h = y2 - y1
    return (x1, y1 + frac[0] * h, x2, y1 + frac[1] * h)


def wears(person, items, frac):
    region = band(person, frac)
    return any(inside_ratio(b, region) >= OVERLAP_MIN for b in items)


def person_distance(depth, person, scale):
    """Median depth (m) of the central torso patch; None if no valid pixels."""
    x1, y1, x2, y2 = map(int, person)
    cx, cy = (x1 + x2) // 2, y1 + (y2 - y1) // 3
    w, h = max(2, (x2 - x1) // 6), max(2, (y2 - y1) // 8)
    patch = depth[max(0, cy - h):cy + h, max(0, cx - w):cx + w]
    valid = patch[patch > 0]
    return float(np.median(valid)) * scale if valid.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "best.pt"), help="PPE model")
    ap.add_argument("--person-model", default=str(MODELS / "yolo11s.pt"), help="COCO model for person + tracking")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-dist", type=float, default=0.5, help="m")
    ap.add_argument("--max-dist", type=float, default=5.0, help="m")
    ap.add_argument("--smooth", type=int, default=15, help="frames for majority vote")
    args = ap.parse_args()

    # The PPE model barely emits its "Person" class, so a COCO model finds/tracks people.
    model = YOLO(args.model)
    person_model = YOLO(args.person_model)
    names = model.names

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    history = defaultdict(lambda: (deque(maxlen=args.smooth), deque(maxlen=args.smooth)))
    last_print = 0.0
    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            img = np.asanyarray(color.get_data())
            dep = np.asanyarray(depth.get_data())

            ppe = model.predict(img, conf=args.conf, verbose=False)[0].boxes
            cls = [names[int(c)] for c in ppe.cls.cpu().numpy()]
            hats = [b for b, c in zip(ppe.xyxy.cpu().numpy(), cls) if c == HARDHAT]
            vests = [b for b, c in zip(ppe.xyxy.cpu().numpy(), cls) if c == VEST]
            for b in hats + vests:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (255, 200, 0), 1)

            people = person_model.track(img, persist=True, classes=[0], conf=args.conf,
                                        tracker="bytetrack.yaml", verbose=False)[0].boxes
            if people.id is None:
                people_iter = []
            else:
                people_iter = zip(people.xyxy.cpu().numpy(), people.id.int().cpu().tolist())

            lines = []
            for b, tid in people_iter:
                dist = person_distance(dep, b, scale)
                p1, p2 = tuple(map(int, b[:2])), tuple(map(int, b[2:]))
                if dist is None or not args.min_dist <= dist <= args.max_dist:
                    cv2.rectangle(img, p1, p2, (128, 128, 128), 1)
                    continue
                hh, vh = history[tid]
                hh.append(wears(b, hats, HEAD_REGION))
                vh.append(wears(b, vests, TORSO_REGION))
                hat_ok, vest_ok = sum(hh) * 2 > len(hh), sum(vh) * 2 > len(vh)

                color_ = (0, 200, 0) if hat_ok and vest_ok else (0, 0, 255)
                cv2.rectangle(img, p1, p2, color_, 2)
                label = f"#{tid} {dist:.1f}m H:{'OK' if hat_ok else 'NO'} V:{'OK' if vest_ok else 'NO'}"
                cv2.putText(img, label, (p1[0], max(15, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_, 2)
                lines.append(f"person {tid} ({dist:.1f}m) : 안전모 {'OK' if hat_ok else 'NO'}  안전복 {'OK' if vest_ok else 'NO'}")

            now = time.time()
            if lines and now - last_print >= 1.0:
                print("\n".join(lines), end="\n\n", flush=True)
                last_print = now

            cv2.imshow("PPE check (q: quit)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
