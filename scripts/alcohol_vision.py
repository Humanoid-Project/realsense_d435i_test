import argparse
import collections
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REF_BAC = 0.071

STUDY = {
    "mtf_cutoff_cpd": (41.13, 35.34, "Casares-Lopez et al. 2021, IJERPH 18:6790, Table 2 (MTF cut-off, 4 mm pupil, BrAC 0.34 mg/L)"),
    "contrast_sensitivity": (149.8, 130.1, "Casares-Lopez et al. 2021, IJERPH 18:6790, Table 1 (binocular CS)"),
    "straylight_log_s": (0.87, 0.97, "Casares-Lopez et al. 2020, Sci Rep 10:13599 (doi 10.1038/s41598-020-70645-3), high BrAC group 0.36 mg/L"),
    "halo_vdi": (0.14, 0.19, "Casares-Lopez et al. 2021, IJERPH 18:6790, Table 1 (binocular visual disturbance index)"),
    "reaction_time_s": (0.89, 1.04, "Casares-Lopez et al. 2020, Sci Rep, Table 3 (braking reaction time, high BrAC group)"),
}

PRESETS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]

FIELD_ONSET_BAC = 0.05
FIELD_FULL_BAC = 0.15
FIELD_MIN_RATIO = 0.6
DOUBLE_ONSET_BAC = 0.08
DOUBLE_MAX_FRACTION = 0.012
SWAY_MAX_FRACTION = 0.008
VEIL_BASE = 0.35
HALO_BASE = 0.6
NYQUIST_SIGMA = math.sqrt(math.log(10.0) / (2.0 * math.pi ** 2)) / 0.5


@dataclass
class Effects:
    bac: float
    blur_sigma: float
    contrast: float
    veil: float
    halo: float
    field_ratio: float
    delay_s: float
    double_px: float
    sway_px: float


def ramp(bac, onset, full):
    return float(np.clip((bac - onset) / (full - onset), 0.0, 1.0))


def effects_for(bac, gain, width):
    s = max(bac, 0.0) / REF_BAC * gain
    mtf0, mtf1, _ = STUDY["mtf_cutoff_cpd"]
    cutoff_ratio = max(1.0 - (1.0 - mtf1 / mtf0) * s, 0.15)
    blur_sigma = NYQUIST_SIGMA * math.sqrt(1.0 / cutoff_ratio ** 2 - 1.0) * width / 1280.0
    cs0, cs1, _ = STUDY["contrast_sensitivity"]
    contrast = max(1.0 - (1.0 - cs1 / cs0) * s, 0.2)
    ls0, ls1, _ = STUDY["straylight_log_s"]
    veil = VEIL_BASE * (10.0 ** ((ls1 - ls0) * s) - 1.0)
    v0, v1, _ = STUDY["halo_vdi"]
    halo = HALO_BASE * (v1 / v0 - 1.0) * s
    rt0, rt1, _ = STUDY["reaction_time_s"]
    delay_s = (rt1 - rt0) * s
    field_ratio = 1.0 - (1.0 - FIELD_MIN_RATIO) * ramp(bac, FIELD_ONSET_BAC, FIELD_FULL_BAC)
    late = ramp(bac, DOUBLE_ONSET_BAC, FIELD_FULL_BAC)
    return Effects(bac, blur_sigma, contrast, veil, halo, field_ratio, delay_s,
                   DOUBLE_MAX_FRACTION * width * late, SWAY_MAX_FRACTION * width * late)


class DelayLine:
    def __init__(self, max_s=2.0):
        self.frames = collections.deque()
        self.max_s = max_s

    def push_and_get(self, frame, now, delay_s):
        self.frames.append((now, frame))
        while self.frames and now - self.frames[0][0] > self.max_s:
            self.frames.popleft()
        target = now - delay_s
        chosen = self.frames[-1][1]
        for t, f in reversed(self.frames):
            chosen = f
            if t <= target:
                break
        return chosen


def field_mask(h, w, ratio):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2.0)
    inner = ratio * 0.75
    outer = inner + 0.35
    m = 1.0 - np.clip((r - inner) / (outer - inner), 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)
    m3 = cv2.merge([(m * 255.0).astype(np.uint8)] * 3)
    return m3, cv2.bitwise_not(m3)


def blur_small(img, sigma_frac, div):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // div, h // div), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma_frac * w / div)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_effects(frame, fx, t, mask_cache):
    h, w = frame.shape[:2]
    img = frame

    if fx.sway_px > 0.0:
        dx = fx.sway_px * math.sin(2.0 * math.pi * 0.35 * t) + 0.3 * fx.sway_px * math.sin(2.0 * math.pi * 1.7 * t)
        dy = 0.4 * fx.sway_px * math.sin(2.0 * math.pi * 0.23 * t + 1.0)
        img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)

    if fx.double_px > 0.5:
        m = np.float32([[1, 0, fx.double_px], [0, 1, 0.15 * fx.double_px]])
        ghost = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = cv2.addWeighted(img, 0.62, ghost, 0.38, 0.0)

    if fx.blur_sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), fx.blur_sigma)

    if fx.halo > 0.0:
        lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.multiply(cv2.subtract(lum, 191), 4)
        bright = cv2.multiply(img, cv2.merge([bright_mask] * 3), scale=1.0 / 255.0)
        img = cv2.addWeighted(img, 1.0, blur_small(bright, 0.02, 4), fx.halo, 0.0)

    if fx.veil > 0.0:
        img = cv2.addWeighted(img, 1.0 - fx.veil, blur_small(img, 1.0 / 64.0, 8), fx.veil, 0.0)

    if fx.contrast < 1.0:
        mean = np.full_like(img, np.uint8(np.clip(cv2.mean(img)[:3], 0, 255)))
        img = cv2.addWeighted(img, fx.contrast, mean, 1.0 - fx.contrast, 0.0)

    if fx.field_ratio < 0.999:
        key = (h, w, round(fx.field_ratio, 3))
        if key not in mask_cache:
            mask_cache.clear()
            mask_cache[key] = field_mask(h, w, fx.field_ratio)
        m3, inv3 = mask_cache[key]
        periphery = cv2.convertScaleAbs(blur_small(img, 0.01, 4), alpha=0.25)
        img = cv2.add(cv2.multiply(img, m3, scale=1.0 / 255.0), cv2.multiply(periphery, inv3, scale=1.0 / 255.0))

    return img


def put_label(img, lines, origin=(16, 34)):
    x, y = origin
    for text in lines:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32


def compose(normal, drunk, fx, gain, view, show_info):
    left = normal.copy()
    right = drunk.copy()
    put_label(left, ["NORMAL  BAC 0.00%"])
    lines = [f"DRUNK  BAC {fx.bac:.2f}%"]
    if show_info:
        lines += [
            f"blur {fx.blur_sigma:.2f}px  contrast x{fx.contrast:.2f}",
            f"glare {fx.veil:.3f}  halo {fx.halo:.2f}",
            f"field x{fx.field_ratio:.2f}  delay {fx.delay_s * 1000:.0f}ms",
            f"gain x{gain:.1f}",
        ]
    put_label(right, lines)
    if view == "drunk":
        return right
    return np.hstack([left, right])


class RealSenseSource:
    def __init__(self, width, height, fps):
        import pyrealsense2 as rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)

    def read(self):
        frames = self.pipe.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data()).copy()

    def close(self):
        self.pipe.stop()


class FileSource:
    def __init__(self, path):
        self.image = cv2.imread(str(path)) if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} else None
        self.cap = None if self.image is not None else cv2.VideoCapture(str(path))

    def read(self):
        if self.image is not None:
            return self.image.copy()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


def print_sources():
    for key, (before, after, src) in STUDY.items():
        print(f"{key:22s} {before:>7} -> {after:<7} at BAC ~{REF_BAC:.3f}%  {src}")
    print(f"visual field narrowing: illustrative, {FIELD_ONSET_BAC}% -> {FIELD_FULL_BAC}% shrinks field to x{FIELD_MIN_RATIO}")
    print(f"double vision / sway: illustrative, from {DOUBLE_ONSET_BAC}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bac", type=float, default=0.08)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--view", choices=["side", "drunk"], default="side")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-info", action="store_true")
    ap.add_argument("--sources", action="store_true")
    args = ap.parse_args()

    if args.sources:
        print_sources()
        return

    src = FileSource(args.input) if args.input else RealSenseSource(args.width, args.height, args.fps)
    out_dir = Path(__file__).resolve().parent.parent / "captures"
    bac, gain, view, show_info = args.bac, args.gain, args.view, not args.no_info
    delay = DelayLine()
    mask_cache = {}
    writer = None
    recording = args.record is not None
    record_path = args.record
    t0 = time.monotonic()
    shown = 0

    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            now = time.monotonic()
            h, w = frame.shape[:2]
            fx = effects_for(bac, gain, w)
            delayed = delay.push_and_get(frame, now, fx.delay_s)
            drunk = apply_effects(delayed, fx, now - t0, mask_cache)
            canvas = compose(frame, drunk, fx, gain, view, show_info)

            if recording:
                if writer is None or writer.get("shape") != canvas.shape:
                    if writer is not None:
                        writer["w"].release()
                    if record_path is None:
                        out_dir.mkdir(exist_ok=True)
                        record_path = str(out_dir / time.strftime("alcohol_%Y%m%d_%H%M%S.mp4"))
                    vw = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (canvas.shape[1], canvas.shape[0]))
                    writer = {"w": vw, "shape": canvas.shape}
                    print(f"recording -> {record_path}")
                writer["w"].write(canvas)

            if not args.no_window:
                disp = canvas
                if disp.shape[1] > 1920:
                    scale = 1920 / disp.shape[1]
                    disp = cv2.resize(disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("alcohol vision", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if ord("0") <= key <= ord("5"):
                    bac = PRESETS[key - ord("0")]
                elif key in (ord("+"), ord("=")):
                    bac = round(min(bac + 0.01, 0.30), 3)
                elif key in (ord("-"), ord("_")):
                    bac = round(max(bac - 0.01, 0.0), 3)
                elif key == ord("]"):
                    gain = min(gain + 0.5, 10.0)
                elif key == ord("["):
                    gain = max(gain - 0.5, 0.5)
                elif key == ord("v"):
                    view = "drunk" if view == "side" else "side"
                elif key == ord("i"):
                    show_info = not show_info
                elif key == ord("s"):
                    out_dir.mkdir(exist_ok=True)
                    p = out_dir / time.strftime(f"alcohol_bac{bac:.2f}_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(str(p), canvas)
                    print(f"saved {p}")
                elif key == ord("r"):
                    recording = not recording
                    if not recording and writer is not None:
                        writer["w"].release()
                        print(f"saved {record_path}")
                        writer, record_path = None, None

            shown += 1
            if args.duration and now - t0 >= args.duration:
                break
    finally:
        if writer is not None:
            writer["w"].release()
            print(f"saved {record_path}")
        src.close()
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t0
        if elapsed > 0:
            print(f"{shown} frames, {shown / elapsed:.1f} fps")


if __name__ == "__main__":
    main()
