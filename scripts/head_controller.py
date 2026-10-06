import argparse
import json
import math
import os
import signal
import sys
import time

import can
from robonex_common.can import FeedbackHub, Motor
from robonex_common.joints import MOTOR_BY_ID, channel_for_motor_id
from robonex_common.motors import MOTOR_SPECS
from robonex_common.protocol import DEFAULT_INTERFACE, HOST_ID

MOTOR_ID = 13
LOW_DEG = -70.0
HIGH_DEG = 30.0
SPEED = 0.3
KP = 20.0
KD = 1.0
RATE = 100.0
OVERSPEED_STOP = 2.0
FEEDBACK_TIMEOUT = 0.3
TRACKING_STOP = math.radians(20.0)
ARRIVE_TOL = math.radians(2.0)


class Stop(Exception):
    pass


def raise_stop(signum, frame):
    raise Stop(f"signal {signum}")


def write_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def read_command(path, last):
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None, last
    if not text or text == last:
        return None, last
    parts = text.split()
    return parts[1:] if len(parts) > 1 else None, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-file", default="/tmp/head_cmd")
    ap.add_argument("--state-file", default="/tmp/head_state.json")
    args = ap.parse_args()

    joint = MOTOR_BY_ID[MOTOR_ID]
    spec = MOTOR_SPECS[joint.motor_model]
    channel = channel_for_motor_id(MOTOR_ID)
    low, high = math.radians(LOW_DEG), math.radians(HIGH_DEG)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, raise_stop)

    try:
        last_cmd = open(args.cmd_file).read().strip()
    except OSError:
        last_cmd = None

    bus = can.Bus(channel=channel, interface=DEFAULT_INTERFACE)
    motor = Motor(bus, MOTOR_ID, spec, host_id=HOST_ID)
    hub = FeedbackHub(bus, [motor], HOST_ID)
    exit_reason = "finished"
    try:
        motor.write_run_mode_operation()
        time.sleep(0.005)
        motor.enable()
        start = hub.wait_for(MOTOR_ID, timeout=0.3)
        if start is None:
            raise Stop("no live feedback after enable")
        if abs(start) > math.pi:
            raise Stop(f"feedback {math.degrees(start):+.1f} deg outside -180..+180")
        print(f"enabled ID {MOTOR_ID} on {channel}, start {math.degrees(start):+.1f} deg, "
              f"range {LOW_DEG:+.0f}..{HIGH_DEG:+.0f} deg, speed {SPEED} rad/s, kp {KP} kd {KD}", flush=True)

        cmd = start
        target = start
        park_then_stop = False
        period = 1.0 / RATE
        next_state = 0.0
        next_poll = 0.0
        while True:
            now = time.monotonic()
            if now >= next_poll:
                next_poll = now + 0.05
                words, last_cmd = read_command(args.cmd_file, last_cmd)
                if words:
                    op = words[0]
                    if op == "goto" and len(words) == 2:
                        target = min(max(math.radians(float(words[1])), low), high)
                        print(f"goto {math.degrees(target):+.1f} deg", flush=True)
                    elif op == "park":
                        target = low
                        park_then_stop = True
                        print(f"park to {LOW_DEG:+.0f} deg then stop", flush=True)
                    elif op == "stop":
                        raise Stop("stop command")
                    else:
                        print(f"ignored command {words}", flush=True)

            step = SPEED * period
            delta = target - cmd
            vel = 0.0
            if abs(delta) > step:
                cmd += math.copysign(step, delta)
                vel = math.copysign(SPEED, delta)
            else:
                cmd = target
            motor.control(pos=cmd, vel=vel, kp=KP, kd=KD)
            hub.pump()

            if abs(motor.last_velocity) > OVERSPEED_STOP:
                raise Stop(f"overspeed {motor.last_velocity:+.2f} rad/s")
            age = time.monotonic() - motor.last_feedback_time
            if motor.last_feedback_time <= 0.0 or age > FEEDBACK_TIMEOUT:
                raise Stop(f"feedback timeout {age:.2f} s")
            pos = motor.last_position
            if pos is not None and abs(cmd - pos) > TRACKING_STOP:
                raise Stop(f"tracking error {math.degrees(cmd - pos):+.1f} deg")

            arrived = cmd == target and pos is not None and abs(target - pos) < ARRIVE_TOL
            if park_then_stop and arrived:
                raise Stop("parked")
            if now >= next_state:
                next_state = now + 0.1
                write_state(args.state_file, {
                    "t": time.time(), "enabled": True,
                    "pos_deg": None if pos is None else round(math.degrees(pos), 2),
                    "cmd_deg": round(math.degrees(cmd), 2), "target_deg": round(math.degrees(target), 2),
                    "arrived": arrived,
                })
            sleep = period - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)
    except Stop as e:
        exit_reason = str(e)
    except Exception as e:
        exit_reason = f"error: {e!r}"
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            motor.stop()
        except can.CanError:
            pass
        bus.shutdown()
        try:
            write_state(args.state_file, {"t": time.time(), "enabled": False, "reason": exit_reason})
        except OSError:
            pass
        print(f"stopped: {exit_reason}", flush=True)
    return 0 if exit_reason in ("parked", "stop command") else 1


if __name__ == "__main__":
    sys.exit(main())
