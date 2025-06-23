import math
import sys
from typing import List, Tuple

from engine.game_rule import GameSpec, empty_pod_state
from engine.game_sim import (
    Boost,
    Normal,
    PodMovement,
    PodState,
    Shield,
    shield_next_state,
)
from engine.vec2 import Vec2

# I/O for game simulation or player interaction

Ckpts = List[Vec2]


def print_point(v: Vec2) -> None:
    x, y = round(v.x), round(v.y)
    print(f"{x} {y}")


def read_point() -> Vec2:
    parts = sys.stdin.readline().split()
    x, y = map(float, parts)
    return Vec2(x, y)


def print_ckpts(spec: GameSpec) -> None:
    print(spec.laps)
    print(len(spec.checkpoints))
    for ck in spec.checkpoints:
        print_point(ck)


def read_ckpts() -> Tuple[int, Ckpts]:
    laps = int(sys.stdin.readline())
    cnt = int(sys.stdin.readline())
    ckpts = [read_point() for _ in range(cnt)]
    return laps, ckpts


def print_pod(spec: GameSpec, pod: PodState) -> None:
    # prints x y vx vy angle_deg checkpoint_index
    x, y = pod.pod_position.x, pod.pod_position.y
    vx, vy = pod.pod_speed.x, pod.pod_speed.y
    angle = pod.pod_angle or 0.0
    # find index of next checkpoint in spec
    idx = spec.checkpoints.index(pod.pod_next_checkpoints[0]) if pod.pod_next_checkpoints else 0
    print(f"{round(x)} {round(y)} {round(vx)} {round(vy)} {round(math.degrees(angle))} {idx}")


def read_pod(info: Tuple[int, Ckpts]) -> PodState:
    laps, ckpts = info
    parts = sys.stdin.readline().split()
    x, y, vx, vy, angle_deg, ck_idx = map(int, parts)
    pod = empty_pod_state
    pod.pod_position = Vec2(x, y)
    pod.pod_speed = Vec2(vx, vy)
    pod.pod_angle = math.radians(angle_deg)
    # reconstruct next checkpoints
    pod.pod_next_checkpoints = (ckpts[1:] * laps)[: len(ckpts) * laps]
    return pod


def put_movement(mov: PodMovement) -> None:
    tx, ty = mov.target.x, mov.target.y
    thrust = mov.thrust
    if isinstance(thrust, Boost):
        tstr = "BOOST"
    elif isinstance(thrust, Shield):
        tstr = "SHIELD"
    else:
        tstr = str(thrust.n)
    print(f"{round(tx)} {round(ty)} {tstr}")


def log_str(msg: str) -> None:
    print(msg, file=sys.stderr)
