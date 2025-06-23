import math
import random
from array import array
from typing import List, Tuple, TypeVar

from engine.vec2 import Vec2

T = TypeVar("T")

game_world_size = Vec2(16000.0, 9000.0)
max_turn_angle = math.radians(18)
boost_accel = 650.0
pod_force_field_radius = 400.0
check_point_radius = 600.0  # Standard checkpoint radius for capture
pod_min_collision_impact = 120.0


def deg_to_rad(x: float) -> float:
    return math.radians(x)


def rad_to_deg(x: float) -> float:
    return math.degrees(x)


def normalize_rad(x: float) -> float:
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def normalize_deg(x: float) -> float:
    return x - round(x / 360.0) * 360.0


def clamp(mi: float, x: float, ma: float) -> float:
    return max(mi, min(x, ma))


def fmod(x: float, y: float) -> float:
    return x - math.floor(x / y) * y


def distinct_pairs(xs: List[T]) -> List[Tuple[T, T]]:
    pairs: List[Tuple[T, T]] = []
    n = len(xs)
    for i in range(n - 1):
        for j in range(i + 1, n):
            pairs.append((xs[i], xs[j]))
    return pairs


def random_perm(xs: List[T]) -> List[T]:
    arr = xs[:]
    random.shuffle(arr)
    return arr


def get_min_rep_len(xs: List[T]) -> int:
    def is_rep_of(needle: List[T], haystack: List[T]) -> bool:
        if not haystack:
            return True
        if haystack[: len(needle)] == needle:
            return is_rep_of(needle, haystack[len(needle) :])
        return False

    for n in range(1, len(xs) + 1):
        if is_rep_of(xs[:n], xs):
            return n
    return len(xs)
