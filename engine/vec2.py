import math
import random
from typing import List, Tuple


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> "Vec2":
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Vec2":
        return Vec2(self.x / other, self.y / other)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec2) and self.x == other.x and self.y == other.y

    def distance(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_diff(self, p):
        # print('px = {}  x= {} py = {} y = {}'.format(p.x, self.x, p.y, self.y))
        return math.atan2(p.y - self.y, p.x - self.x)

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"


def vec2ByAngle(angle):
    return Vec2(math.cos(angle), math.sin(angle))


def element_wise(f, v: Vec2) -> Vec2:
    return Vec2(f(v.x), f(v.y))


def element_wise2(f, v1: Vec2, v2: Vec2) -> Vec2:
    return Vec2(f(v1.x, v2.x), f(v1.y, v2.y))


def scalar_mul(c: float, v: Vec2) -> Vec2:
    return Vec2(c * v.x, c * v.y)


def scalar_div(v: Vec2, c: float) -> Vec2:
    return Vec2(v.x / c, v.y / c)


def dot(v1: Vec2, v2: Vec2) -> float:
    return v1.x * v2.x + v1.y * v2.y


def cross(a: Vec2, b: Vec2) -> float:
    return a.y * b.x - a.x * b.y


def norm(v: Vec2) -> float:
    return math.hypot(v.x, v.y)


def dist(v1: Vec2, v2: Vec2) -> float:
    return norm(v1 - v2)


def arg(v: Vec2) -> float:
    return math.atan2(v.y, v.x)


def proj(v1: Vec2, v2: Vec2) -> Vec2:
    coeff = dot(v1, v2) / dot(v2, v2)
    return scalar_mul(coeff, v2)


def rotate(theta: float, v: Vec2) -> Vec2:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return Vec2(cos_t * v.x - sin_t * v.y, sin_t * v.x + cos_t * v.y)


def unit_vec(theta: float) -> Vec2:
    return Vec2(math.cos(theta), math.sin(theta))


def rotate90(v: Vec2) -> Vec2:
    return Vec2(-v.y, v.x)


def reflect(mirror: Vec2, v: Vec2) -> Vec2:
    mu = mirror / norm(mirror)
    projection = scalar_mul(dot(mu, v), mu)
    return projection * 2 - v


def is_zero(v: Vec2) -> bool:
    return v.x == 0 and v.y == 0


zero_vec = Vec2(0.0, 0.0)


def random_vec(p1: Vec2, p2: Vec2) -> Vec2:
    min_x, max_x = min(p1.x, p2.x), max(p1.x, p2.x)
    min_y, max_y = min(p1.y, p2.y), max(p1.y, p2.y)
    return Vec2(random.uniform(min_x, max_x), random.uniform(min_y, max_y))


def round_vec(v: Vec2) -> Vec2:
    return Vec2(round(v.x), round(v.y))


def normalize(v: Vec2) -> Vec2:
    n = norm(v)
    return v / n if n != 0 else Vec2(0.0, 0.0)
