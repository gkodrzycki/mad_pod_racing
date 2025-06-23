# game_sim.py
import math
from typing import List, Optional, Tuple

import engine.util as U
from engine.vec2 import Vec2, arg, dot, element_wise, norm, proj, scalar_mul, unit_vec

# -----------------------------------------------------------------------------#
#  Typy pomocnicze                                                              #
# -----------------------------------------------------------------------------#
Angle = float


class Thrust:
    pass


class Normal(Thrust):
    def __init__(self, n: int) -> None:
        self.n = n


class Shield(Thrust):
    pass


class Boost(Thrust):
    pass


class PodMovement:
    def __init__(self, target: Vec2, thrust: Thrust) -> None:
        self.target = target
        self.thrust = thrust

    def __repr__(self):
        return f"PodMovement(target={self.target}, thrust={self.thrust})"


class PodState:
    def __init__(
        self,
        pod_position: Vec2 = Vec2(0, 0),
        pod_speed: Vec2 = Vec2(0, 0),
        pod_angle: Optional[Angle] = None,
        pod_boost_avail: bool = True,
        pod_shield_state: Optional[int] = None,
        pod_movement: PodMovement = PodMovement(Vec2(0, 0), Normal(0)),
        pod_next_checkpoints: List[Vec2] = None,
        pod_checkpoints: Optional[List[Vec2]] = None,
    ) -> None:
        self.pod_position = pod_position
        self.pod_speed = pod_speed
        self.pod_angle = pod_angle
        self.pod_boost_avail = pod_boost_avail
        self.pod_shield_state = pod_shield_state
        self.pod_movement = pod_movement
        self.pod_next_checkpoints = pod_next_checkpoints
        self.pod_checkpoints = pod_checkpoints

    def __repr__(self):
        return (
            f"PodState(pos={self.pod_position}, speed={self.pod_speed}, "
            f"angle={round(self.pod_angle, 2) if self.pod_angle is not None else None}, "
            f"boost={self.pod_boost_avail}, shield={self.pod_shield_state}, "
            f"move={self.pod_movement}, ckpts_left={len(self.pod_next_checkpoints)})"
        )


# -----------------------------------------------------------------------------#
#  Funkcje pomocnicze                                                          #
# -----------------------------------------------------------------------------#
def shield_next_state(activated: bool, ss: Optional[int]) -> Optional[int]:
    """Aktualizacja stanu tarczy (3 → 2 → 1 → None)."""
    if activated:
        return 3
    if ss is None:
        return None
    return ss - 1 if ss > 0 else None


# -----------------------------------------------------------------------------#
#  Obrót (max 18°)                                                             #
# -----------------------------------------------------------------------------#
def rotate_pod(ps: PodState) -> PodState:
    target = ps.pod_movement.target
    if ps.pod_angle is None:
        ps.pod_angle = arg(target - ps.pod_position)
    else:
        delta = (arg(target - ps.pod_position) - ps.pod_angle + math.pi) % (2 * math.pi) - math.pi
        r = U.max_turn_angle
        ps.pod_angle += max(-r, min(delta, r))
    return ps


# -----------------------------------------------------------------------------#
#  Przyspieszenie / BOOST / tarcza                                             #
# -----------------------------------------------------------------------------#
def thrust_pod(ps: PodState) -> PodState:
    thrust = ps.pod_movement.thrust
    ps.pod_shield_state = shield_next_state(isinstance(thrust, Shield), ps.pod_shield_state)
    idle = ps.pod_shield_state is not None

    if idle:
        acc_mag = 0.0
    else:
        if isinstance(thrust, Normal):
            acc_mag = max(0, min(thrust.n, 200))
        elif isinstance(thrust, Boost):
            acc_mag = U.boost_accel if ps.pod_boost_avail else 200
        else:  # Shield
            acc_mag = 0.0

    acc_vec = scalar_mul(acc_mag, unit_vec(ps.pod_angle or 0.0))
    if not idle:
        ps.pod_speed = ps.pod_speed + acc_vec

    if isinstance(thrust, Boost):
        ps.pod_boost_avail = False

    return ps


# -----------------------------------------------------------------------------#
#  Dryf + checkpoint                                                           #
# -----------------------------------------------------------------------------#
def drift_pod(p: PodState, dt: float) -> PodState:
    """Przesuń pada o dt i obsłuż przekroczenie checkpointu."""
    old_pos = p.pod_position
    new_pos = old_pos + scalar_mul(dt, p.pod_speed)

    if p.pod_next_checkpoints:
        checkpoint = p.pod_next_checkpoints[0]

        # Check if we crossed the checkpoint during this movement
        # Use the full checkpoint radius for capture
        capture_radius = U.check_point_radius

        # Check distance at start and end of movement
        old_dist = norm(old_pos - checkpoint)
        new_dist = norm(new_pos - checkpoint)

        # If we're within capture radius at the end, or we crossed through it
        if new_dist <= capture_radius:
            p.pod_next_checkpoints = p.pod_next_checkpoints[1:]
        else:
            # Check if we crossed through the checkpoint circle during movement
            rel_start = old_pos - checkpoint
            movement = new_pos - old_pos

            if norm(movement) > 0:  # Only check if we actually moved
                t_hit = collide_time(rel_start, movement, capture_radius)
                if t_hit is not None and 0 <= t_hit <= 1:
                    p.pod_next_checkpoints = p.pod_next_checkpoints[1:]

    p.pod_position = new_pos
    return p


# -----------------------------------------------------------------------------#
#  Detekcja kolizji                                                            #
# -----------------------------------------------------------------------------#
def collide_time(pos: Vec2, vel: Vec2, radius: float) -> Optional[float]:
    if vel.x == 0 and vel.y == 0:
        return None
    proj_vel = proj(pos, vel)
    nearest = pos - proj_vel
    if dot(pos, vel) >= 0 or dot(nearest, nearest) > radius * radius:
        return None
    dist_before = norm(proj_vel) - math.sqrt(radius * radius - dot(nearest, nearest))
    return dist_before / norm(vel)


def first_collision(pods: List[PodState]) -> Optional[Tuple[int, int, float]]:
    """Zwraca (i1,i2,t) dla najbliższej kolizji lub None."""
    best: Optional[Tuple[int, int, float]] = None
    n = len(pods)
    for i in range(n):
        for j in range(i + 1, n):
            rel_pos = pods[j].pod_position - pods[i].pod_position
            rel_vel = pods[j].pod_speed - pods[i].pod_speed
            t = collide_time(rel_pos, rel_vel, U.pod_force_field_radius * 2)
            if t is not None and t >= 0:
                if best is None or t < best[2]:
                    best = (i, j, t)
    return best


# -----------------------------------------------------------------------------#
#  Odbicie dwóch podów                                                         #
# -----------------------------------------------------------------------------#
def collide_points(p1: PodState, p2: PodState) -> Tuple[PodState, PodState]:
    m1 = 10 if p1.pod_shield_state == 3 else 1
    m2 = 10 if p2.pod_shield_state == 3 else 1

    rel = p2.pod_position - p1.pod_position
    velr = p2.pod_speed - p1.pod_speed
    impact_coeff = 2 * (m1 * m2) / (m1 + m2)
    impact = scalar_mul(impact_coeff, proj(velr, rel))

    if norm(impact) < U.pod_min_collision_impact:
        impact = scalar_mul(U.pod_min_collision_impact / norm(impact), impact)

    p1.pod_speed = p1.pod_speed + (impact / m1)
    p2.pod_speed = p2.pod_speed - (impact / m2)
    return p1, p2


EPS = 1e-8


# -----------------------------------------------------------------------------#
#  Pętla fizyki — bez rekurencji głębokiej                                     #
# -----------------------------------------------------------------------------#
def move_pods(duration: float, pods: List[PodState]) -> List[PodState]:
    elapsed = 0.0
    pss = pods
    while elapsed < duration:
        fc = first_collision(pss)
        if fc is None or elapsed + fc[2] > duration - EPS:
            # drift pozostały czas i koniec
            dt = duration - elapsed
            pss = [drift_pod(p, dt) for p in pss]
            break

        i1, i2, t = fc

        # ►► jeśli kolizja „tu i teraz” – zignoruj i przesuń się o EPS
        if t < EPS:
            pss = [drift_pod(p, EPS) for p in pss]
            elapsed += EPS
            continue

        # dryf do chwili kolizji
        pss = [drift_pod(p, t) for p in pss]
        elapsed += t
        # odbicie tylko kolidujących
        pss[i1], pss[i2] = collide_points(pss[i1], pss[i2])

    return pss


# -----------------------------------------------------------------------------#
#  Dodatkowe operacje                                                          #
# -----------------------------------------------------------------------------#
def speed_decay(ps: PodState) -> PodState:
    ps.pod_speed = scalar_mul(0.85, ps.pod_speed)
    return ps


def round_trunc(ps: PodState) -> PodState:
    ps.pod_position = element_wise(round, ps.pod_position)
    ps.pod_speed = element_wise(lambda x: math.trunc(x), ps.pod_speed)
    return ps


def speed_decay_list(ps: List[PodState]) -> List[PodState]:
    for p in ps:
        p.pod_speed = scalar_mul(0.85, p.pod_speed)
    return ps


def round_trunc_list(ps: List[PodState]) -> List[PodState]:
    for p in ps:
        p.pod_position = element_wise(round, p.pod_position)
        p.pod_speed = element_wise(lambda x: math.trunc(x), p.pod_speed)

    return ps
