import copy
import math
import random
from typing import Any, Callable, List, Tuple

import engine.util as U
from engine.game_sim import (
    Normal,
    PodMovement,
    PodState,
    move_pods,
    rotate_pod,
    round_trunc,
    speed_decay,
    thrust_pod,
)
from engine.vec2 import Vec2, dist, norm, random_vec, rotate90, scalar_mul, zero_vec

max_sim_turn = 1000


class GameSpec:
    def __init__(self, pod_count: int, laps: int, checkpoints: List[Vec2]) -> None:
        self.pod_count = pod_count
        self.laps = laps
        self.checkpoints = checkpoints


GameHistory = List[List[PodState]]

empty_movement = PodMovement(zero_vec, Normal(0))
empty_pod_state = PodState(zero_vec, zero_vec, None, True, None, empty_movement, [])


def init_pod_states(spec: GameSpec) -> List[PodState]:
    laps, ckpts = spec.laps, spec.checkpoints
    # Include all checkpoints for each lap, starting from checkpoint 0
    pod_ckpts = (ckpts * laps)[: len(ckpts) * laps]
    ckpt0, ckpt1 = ckpts[0], ckpts[1]
    perp = rotate90(ckpt1 - ckpt0)
    shift = scalar_mul(450 / norm(perp), perp)
    if spec.pod_count == 1:
        # Single pod game, only one pod state
        positions = [ckpt0 + shift]
    else:
        # Two pods game, both starting at the first checkpoint
        positions = [ckpt0 + shift, ckpt0 - shift]

    pod_next_ckpts = pod_ckpts.copy()
    pod_next_ckpts.pop(0)
    states = []
    for pos in positions:

        angle = random.uniform(-math.pi, math.pi)
        state = PodState(
            pod_position=pos,
            pod_speed=zero_vec,
            pod_angle=angle,
            pod_boost_avail=True,
            pod_shield_state=None,
            pod_movement=PodMovement(zero_vec, Normal(0)),
            pod_next_checkpoints=pod_next_ckpts.copy(),
            pod_checkpoints=pod_ckpts.copy(),
        )
        states.append(state)
    return states


def specific_game_spec(pod_count: int, lap_count: int, checkpoints: List[Vec2]) -> GameSpec:
    return GameSpec(pod_count, lap_count, checkpoints)


def specific_game_spec_w_randomly_positioned_ckpts(pod_count: int, lap_count: int, n_ckpts: int) -> GameSpec:
    ckpts: List[Vec2] = []
    while len(ckpts) < n_ckpts:
        candidate = random_vec(zero_vec, U.game_world_size)
        if all(dist(candidate, existing) >= 2 * U.check_point_radius for existing in ckpts):
            ckpts.append(candidate)
    return GameSpec(pod_count, lap_count, ckpts)


def random_game_spec(pod_count, lap_count, small_map=False) -> GameSpec:
    n_ckpts = random.randint(3, 8)
    if small_map:
        n_ckpts = random.randint(3, 4)

    ckpts: List[Vec2] = []
    while len(ckpts) < n_ckpts:
        candidate = random_vec(zero_vec, U.game_world_size)
        if all(dist(candidate, existing) >= 2 * U.check_point_radius for existing in ckpts):
            ckpts.append(candidate)
    return GameSpec(pod_count, lap_count, ckpts)


def game_end(history: GameHistory) -> bool:
    return any(len(p.pod_next_checkpoints) == 0 for p in history[-1])


def game_end_solo(history: GameHistory) -> bool:
    # check if player got stuck on 100 turns on some checkpoint
    max_sim_turn = 100

    if len(history) >= 500:
        return True

    if any(len(p.pod_next_checkpoints) == 0 for p in history[-1]):
        return True

    if len(history) <= max_sim_turn:
        return False

    for i in range(len(history) - 1, len(history) - max_sim_turn - 1, -1):

        if len(history[i][0].pod_next_checkpoints) != len(history[i - 1][0].pod_next_checkpoints):
            return False

    return True


def game_end_solo_v2(history: GameHistory) -> bool:
    # check if player got stuck on 100 turns on some checkpoint
    max_sim_turn = 100

    if len(history) >= 100:
        return True

    if any(len(p.pod_next_checkpoints) == 0 for p in history[-1]):
        return True

    if len(history) <= max_sim_turn:
        return False

    for i in range(len(history) - 1, len(history) - max_sim_turn - 1, -1):

        if len(history[i][0].pod_next_checkpoints) != len(history[i - 1][0].pod_next_checkpoints):
            return False

    return True


def player_drive_pod(players: List[Any], pods: List[PodState]) -> Tuple[Any, Any, List[PodState]]:
    # p1 and p2 implement init/run like PlayerIO
    g = pods
    if len(players) == 2:
        p1, p2 = players
        p1_in = (g[:1], g[1:])  # Each player gets 1 pod
        p2_in = (g[1:], g[:1])  # Each player gets 1 pod
    else:  # Single player mode
        p1, p2 = players[0], None
        p1_in = (g, [])  # Single player gets all pods, no opponent
        p2_in = ([], g)  # No opponent pods

    outputs1, p1_next = p1.run(p1_in)
    if p2 is None:
        outputs2 = []
        p2_next = None
    else:
        outputs2, p2_next = p2.run(p2_in)

    # Make sure we have the right number of outputs
    all_outputs = outputs1 + outputs2
    updated = []

    for i, pod in enumerate(g):
        if i < len(all_outputs):
            pod.pod_movement = all_outputs[i]
        updated.append(pod)

    return p1_next, p2_next, updated


def run_game(players: Tuple[Any, Any], spec: GameSpec, stop_rule: Callable[[GameHistory], bool]) -> GameHistory:
    p1, p2 = players
    p1_init = p1.init()
    p2_init = p2.init()

    g0 = init_pod_states(spec)

    p1_next, p2_next, g1 = player_drive_pod([p1_init, p2_init], g0)

    g1 = [thrust_pod(rotate_pod(ps)) for ps in g1]

    history: GameHistory = []
    history.append(copy.deepcopy(g1))

    while len(history) <= max_sim_turn and not stop_rule(history):
        current = history[-1]
        moved = [speed_decay(ps) for ps in move_pods(1.0, current)]
        p1, p2, driven = player_drive_pod([p1_next, p2_next], moved)
        round_rotated = [round_trunc(thrust_pod(rotate_pod(ps))) for ps in driven]

        history.append(copy.deepcopy(round_rotated))

    return history


def run_game_one_player(
    player: Any, spec: GameSpec, stop_rule: Callable[[GameHistory], bool], debug: bool = False
) -> GameHistory:
    p1 = player.init()
    g0 = init_pod_states(spec)

    history: GameHistory = []
    history.append(copy.deepcopy(g0))

    p1_next, _, g1 = player_drive_pod([p1], g0)
    g1 = [thrust_pod(rotate_pod(ps)) for ps in g1]

    history.append(copy.deepcopy(g1))

    if debug:
        print(f"Initial state: {g1}")

    while len(history) <= max_sim_turn and not stop_rule(history):
        current = history[-1]
        moved = [speed_decay(ps) for ps in move_pods(1.0, current)]
        p1, _, driven = player_drive_pod([p1_next], moved)
        round_rotated = [round_trunc(thrust_pod(rotate_pod(ps))) for ps in driven]

        history.append(copy.deepcopy(round_rotated))

    return history
