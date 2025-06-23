import copy
import math
from random import randint
from typing import List

import numpy as np
from tqdm import tqdm

from engine.game_rule import init_pod_states
from engine.game_sim import (
    Boost,
    Normal,
    PodMovement,
    PodState,
    Shield,
    first_collision,
    move_pods,
    rotate_pod,
    round_trunc,
    round_trunc_list,
    speed_decay,
    speed_decay_list,
    thrust_pod,
)
from engine.vec2 import Vec2, cross, dot, vec2ByAngle

ACTIONS = []

angle_deltas = [-18, 0, 18]
thrust_values = [0, 200]

for thrust in thrust_values:
    for angle_delta in angle_deltas:
        ACTIONS.append((angle_delta, Normal(thrust)))

ACTIONS.append((0, Shield()))  # No angle change with Shield


def discretize_state_runner_solo(state):
    if not state.pod_next_checkpoints:
        return [0] * 8  # Expanded state space

    CP1 = state.pod_next_checkpoints[0]
    CP2 = state.pod_next_checkpoints[1] if len(state.pod_next_checkpoints) > 1 else state.pod_checkpoints[0]

    angle = state.pod_angle or 0

    # print(f"angle1: {angle}")

    # angle = math.radians(angle)

    pod = Vec2(state.pod_position.x, state.pod_position.y)
    speed = Vec2(state.pod_speed.x, state.pod_speed.y)

    distance = pod.distance(CP1)

    angle_to_cp1 = pod.angle_diff(CP1)
    angle -= angle_to_cp1  # Normalize angle to CP1

    angle_v = Vec2(0, 0).angle_diff(speed) - angle_to_cp1  # Angle of velocity vector towards CP1
    norm_v = math.sqrt(speed.x**2 + speed.y**2)

    angle_next_dir = CP1.angle_diff(CP2) - angle_to_cp1  # Angle to next checkpoint
    distance_cp1_cp2 = CP1.distance(CP2)

    # print(f"angle2: {angle}")

    angle_dot = dot(vec2ByAngle(angle), Vec2(1, 0))  # Cosine of pod angle
    angle_cross = cross(vec2ByAngle(angle), Vec2(1, 0))  # Sine of pod angle

    v_dot = int(norm_v * dot(vec2ByAngle(angle_v), Vec2(1, 0)))  # Cosine of pod speed angle
    v_cross = int(norm_v * cross(vec2ByAngle(angle_v), Vec2(1, 0)))  # Sine of pod speed angle

    next_dir_dot = dot(vec2ByAngle(angle_next_dir), Vec2(1, 0))  # Cosine of target direction
    next_dir_cross = cross(vec2ByAngle(angle_next_dir), Vec2(1, 0))  # Sine of target direction

    return np.array(
        [
            (distance - 600) / 20000.0,  # Normalize distance to CP1
            angle_dot,
            angle_cross,
            v_dot / 1000.0,  # Normalize speed components
            v_cross / 1000.0,  # Normalize speed components
            next_dir_dot,
            next_dir_cross,
            (distance_cp1_cp2 - 1200.0) / 10000.0,  # Normalize distance between CP1 and CP2
        ]
    )


def discretize_state_runner(runner_state: PodState, blocker_state: PodState):
    if not runner_state.pod_next_checkpoints:
        return [0] * 13  # Expanded state space #

    CP1 = runner_state.pod_next_checkpoints[0]
    CP2 = (
        runner_state.pod_next_checkpoints[1]
        if len(runner_state.pod_next_checkpoints) > 1
        else runner_state.pod_checkpoints[0]
    )

    # print(f"angle1: {angle}")

    # angle = math.radians(angle)

    runner_pos = Vec2(runner_state.pod_position.x, runner_state.pod_position.y)
    blocker_pos = Vec2(blocker_state.pod_position.x, blocker_state.pod_position.y)

    runner_distance_to_cp1 = runner_pos.distance(CP1)
    runner_angle_to_cp1 = runner_pos.angle_diff(CP1)

    runner_angle = runner_state.pod_angle - runner_angle_to_cp1  # Normalize angle to CP1
    runner_angle_dot = dot(vec2ByAngle(runner_angle), Vec2(1, 0))  # Cosine of runner angle
    runner_angle_cross = cross(vec2ByAngle(runner_angle), Vec2(1, 0))  # Sine of runner angle

    angle_v = Vec2(0, 0).angle_diff(runner_state.pod_speed) - runner_angle_to_cp1
    norm_v = math.sqrt(runner_state.pod_speed.x**2 + runner_state.pod_speed.y**2)

    runner_v_dot = int(norm_v * dot(vec2ByAngle(angle_v), Vec2(1, 0)))  # Cosine of runner speed angle
    runner_v_cross = int(norm_v * cross(vec2ByAngle(angle_v), Vec2(1, 0)))  # Sine of runner speed angle

    angle_cp1_cp2 = CP1.angle_diff(CP2) - runner_angle_to_cp1  # Angle to next checkpoint
    distance_cp1_cp2 = CP1.distance(CP2)

    angle_cp1_cp2_dot = dot(vec2ByAngle(angle_cp1_cp2), Vec2(1, 0))  # Cosine of target direction
    angle_cp1_cp2_cross = cross(vec2ByAngle(angle_cp1_cp2), Vec2(1, 0))  # Sine of target direction

    distance_runner_to_blocker = runner_pos.distance(blocker_pos)
    runner_angle_to_blocker = runner_pos.angle_diff(blocker_pos)

    # Normalize runner angle to blocker
    blocker_angle = runner_state.pod_angle - runner_angle_to_blocker  # Normalize angle to blocker

    runner_angle_to_blocker_dot = dot(vec2ByAngle(blocker_angle), Vec2(1, 0))  # Cosine of runner angle to blocker
    runner_angle_to_blocker_cross = cross(vec2ByAngle(blocker_angle), Vec2(1, 0))  # Sine of runner angle to blocker

    angle_v_blocker = Vec2(0, 0).angle_diff(blocker_state.pod_speed) - runner_angle_to_blocker
    norm_v_blocker = math.sqrt(blocker_state.pod_speed.x**2 + blocker_state.pod_speed.y**2)

    blocker_v_dot = int(
        norm_v_blocker * dot(vec2ByAngle(angle_v_blocker), Vec2(1, 0))
    )  # Cosine of runner speed angle to blocker
    blocker_v_cross = int(
        norm_v_blocker * cross(vec2ByAngle(angle_v_blocker), Vec2(1, 0))
    )  # Sine of runner speed angle to blocker

    return np.array(
        [
            (runner_distance_to_cp1 - 600) / 20000.0,  # Normalize distance to CP1
            runner_angle_dot,
            runner_angle_cross,
            runner_v_dot / 1000.0,  # Normalize speed components
            runner_v_cross / 1000.0,  # Normalize speed components
            angle_cp1_cp2_dot,
            angle_cp1_cp2_cross,
            (distance_cp1_cp2 - 1200.0) / 10000.0,  # Normalize distance between CP1 and CP2
            distance_runner_to_blocker / 20000.0,  # Normalize distance to blocker
            runner_angle_to_blocker_dot,
            runner_angle_to_blocker_cross,
            blocker_v_dot / 1000.0,  # Normalize blocker speed components
            blocker_v_cross / 1000.0,  # Normalize blocker speed components
        ]
    )


def discretize_state_blocker(blocker_state: PodState, runner_state: PodState):
    if not blocker_state.pod_next_checkpoints or not runner_state.pod_next_checkpoints:
        return [0] * 19  # Expanded state space

    discretized_state = [0] * 19  # Expanded state space
    CP1 = runner_state.pod_next_checkpoints[0]
    # CP2 = (
    # runner_state.pod_next_checkpoints[1]
    # if len(runner_state.pod_next_checkpoints) > 1
    # else runner_state.pod_checkpoints[0]
    # )
    # CP3 = (
    # runner_state.pod_next_checkpoints[2]
    # if len(runner_state.pod_next_checkpoints) > 2
    # else runner_state.pod_checkpoints[1]
    # )

    # angle = state.pod_angle or 0

    # print(f"angle1: {angle}")

    # angle = math.radians(angle)
    runner_pos = Vec2(runner_state.pod_position.x, runner_state.pod_position.y)
    blocker_pos = Vec2(blocker_state.pod_position.x, blocker_state.pod_position.y)

    blocker_distance_to_runner = blocker_pos.distance(runner_pos)
    blocker_angle_to_runner = blocker_pos.angle_diff(runner_pos)

    # Normalize blocker angle to runner
    blocker_angle = blocker_state.pod_angle - blocker_angle_to_runner  # Normalize angle to runner
    blocker_angle_dot = dot(vec2ByAngle(blocker_angle), Vec2(1, 0))  # Cosine of blocker angle
    blocker_angle_cross = cross(vec2ByAngle(blocker_angle), Vec2(1, 0))  # Sine of blocker angle

    angle_v = (
        Vec2(0, 0).angle_diff(blocker_state.pod_speed) - blocker_angle_to_runner
    )  # Angle of velocity vector towards runner
    norm_v = math.sqrt(blocker_state.pod_speed.x**2 + blocker_state.pod_speed.y**2)
    blocker_v_dot = int(norm_v * dot(vec2ByAngle(angle_v), Vec2(1, 0)))  # Cosine of blocker speed angle
    blocker_v_cross = int(norm_v * cross(vec2ByAngle(angle_v), Vec2(1, 0)))  # Sine of blocker speed angle

    runner_angle_to_cp1 = runner_pos.angle_diff(CP1)

    # Normalize runner angle to CP1
    runner_angle = runner_state.pod_angle - runner_angle_to_cp1  # Normalize angle to CP1
    runner_angle_dot = dot(vec2ByAngle(runner_angle), Vec2(1, 0))  # Cosine of runner angle
    runner_angle_cross = cross(vec2ByAngle(runner_angle), Vec2(1, 0))  # Sine of runner angle

    angle_v = (
        Vec2(0, 0).angle_diff(runner_state.pod_speed) - runner_angle_to_cp1
    )  # Angle of velocity vector towards CP1
    norm_v = math.sqrt(runner_state.pod_speed.x**2 + runner_state.pod_speed.y**2)
    runner_v_dot = int(norm_v * dot(vec2ByAngle(angle_v), Vec2(1, 0)))  # Cosine of runner speed angle
    runner_v_cross = int(norm_v * cross(vec2ByAngle(angle_v), Vec2(1, 0)))  # Sine of runner speed angle

    runner_distance_to_CP1 = runner_pos.distance(CP1)

    # blocker_distance_to_CP1 = blocker_pos.distance(CP1)
    # blocker_distance_to_CP2 = blocker_pos.distance(CP2)
    # blocker_distance_to_CP3 = blocker_pos.distance(CP3)

    # blocker_angle_to_CP1 = blocker_pos.angle_diff(CP1)
    # blocker_angle_to_CP2 = blocker_pos.angle_diff(CP2)
    # blocker_angle_to_CP3 = blocker_pos.angle_diff(CP3)

    # blocker_angle_to_CP1 = blocker_state.pod_angle - blocker_angle_to_CP1  # Normalize angle to CP1
    # blocker_angle_to_CP2 = blocker_state.pod_angle - blocker_angle_to_CP2  # Normalize angle to CP2
    # blocker_angle_to_CP3 = blocker_state.pod_angle - blocker_angle_to_CP3  # Normalize angle to CP3

    # blocker_angle_dot_CP1 = dot(vec2ByAngle(blocker_angle_to_CP1), Vec2(1, 0))  # Cosine of blocker angle to CP1
    # blocker_angle_cross_CP1 = cross(vec2ByAngle(blocker_angle_to_CP1), Vec2(1, 0))  # Sine of blocker angle to CP1
    # blocker_angle_dot_CP2 = dot(vec2ByAngle(blocker_angle_to_CP2), Vec2(1, 0))  # Cosine of blocker angle to CP2
    # blocker_angle_cross_CP2 = cross(vec2ByAngle(blocker_angle_to_CP2), Vec2(1, 0))  # Sine of blocker angle to CP2
    # blocker_angle_dot_CP3 = dot(vec2ByAngle(blocker_angle_to_CP3), Vec2(1, 0))  # Cosine of blocker angle to CP3
    # blocker_angle_cross_CP3 = cross(vec2ByAngle(blocker_angle_to_CP3), Vec2(1, 0))  # Sine of blocker angle to CP3

    discretized_state[0] = blocker_distance_to_runner / 20000.0  # Normalize distance to runner
    discretized_state[1] = blocker_angle_dot
    discretized_state[2] = blocker_angle_cross
    discretized_state[3] = blocker_v_dot / 1000.0  # Normalize blocker speed components
    discretized_state[4] = blocker_v_cross / 1000.0  # Normalize blocker speed components
    discretized_state[5] = runner_distance_to_CP1 / 20000.0  # Normalize distance to CP1
    discretized_state[6] = runner_angle_dot
    discretized_state[7] = runner_angle_cross
    discretized_state[8] = runner_v_dot / 1000.0  # Normalize runner speed components
    discretized_state[9] = runner_v_cross / 1000.0  # Normalize runner speed components

    for idx in range(3):
        CP = (
            runner_state.pod_next_checkpoints[idx]
            if idx < len(runner_state.pod_next_checkpoints)
            else runner_state.pod_checkpoints[idx - len(runner_state.pod_next_checkpoints)]
        )

        blocker_distance_to_CP = blocker_pos.distance(CP)
        blocker_angle = blocker_state.pod_angle - blocker_pos.angle_diff(CP)
        blocker_angle_dot = dot(vec2ByAngle(blocker_angle), Vec2(1, 0))
        blocker_angle_cross = cross(vec2ByAngle(blocker_angle), Vec2(1, 0))

        discretized_state[10 + idx * 3] = blocker_distance_to_CP / 20000.0  # Normalize distance to CP
        discretized_state[10 + idx * 3 + 1] = blocker_angle_dot
        discretized_state[10 + idx * 3 + 2] = blocker_angle_cross

    return discretized_state
    # return np.array(
    #     [
    #         blocker_distance_to_runner / 20000.0,  # Normalize distance to runner
    #         blocker_angle_dot,
    #         blocker_angle_cross,
    #         blocker_v_dot / 1000.0,  # Normalize blocker speed components
    #         blocker_v_cross / 1000.0,  # Normalize blocker speed components
    #         runner_distance_to_CP1 / 20000.0,  # Normalize distance to CP1
    #         runner_angle_dot,
    #         runner_angle_cross,
    #         runner_v_dot / 1000.0,  # Normalize runner speed components
    #         runner_v_cross / 1000.0,  # Normalize runner speed components
    #         # blocker_distance_to_CP1 / 20000.0,  # Normalize distance to CP1
    #         # blocker_angle_dot_CP1,
    #         # blocker_angle_cross_CP1,
    #         # blocker_distance_to_CP2 / 20000.0,  # Normalize distance to CP2
    #         # blocker_angle_dot_CP2,
    #         # blocker_angle_cross_CP2,
    #         # blocker_distance_to_CP3 / 20000.0,  # Normalize distance to CP3
    #         # blocker_angle_dot_CP3,
    #         # blocker_angle_cross_CP3,
    #     ]
    # )


class envSolo:
    def __init__(self, game_spec=None):
        self.state = PodState()
        self.state_for_history = PodState()
        self.game_spec = game_spec
        self.timeout = 100
        self.checkpoints_left = None

    def reward(self):

        # if len(self.state.pod_next_checkpoints) == 0:
        #     return 10

        checkpoint_reward = 0
        if self.checkpoints_left is not None and len(self.state.pod_next_checkpoints) < self.checkpoints_left:
            checkpoint_reward = 5

        timeout_penalty = self.timeout == 0

        out_of_bounds_penalty = (
            self.state.pod_position.x < -2000
            or self.state.pod_position.x > 20000
            or self.state.pod_position.y < -2000
            or self.state.pod_position.y > 15000
        )

        # distance_reward = (
        #     0
        #     if len(self.state.pod_next_checkpoints) == 0
        #     else self.state.pod_position.distance(self.state.pod_next_checkpoints[0]) / 20000.0
        # )

        return checkpoint_reward - timeout_penalty - out_of_bounds_penalty  # - distance_reward

    def step(self, action_idx):
        self.timeout -= 1

        angle_delta, thrust = ACTIONS[action_idx]

        pod_angle = self.state.pod_angle or 0
        px = self.state.pod_position.x
        py = self.state.pod_position.y

        a = pod_angle + (angle_delta * np.pi / 180.0)
        nx = int(px + math.cos(a) * 1000)
        ny = int(py + math.sin(a) * 1000)

        target = Vec2(nx, ny)
        movement = PodMovement(target, thrust)
        self.state.pod_movement = movement

        self.state = thrust_pod(rotate_pod(self.state))
        self.state_for_history = copy.deepcopy(self.state)
        self.state = round_trunc(speed_decay(move_pods(1.0, [self.state])[0]))

        if self.checkpoints_left is not None and len(self.state.pod_next_checkpoints) < self.checkpoints_left:
            self.timeout = 100

        reward = self.reward()

        self.checkpoints_left = len(self.state.pod_next_checkpoints)

        if len(self.state.pod_next_checkpoints) == 0 or self.timeout == 0:
            return reward, self.state, True
        return reward, self.state, False

    def reset(self, spec):
        self.state = init_pod_states(spec)[0]
        self.timeout = 100
        self.checkpoints_left = None

        return self.state


class envBlocker:
    def __init__(self, game_spec=None, num_agents=2):
        self.states = [PodState() for _ in range(num_agents)]
        self.states_for_history = [PodState() for _ in range(num_agents)]
        self.game_spec = game_spec
        self.runner_timeout = 100
        self.checkpoints_left = [None] * num_agents
        self.blocker_idx = 0

    def reward(self):
        runner_idx = 1 - self.blocker_idx
        blocker_idx = self.blocker_idx
        blocker_state = self.states[self.blocker_idx]
        runner_state = self.states[1 - self.blocker_idx]

        checkpoint_reward = 0
        if (
            self.checkpoints_left[runner_idx] is not None
            and len(runner_state.pod_next_checkpoints) < self.checkpoints_left[runner_idx]
        ):
            checkpoint_reward = 5

        timeout_reward = 3 if (self.runner_timeout == 0) else 0

        out_of_bounds_reward_blocker = (
            blocker_state.pod_position.x < -2000
            or blocker_state.pod_position.x > 20000
            or blocker_state.pod_position.y < -2000
            or blocker_state.pod_position.y > 15000
        )

        out_of_bounds_reward_runner = (
            runner_state.pod_position.x < -2000
            or runner_state.pod_position.x > 20000
            or runner_state.pod_position.y < -2000
            or runner_state.pod_position.y > 15000
        )

        collision_reward = 0
        fc = first_collision([runner_state, blocker_state])
        if fc is not None and fc[2] < 1.0:
            collision_reward = 0.5

        distance_runner_to_ckpt_reward = (
            0
            if len(runner_state.pod_next_checkpoints) == 0
            else runner_state.pod_position.distance(runner_state.pod_next_checkpoints[0]) / 20000.0
        )

        blocker_reward = -checkpoint_reward + timeout_reward - out_of_bounds_reward_blocker + collision_reward

        runner_reward = (
            +checkpoint_reward - timeout_reward - out_of_bounds_reward_runner - distance_runner_to_ckpt_reward
        )

        return blocker_reward, runner_reward

    def step(self, actions):
        self.runner_timeout -= 1

        runner_idx = 1 - self.blocker_idx
        blocker_idx = self.blocker_idx

        if len(actions) != len(self.states):
            raise ValueError(
                f"Number of actions does not match the number of agents in the environment. Actions: {actions} States: {len(self.states)}"
            )

        for idx, action_idx in enumerate(actions):
            angle_delta, thrust = ACTIONS[action_idx]

            pod_angle = self.states[idx].pod_angle or 0
            px = self.states[idx].pod_position.x
            py = self.states[idx].pod_position.y

            a = pod_angle + (angle_delta * np.pi / 180.0)
            nx = int(px + math.cos(a) * 1000)
            ny = int(py + math.sin(a) * 1000)

            target = Vec2(nx, ny)
            movement = PodMovement(target, thrust)
            self.states[idx].pod_movement = movement

            self.states[idx] = thrust_pod(rotate_pod(self.states[idx]))

        self.states_for_history = copy.deepcopy(self.states)
        self.states = round_trunc_list(speed_decay_list(move_pods(1.0, self.states)))

        if (
            self.checkpoints_left[runner_idx] is not None
            and len(self.states[runner_idx].pod_next_checkpoints) < self.checkpoints_left[runner_idx]
        ):
            # print(f"Checkpoint passed by runner {runner_idx}, resetting enemy timeout.")
            # print(self.states[runner_idx].pod_next_checkpoints)
            # print(self.checkpoints_left[runner_idx])
            self.runner_timeout = 100

        rewards = self.reward()
        self.checkpoints_left[runner_idx] = len(self.states[runner_idx].pod_next_checkpoints)

        done = any(len(state.pod_next_checkpoints) == 0 for state in self.states) or self.runner_timeout == 0
        return rewards, self.states, done

    def reset(self, spec):
        self.states = init_pod_states(spec)
        self.runner_timeout = 100
        self.checkpoints_left = [None] * len(self.states)

        return self.states
