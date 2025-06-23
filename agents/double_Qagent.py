import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from engine.game_sim import Boost, Normal, PodMovement, Shield
from engine.model import QNetwork
from engine.model_dueling import DuelingMLP
from engine.replay_buffer import ReplayBuffer
from engine.util import check_point_radius, game_world_size
from engine.vec2 import Vec2
from env import discretize_state_runner_solo

ACTIONS = []
angle_deltas = [-18, 0, 18]
thrust_values = [0, 100]

for thrust in thrust_values:
    for angle_delta in angle_deltas:
        ACTIONS.append((angle_delta, Normal(thrust)))


class DoubleQAgent:
    def __init__(
        self,
        model_type: Literal["standard", "dueling"] = "standard",
        state_dim=8,
        action_dim=len(ACTIONS),
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        alpha=1e-4,
        gamma=0.9,
        replay_buffer_size=2**18,
        batch_size=64,
    ):

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.pick_model = {"standard": QNetwork, "dueling": DuelingMLP}
        self.QA = self.pick_model[model_type](state_dim, action_dim).to(self.device)
        self.QB = self.pick_model[model_type](state_dim, action_dim).to(self.device)
        self.optimizer_A = optim.Adam(self.QA.parameters(), lr=alpha)
        self.optimizer_B = optim.Adam(self.QB.parameters(), lr=alpha)

        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size

        self.prev_state = None
        self.prev_action = None
        self.prev_checkpoint_count = None
        self.steps_since_progress = 0
        self.turns = 0

    def init(self):
        return self

    def forward(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = (self.QA(state_tensor) + self.QB(state_tensor)) / 2.0
        return q_values

    def act_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = (self.QA(state_tensor) + self.QB(state_tensor)) / 2.0
        return q_values.argmax().item()

    def act_non_greedy(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = (self.QA(state_tensor) + self.QB(state_tensor)) / 2.0
        return q_values.argmax().item()

    def evaluate(self):
        self.QA.eval()
        self.QB.eval()

    def train(self):
        self.QA.train()
        self.QB.train()

    def run(self, inputs):
        # import dyskretyzacji z enva
        self_pods, _ = inputs
        pod = self_pods[0]
        state = discretize_state_runner_solo(pod)
        action_idx = self.act_non_greedy(state)

        pod_angle = pod.pod_angle or 0
        angle_delta, thrust = ACTIONS[action_idx]

        a = pod_angle + (angle_delta * np.pi / 180.0)
        nx = int(pod.pod_position.x + math.cos(a) * 1000)
        ny = int(pod.pod_position.y + math.sin(a) * 1000)

        target = Vec2(nx, ny)
        movement = PodMovement(target, thrust)

        return [movement], self

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        if np.random.rand() < 0.5:
            # Update QA using QB
            best_actions = self.QA(next_state_batch).argmax(1).unsqueeze(1)
            q_targets = self.QB(next_state_batch).gather(1, best_actions)
            q_values = self.QA(state_batch).gather(1, action_batch)
            target = reward_batch + self.gamma * q_targets * (1 - done_batch)

            loss = self.criterion(q_values, target)
            self.optimizer_A.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.QA.parameters(), 1.0)
            self.optimizer_A.step()
        else:
            # Update QB using QA
            best_actions = self.QB(next_state_batch).argmax(1).unsqueeze(1)
            q_targets = self.QA(next_state_batch).gather(1, best_actions)
            q_values = self.QB(state_batch).gather(1, action_batch)
            target = reward_batch + self.gamma * q_targets * (1 - done_batch)

            loss = self.criterion(q_values, target)
            self.optimizer_B.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.QB.parameters(), 1.0)
            self.optimizer_B.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_model(self):
        """method added for consistent api"""
        pass

    def save(self, filepath):
        torch.save(
            {
                "QA_state_dict": self.QA.state_dict(),
                "QB_state_dict": self.QB.state_dict(),
                "optimizer_A": self.optimizer_A.state_dict(),
                "optimizer_B": self.optimizer_B.state_dict(),
            },
            filepath,
        )

    def load(self, filepath=None, strat: Literal["different", "target", "model"] = "different"):
        "loads for finetuning"
        state = torch.load(filepath)
        strat_dct = {
            "different": ("model_state_dict", "target_model_state_dict"),
            "target": ("target_model_state_dict", "target_model_state_dict"),
            "model": ("model_state_dict", "model_state_dict"),
        }
        state_A, state_B = strat_dct[strat]
        self.QA.load_state_dict(state[state_A])
        self.QB.load_state_dict(state[state_B])

    def load_fine_tuned(self, filepath):
        state = torch.load(filepath)
        self.QA.load_state_dict(state["QA_state_dict"])
        self.QB.load_state_dict(state["QB_state_dict"])
