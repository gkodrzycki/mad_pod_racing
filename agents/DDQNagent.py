import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from engine.game_sim import Boost, Normal, PodMovement, Shield
from engine.model import QNetwork
from engine.model_dueling import DuelingMLP
from engine.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from engine.util import check_point_radius, game_world_size
from engine.vec2 import Vec2, cross, dot, vec2ByAngle
from env import ACTIONS, discretize_state_runner_solo


class DDQNAgent:
    def __init__(
        self,
        model_type: Literal["standard", "dueling"] = "standard",
        state_dim=8,
        action_dim=len(ACTIONS),
        epsilon=0.9,
        alpha=1e-4,
        gamma=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_buffer_size=2**18,
        batch_size=64,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")  # Force CPU for compatibility

        self.pick_model = {"standard": QNetwork, "dueling": DuelingMLP}
        self.model = self.pick_model[model_type](state_dim, action_dim).to(self.device)
        self.target_model = self.pick_model[model_type](state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size

    def init(self):
        return self

    def forward(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values

    def act_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def act_non_greedy(self, state):

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def evaluate(self):
        self.model.eval()
        self.target_model.eval()

    def train(self):
        self.model.train()
        self.target_model.train()

    def run(self, inputs):
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

        state_batch = torch.from_numpy(np.array(batch[0])).float().to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.array(batch[3])).float().to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        target = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        print(f"Loading model from {filepath}")
        print(f"Model state dict keys: {checkpoint.keys()}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
