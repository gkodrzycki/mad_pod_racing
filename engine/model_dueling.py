import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

        self.value_stream = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv_stream = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
