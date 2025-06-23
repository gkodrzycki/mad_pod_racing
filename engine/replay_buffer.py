import random
from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=np.int32)
        self.data_pointer = 0

    def add(self, priority, data_index):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data_index
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def batch_update(self, idxs, priorities):
        idxs = np.asarray(idxs, dtype=np.int32)
        priorities = np.asarray(priorities, dtype=np.float32)

        # Set new priorities
        leaf_idxs = idxs
        self.tree[leaf_idxs] = priorities

        # Propagate up the tree
        parents = ((leaf_idxs - 1) // 2).astype(np.int32)
        unique_parents = np.unique(parents)

        while unique_parents.size > 0:
            self.tree[unique_parents] = self.tree[2 * unique_parents + 1] + self.tree[2 * unique_parents + 2]
            unique_parents = ((unique_parents - 1) // 2).astype(np.int32)
            unique_parents = np.unique(unique_parents[unique_parents >= 0])

    def get_leaf(self, v):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, device="cpu", alpha=0.6):
        self.cnt = 0
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.capacity = capacity
        self.device = device
        self.position = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0

        i = self.position % self.capacity
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done

        self.tree.add(max_priority, i)
        self.position += 1

    def sample(self, batch_size, beta=0.4):
        batch = []
        batch_indices = []
        tree_indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            # print(f"segment: {segment}, i: {i}")
            a = segment * i
            b = segment * (i + 1)
            # print(f"Sampling range: [{a}, {b})")
            s = np.random.uniform(a, b)
            tree_idx, p, data_idx = self.tree.get_leaf(s)
            batch.append(
                (
                    self.states[data_idx],
                    self.actions[data_idx],
                    self.rewards[data_idx],
                    self.next_states[data_idx],
                    self.dones[data_idx],
                )
            )
            tree_indices.append(tree_idx)
            batch_indices.append(data_idx)
            priorities.append(p)

        probs = np.array(priorities) / self.tree.total_priority
        probs = np.clip(probs, 1e-8, 1.0)  # avoid 0
        # print(f"Probabilities: {probs}")
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max()

        # print(f"Sampled batch size: {len(batch)}, Batch indices: {batch_indices}, Weights: {weights}")

        batch_indices = np.array(batch_indices)

        return batch, batch_indices, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.array(priorities, dtype=np.float32)

        # Replace inf/nan with finite default (e.g. 1.0)
        self.cnt += np.sum(np.isinf(priorities) | np.isnan(priorities))
        # print(f"Number of bad priorities encountered: {self.cnt}")
        priorities = np.nan_to_num(priorities, nan=1.0, posinf=1.0, neginf=1.0)

        priorities = np.power(np.abs(priorities) + self.epsilon, self.alpha)
        self.tree.batch_update(np.array(idxs), priorities)

    def __len__(self):
        return min(self.position, self.capacity)
