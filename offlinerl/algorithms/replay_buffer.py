"""
Implements the replay buffer class.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
from beartype.typing import Tuple


class ReplayBuffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size=int(1e6)):
        """Implement the replay buffer mechanism for both online and offline algorithms.

        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        :param max_size: number of transitions to store in the buffer, defaults to int(1e6)
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Store a new transition in the buffer, remove the oldest one if needed.

        :param state: current state
        :param action: action taken from the current state
        :param next_state: state returned by the environment after the action
        :param reward: reward resulting from taking action
        :param done: true if action leads to a terminal state
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Store a batch of transitions in the buffer, remove the oldest ones if needed.

        :param states: current states (shape: (batch_size, state_dim))
        :param actions: actions taken from the current states (shape: (batch_size, action_dim))
        :param next_states: states returned by the environment after taking actions (shape: (batch_size, state_dim))
        :param rewards: rewards resulting from taking actions (shape: (batch_size, 1))
        :param dones: flags indicating whether the actions lead to terminal states (shape: (batch_size, 1))
        """
        batch_size = states.shape[0]
        if batch_size == 0:
            return
        end_idx = (self.ptr + batch_size) % self.max_size

        if end_idx > self.ptr:
            # The batch fits without wrapping around
            indices = np.arange(self.ptr, end_idx)
        else:
            # The batch wraps around the buffer
            indices = np.concatenate((np.arange(self.ptr, self.max_size), np.arange(0, end_idx)))

        self._replace(
            indices,
            states,
            actions,
            next_states,
            rewards,
            1.0 - dones,
        )

        self.ptr = end_idx
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        """Sample batch_size transition from the replay buffer.

        :param batch_size: number of element in the sampled replay buffer
        :return: batch_size transitions
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def normalize_states(self, eps: float = 1e-3) -> Tuple[float]:
        """Normalize state in the replay buffer.

        TODO : understand what is the purpose of this function

        :param eps: _description_, defaults to 1e-3
        :return: _description_
        """
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def save(self, filepath: str) -> None:
        """Save replay buffer to file

        :param filepath: path where to save the file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> ReplayBuffer:
        """Load replay buffer from file.

        :param filepath: path of the file to load
        :return: replay buffer object
        """
        with open(filepath, "rb") as f:
            replay_buffer = pickle.load(f)
        return replay_buffer

    def _replace(
        self,
        indices: np.ndarray,
        new_states: np.ndarray,
        new_actions: np.ndarray,
        new_next_states: np.ndarray,
        new_rewards: np.ndarray,
        new_not_dones: np.ndarray,
    ) -> None:
        """Replace specified elements of the buffer with new transitions.

        :param indices: indices of the elements to replace
        :param new_states: new states to replace the existing ones
        :param new_actions: new actions to replace the existing ones
        :param new_next_states: new next states to replace the existing ones
        :param new_rewards: new rewards to replace the existing ones
        :param new_not_dones: new not-done flags to replace the existing ones
        """
        self.state[indices] = new_states
        self.action[indices] = new_actions
        self.next_state[indices] = new_next_states
        self.reward[indices] = new_rewards
        self.not_done[indices] = new_not_dones

    def replace_end(
        self,
        new_states: np.ndarray,
        new_actions: np.ndarray,
        new_next_states: np.ndarray,
        new_rewards: np.ndarray,
        new_not_dones: np.ndarray,
    ) -> None:
        """Replace elements at the end of the buffer with new transitions.

        :param new_states: new states to replace the existing ones
        :param new_actions: new actions to replace the existing ones
        :param new_next_states: new next states to replace the existing ones
        :param new_rewards: new rewards to replace the existing ones
        :param new_not_dones: new not-done flags to replace the existing ones
        """
        indices = np.arange(self.size - len(new_states), self.size)
        self._replace(indices, new_states, new_actions, new_next_states, new_rewards, new_not_dones)

    def replace_end_with_buffer(self, other_buffer: ReplayBuffer):
        self.replace_end(
            other_buffer.state[: other_buffer.size],
            other_buffer.action[: other_buffer.size],
            other_buffer.next_state[: other_buffer.size],
            other_buffer.reward[: other_buffer.size],
            other_buffer.not_done[: other_buffer.size],
        )
