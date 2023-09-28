"""
Define an abstract class for agents. This template works for both online and offline agents.
"""


from abc import ABC, abstractmethod

import numpy as np

from offlinerl.algorithms.replay_buffer import ReplayBuffer


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass
