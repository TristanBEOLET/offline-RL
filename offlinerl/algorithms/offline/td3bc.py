from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F
from beartype.typing import Dict, Tuple

from offlinerl.algorithms.online.td3 import TD3
from offlinerl.algorithms.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3_BC(TD3):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr: float = 3e-4,
        batch_size: int = 512,
        rl_alpha: float = 2.5,
        nb_batch_cloning: int = 10000,
        actor_hidden_dims: Tuple[int] = (256, 256),
        critic_hidden_dims: Tuple[int] = (256, 256),
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            lr=lr,
            batch_size=batch_size,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )
        self.rl_alpha = rl_alpha
        self.nb_batch_cloning = nb_batch_cloning
        self.current_batch_nb = 0
        self.alpha = None

    def train_step(self, replay_buffer: ReplayBuffer):
        if self.current_batch_nb < self.nb_batch_cloning:
            self.alpha = 0.0
        else:
            self.alpha = self.rl_alpha
        train_dict = super().train_step(replay_buffer)
        self.current_batch_nb += 1
        return train_dict

    def compute_actor_loss(  # pylint: disable=W0222
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pi = self.actor(state)  # pylint: disable=E1102
        Q = self.critic.Q1(state, pi)
        lmbda = self.alpha / Q.abs().mean().detach()

        actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
        return actor_loss

    @staticmethod
    def load(filename: str,  state_dim, action_dim, max_action, **agent_params) -> TD3_BC:
        """Load td3 from saved .pt file.

        :param filename: filename of the .pt file
        :param cfg: parameters to give to the constructor of the agent
        :param env: gymnasium environment
        :return: instantiated td3 object
        """
        td3 = TD3_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **agent_params,
        )

        checkpoint = torch.load(filename)
        td3.critic.load_state_dict(checkpoint["critic_state_dict"])
        td3.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        td3.actor.load_state_dict(checkpoint["actor_state_dict"])
        td3.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

        return td3
