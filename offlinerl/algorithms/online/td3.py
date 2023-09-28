"""
Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
Paper: https://arxiv.org/abs/1802.09477


This implementation is just un adapted version of https://github.com/sfujim/TD3/ to
match the project needs.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from beartype.typing import Dict, Optional, Tuple

from offlinerl.algorithms.agent import Agent
from offlinerl.algorithms.replay_buffer import ReplayBuffer
from offlinerl.neural_networks.actors import Actor_TD3
from offlinerl.neural_networks.critic import TwinCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(Agent):
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
        expl_noise: float = 0.1,
        actor_hidden_dims: Tuple[int] = (256, 256),
        critic_hidden_dims: Tuple[int] = (256, 256),
    ):
        """Implements the td3 algorithm

        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.observation_space.shape[0]
        :param max_action: float(env.action_space.high[0])
        :param discount: discount factor gamma, defaults to 0.99
        :param tau: soft averaging coeff, defaults to 0.005
        :param policy_noise: noise factor, defaults to 0.2
        :param noise_clip: clip value for noise, defaults to 0.5
        :param policy_freq: delayed policy updates, soft update every policy_freq itgerations,
                                defaults to 2
        :param lr: learning rate, defaults to 3e-4
        :param batch_size: batch size, defaults to 512
        :param expl_noise: exploration noise, defaults to 512
        """
        self.actor = Actor_TD3(state_dim, action_dim, max_action, hidden_dims=list(actor_hidden_dims)).to(
            device
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dims=list(critic_hidden_dims)).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.expl_noise = expl_noise

        self.total_it = 0

    def select_action(self, state: np.ndarray, is_eval: Optional[bool] = True) -> np.ndarray:
        """Select action according to current policy

        :param state: current state
        :return: action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()  # pylint: disable=E1102
        if is_eval:
            return action
        else:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            return (action + noise).clip(-self.max_action, self.max_action)

    def compute_actor_loss(  # pylint: disable=W0613
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> torch.Tensor:
        return -self.critic.Q1(state, self.actor(state)).mean()  # pylint: disable=E1102

    def train_step(self, replay_buffer: ReplayBuffer):
        """Train on one batch.

        :param replay_buffer: replay buffer (offline dataset)
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)  # pylint: disable=E1102

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = self.compute_actor_loss(action=action, state=state)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
        return {"actor_loss": np.nan, "critic_loss": critic_loss.item()}

    def save(self, filename: str):
        """Save td3 model.

        :param filename: filename for the .pt file containing all the networks
        """

        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            },
            filename,
        )

    @staticmethod
    def load(filename: str, state_dim, action_dim, max_action, **agent_params) -> TD3:
        """Load td3 from saved .pt file.

        :param filename: filename of the .pt file
        :param cfg: parameters to give to the constructor of the agent
        :param env: gymnasium environment
        :return: instantiated td3 object
        """
        td3 = TD3(
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
