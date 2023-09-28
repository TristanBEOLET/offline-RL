from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerl.algorithms.agent import Agent
from offlinerl.algorithms.replay_buffer import ReplayBuffer
from offlinerl.neural_networks.actors import Actor_BCQ
from offlinerl.neural_networks.critic import TwinCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, latent_dim: int, max_action: float, device: str
    ):
        """Implements the VAE model for BCQ

        With env a gymnasium environment :
        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        :param latent_dim: dimension of the latent space of the VAE
        :param max_action: float(env.action_space.high[0])
        :param device: device, either "cuda" or "cpu"
        """
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Generate fake actions.

        :param state: current state s
        :param action: action a
        :return:
        """
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state: torch.Tensor, z=None) -> torch.Tensor:
        """Generate a synthetic plausible perturbed action.

        :param state: current state s
        :param z: latent vector, defaults to None
        :return: synthetic action
        """
        # When sampling (z is not given) from the VAE, the latent vector is clipped to [-0.5, 0.5]
        # Clipping the latent vectors ensures that the generated actions remain within
        # a reasonable and meaningful range for the given environment
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class BCQ(Agent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        lmbda: float = 0.75,
        phi: float = 0.05,
        lr: float = 1e-3,
        nb_perturbed_state_action_selection: int = 100,
        nb_repeat_vae: int = 10,
        policy_freq: int = 2,
        batch_size: int = 512,
    ):
        """Implements the BCQ algorithm.

        With env a gymnasium environment :
        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        :param max_action: float(env.action_space.high[0])
        :param device: device, either "cuda" or "cpu"
        :param discount: discount factor gamma, defaults to 0.99
        :param tau: soft averaging coeff, defaults to 0.005
        :param lmbda: Ration between the behavioral policy and the conservative policy,
                        defaults to 0.75
        :param phi: controls exploration - exploitation, defaults to 0.05
        """
        latent_dim = action_dim * 2

        self.actor = Actor_BCQ(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinCritic(
            state_dim=state_dim, action_dim=action_dim, hidden_dims=[400, 300]
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.nb_perturbed_state_action_selection = nb_perturbed_state_action_selection
        self.nb_repeat_vae = nb_repeat_vae
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.total_it = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action according to current policy

        :param state: current state
        :return: action
        """
        with torch.no_grad():
            # Generating a batch of perturbed states with the VAE
            state = (
                torch.FloatTensor(state.reshape(1, -1))
                .repeat(self.nb_perturbed_state_action_selection, 1)
                .to(self.device)
            )
            perturbed_states = self.vae.decode(state)

            # Computing the Q values values for each perturbed state
            action = self.actor(state, perturbed_states)  # pylint: disable=E1102
            q1 = self.critic.Q1(state, action)

            # If the actions produced by the VAE are closed to the one in the dataset,
            # they will have a high Q value -> this way make sure to stick an state/action
            # that is close to one present in the dataset
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def train_step(self, replay_buffer: ReplayBuffer):
        """Train on one batch.

        :param replay_buffer: replay buffer (offline dataset)
        :param iterations: Number of iterations
        """
        self.total_it += 1

        # Sample replay buffer / batch
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action)  # pylint: disable=E1102
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic Training, same mechanism as for action selection : we compute the action
        # to take for multiple perturbed states and keep the one with the highest Q value.
        with torch.no_grad():
            # Duplicate next state nb_repeat_vae times
            # For example for a batch of 2 states [1, 2, 3] and [4,5,6]
            # >>> x = torch.tensor([[1, 2, 3],[4,5,6]])
            # >>> torch.repeat_interleave(x, 3, 0)
            # tensor([[1, 2, 3],
            #         [1, 2, 3],
            #         [1, 2, 3],
            #         [4, 5, 6],
            #         [4, 5, 6],
            #         [4, 5, 6]])

            next_state = torch.repeat_interleave(next_state, self.nb_repeat_vae, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(
                next_state, self.actor_target(next_state, self.vae.decode(next_state))
            )

            # Soft Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (
                1.0 - self.lmbda
            ) * torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)

            target_Q = reward + not_done * self.discount * target_Q

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            current_Q1, current_Q2 = self.critic(state, action)  # pylint: disable=E1102
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)  # pylint: disable=E1102

            # Update through DPG
            actor_loss = -self.critic.Q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
        return {"actor_loss": np.nan, "critic_loss": np.nan}

    def save(self, filename):
        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "vae_state_dict": self.vae.state_dict(),
                "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            },
            filename,
        )

    @staticmethod
    def load(filename, state_dim, action_dim, max_action, **agent_params) -> BCQ:
        bcq = BCQ(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **agent_params,
        )

        checkpoint = torch.load(filename)
        bcq.critic.load_state_dict(checkpoint["critic_state_dict"])
        bcq.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        bcq.critic_target = copy.deepcopy(bcq.critic)
        bcq.actor.load_state_dict(checkpoint["actor_state_dict"])
        bcq.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        bcq.actor_target = copy.deepcopy(bcq.actor)
        bcq.vae.load_state_dict(checkpoint["vae_state_dict"])
        bcq.vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])

        return bcq
