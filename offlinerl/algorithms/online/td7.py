from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype.typing import Callable, Dict

from offlinerl.algorithms.agent import Agent
from offlinerl.algorithms.lap_replay_buffer import LAP
from offlinerl.neural_networks.activations import AvgL1Norm
from offlinerl.neural_networks.actors import Actor_TD7
from offlinerl.neural_networks.critic import Critic_TD7


def LAP_huber(x, min_priority: int = 1) -> torch.Tensor:
    """
    Calculate the LAP (Locally Approximate Prioritization) loss with the Huber loss function.

    :param x: Input tensor.
    :param min_priority: Minimum priority to prevent extreme prioritization.
    :return: Loss value.
    """
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Encoder(nn.Module):  # pylint: disable=W0223
    """
    Neural network model for encoding state and state-action pairs.

    :param state_dim: Dimension of the state.
    :param action_dim: Dimension of the action.
    :param zs_dim: Dimension of the latent state.
    :param hdim: Hidden layer dimension.
    :param activ: Activation function.
    """

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the state to a latent representation.

        :param state: Input state.
        :return: Latent representation of the state.
        """
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Encode the latent state and action to a latent representation.

        :param zs: Latent state.
        :param action: Action.
        :return: Latent representation of the state-action pair.
        """
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class TD7(Agent):
    """
    Implementation of the TD7 algorithm.

    :param state_dim: Dimension of the state.
    :param action_dim: Dimension of the action.
    :param max_action: Maximum value of an action.
    :param offline: Whether the algorithm is used in an offline setting.
    :param batch_size: Batch size for training.
    :param discount: Discount factor.
    :param target_update_rate: Rate at which target networks are updated.
    :param exploration_noise: Exploration noise factor.
    :param target_policy_noise: Noise factor for target policy smoothing.
    :param noise_clip: Value to clip noise.
    :param policy_freq: Frequency of policy updates.
    :param alpha: Parameter for prioritized replay.
    :param min_priority: Minimum priority for prioritized replay.
    :param lmbda: Lambda parameter for TD3+BC.
    :param max_eps_when_checkpointing: Maximum episodes before checkpointing.
    :param steps_before_checkpointing: Steps before checkpointing.
    :param reset_weight: Reset weight for checkpointing.
    :param zs_dim: Dimension of the latent state.
    :param enc_hdim: Dimension of the encoder's hidden layers.
    :param enc_activ: Activation function for the encoder.
    :param encoder_lr: Learning rate for the encoder.
    :param critic_hdim: Dimension of the critic's hidden layers.
    :param critic_activ: Activation function for the critic.
    :param critic_lr: Learning rate for the critic.
    :param actor_hdim: Dimension of the actor's hidden layers.
    :param actor_activ: Activation function for the actor.
    :param actor_lr: Learning rate for the actor.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        offline: bool = False,
        # Generic
        batch_size: int = 256,
        discount: float = 0.99,
        target_update_rate: int = 250,
        exploration_noise: float = 0.1,
        # TD3
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        # LAP
        alpha: float = 0.4,
        min_priority: float = 1,
        # TD3+BC
        lmbda: float = 0.1,
        # Checkpointing
        max_eps_when_checkpointing: int = 20,
        steps_before_checkpointing: float = 75e4,
        reset_weight: float = 0.9,
        # Encoder Model
        zs_dim: int = 256,
        enc_hdim: int = 256,
        enc_activ: Callable = F.elu,
        encoder_lr: float = 3e-4,
        # Critic Model
        critic_hdim: int = 256,
        critic_activ: Callable = F.elu,
        critic_lr: float = 3e-4,
        # Actor Model
        actor_hdim: int = 256,
        actor_activ: Callable = F.relu,
        actor_lr: float = 3e-4,
    ):
        self.batch_size = batch_size
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.exploration_noise = exploration_noise
        self.target_policy_noise = target_policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.min_priority = min_priority
        self.lmbda = lmbda
        self.max_eps_when_checkpointing = max_eps_when_checkpointing
        self.steps_before_checkpointing = steps_before_checkpointing
        self.reset_weight = reset_weight
        self.zs_dim = zs_dim
        self.enc_hdim = enc_hdim
        self.enc_activ = enc_activ
        self.encoder_lr = encoder_lr
        self.critic_hdim = critic_hdim
        self.critic_activ = critic_activ
        self.critic_lr = critic_lr
        self.actor_hdim = actor_hdim
        self.actor_activ = actor_activ
        self.actor_lr = actor_lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor_TD7(state_dim, action_dim, zs_dim, actor_hdim, actor_activ).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic_TD7(state_dim, action_dim, zs_dim, critic_hdim, critic_activ).to(
            self.device
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, zs_dim, enc_hdim, enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.max_action = max_action
        self.offline = offline

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

    def select_action(
        self, state: np.ndarray, use_checkpoint: bool = False, use_exploration: bool = False
    ):
        """
        Select an action based on the current policy.

        :param state: Current state.
        :param use_checkpoint: Whether to use the checkpointed model.
        :param use_exploration: Whether to add exploration noise.
        :return: Selected action.
        """
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)  # pylint: disable=E1102

            if use_exploration:
                action = action + torch.randn_like(action) * self.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def compute_actor_loss(
        self, Q: torch.Tensor, action: torch.Tensor, actor: torch.Tensor  # pylint: disable=W0613
    ) -> torch.Tensor:
        """
        Compute the actor loss based on the Q-values, actions, and actor outputs.

        :param Q: Q-values from the critic.
        :param action: Selected actions.
        :param actor: Actor outputs.
        :return: Actor loss.
        """
        return -Q.mean()

    def train(self, replay_buffer: LAP) -> Dict:
        """
        Perform a training step using the given replay buffer.

        :param replay_buffer: Replay buffer containing experience samples.
        :return: Dictionary of computed losses.
        """
        self.training_steps += 1

        state, action, next_state, reward, not_done, indices = replay_buffer.sample(self.batch_size)

        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        #########################
        # Update Critic
        #########################
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs
            ).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.discount * Q_target.clamp(
                self.min_target, self.max_target
            )

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)  # pylint: disable=E1102
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #########################
        # Update LAP
        #########################
        priority = td_loss.max(1)[0].clamp(min=self.min_priority).pow(self.alpha)
        replay_buffer.update_priority(indices, priority)

        #########################
        # Update Actor
        #########################
        if self.training_steps % self.policy_freq == 0:
            actor = self.actor(state, fixed_zs)  # pylint: disable=E1102
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)  # pylint: disable=E1102

            actor_loss = self.compute_actor_loss(Q=Q, action=action, actor=actor)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        loss_dict = {"encoder_loss": encoder_loss.item(), "critic_loss": critic_loss.item()}
        if self.training_steps % self.policy_freq == 0:
            loss_dict["actor_loss"] = actor_loss.item()
        else:
            loss_dict["actor_loss"] = np.nan

        return loss_dict

    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, replay_buffer: LAP, ep_timesteps: int, ep_return: float):
        """
        Perform training and checkpointing operations based on episode information.

        :param replay_buffer: Replay buffer containing experience samples.
        :param ep_timesteps: Number of timesteps in the episode.
        :param ep_return: Return achieved in the episode.
        """
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset(replay_buffer)

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset(replay_buffer)

    # Batch training
    def train_and_reset(self, replay_buffer: LAP):
        """
        Perform training and reset relevant variables.

        :param replay_buffer: Replay buffer containing experience samples.
        """
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_when_checkpointing

            self.train(replay_buffer)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, filename: str):
        """
        Save the model's state to a file.

        :param filename: Name of the file to save the model state to.
        """

        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
            },
            filename,
        )

    @staticmethod
    def load(filename: str, agent_params: Dict, env: gym.Env) -> TD7:
        """Load td7 from saved .pt file.

        :param filename: filename of the .pt file
        :param env: gymnasium environment
        :return: instantiated td7 object
        """
        # More robust than env.observation_space.shape if the env is not well defined
        state_dim = env.reset()[0].shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        td7 = TD7(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **agent_params,
        )

        checkpoint = torch.load(filename)
        td7.critic.load_state_dict(checkpoint["critic_state_dict"])
        td7.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        td7.actor.load_state_dict(checkpoint["actor_state_dict"])
        td7.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        td7.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        td7.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])

        return td7
