"""
Conservative Q learning implementation, original source code : https://github.com/young-geng/CQL
"""
from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from beartype.typing import Dict, Tuple
from torch import nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import TanhTransform

from offlinerl.algorithms.agent import Agent
from offlinerl.algorithms.replay_buffer import ReplayBuffer
from offlinerl.neural_networks.critic import TwinCritic
from offlinerl.neural_networks.mlp import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat) -> torch.Tensor:
    """
    Extend and repeat the given tensor along a specified dimension.

    :param tensor: The tensor to be extended and repeated.
    :param dim: The dimension along which to repeat the tensor.
    :param repeat: The number of times to repeat the tensor along the specified dimension.
    :return: The extended and repeated tensor.
    """
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(
    network: nn.Module, target_network: nn.Module, soft_target_update_rate: float
) -> None:
    """
    Perform a soft target update of the target network parameters.

    :param network: The source network whose parameters will be updated.
    :param target_network: The target network with parameters to be updated.
    :param soft_target_update_rate: The rate at which the target parameters are updated.
    """
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (1 - soft_target_update_rate) * target_network_params[
            k
        ].data + soft_target_update_rate * v.data


class Scalar(nn.Module):
    """
    Initialize a scalar parameter.

    :param init_value: The initial value of the scalar parameter.
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        """
        Compute the value of the scalar parameter.

        :return: The value of the scalar parameter.
        """
        return self.constant


class ReparameterizedTanhGaussian(nn.Module):
    """
    Initialize a reparameterized Tanh Gaussian distribution module.

    :param log_std_min: Minimum value for the log standard deviation.
    :param log_std_max: Maximum value for the log standard deviation.
    :param no_tanh: If True, no Tanh transformation is applied to the actions.
    """

    def __init__(self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log probability of a sample under the distribution.

        :param mean: Mean of the distribution.
        :param log_std: Log standard deviation of the distribution.
        :param sample: Sample for which to compute the log probability.
        :return: The log probability of the sample under the distribution.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
            sample = sample - (1e-6) * torch.sign(sample)

        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate an action sample from the distribution.

        :param mean: Mean of the distribution.
        :param log_std: Log standard deviation of the distribution.
        :param deterministic: If True, return the mean of the distribution as the action.
        :return: A tuple containing the action sample and its log probability.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        depth: int = 2,
        hidden_dim: int = 256,
        log_std_multiplier=1.0,
        log_std_offset=-1.0,
        orthogonal_init=False,
        no_tanh=False,
    ):
        """
        Initialize a policy network that parameterizes a Tanh Gaussian distribution.

        :param observation_dim: Dimension of the observation space.
        :param action_dim: Dimension of the action space.
        :param depth: Depth of the neural network.
        :param hidden_dim: Dimension of hidden layers in the neural network.
        :param log_std_multiplier: Multiplier applied to the log standard deviation.
        :param log_std_offset: Offset added to the log standard deviation.
        :param orthogonal_init: If True, use orthogonal initialization for network weights.
        :param no_tanh: If True, no Tanh transformation is applied to the actions.
        """
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.base_network = MLP(
            observation_dim,
            [hidden_dim for _ in range(depth)],
            2 * action_dim,
            ["relu" for _ in range(depth)] + [None],
        )

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        """
        Compute the log probability of actions under the policy distribution.

        :param observations: Input observations.
        :param actions: Actions for which to compute the log probability.
        :return: The log probability of the actions under the policy distribution.
        """
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        """
        Generate action samples from the policy distribution.

        :param observations: Input observations.
        :param deterministic: If True, return mean of the distribution as the action.
        :param repeat: If not None, repeat the generated actions multiple times.
        :return: A tuple containing the action samples and their log probabilities.
        """
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class CQL(Agent):
    """
    Implementation of the CQL (Conservative Q-Learning) algorithm.

    CQL combines Q-learning with a conservative objective to improve stability and robustness in reinforcement learning.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        nb_batch_cloning: int = 2e4,
        batch_size: int = 256,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        policy_hidden_dim: int = 256,
        policy_depth: int = 2,
        critic_hidden_dim: int = 256,
        critic_depth: int = 2,
        use_automatic_entropy_tuning: bool = True,
        target_entropy: float = 0.0,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        soft_target_update_rate: float = 5e-3,
        target_update_period: float = 1,
        policy_log_std_multiplier: float = 1.0,
        policy_log_std_offset: float = -1.0,
        use_cql: bool = True,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = 1.0,
        cql_temp: float = 1.0,
        cql_min_q_weight: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        orthogonal_init: bool = False,
        backup_entropy: bool = False,
        no_tanh: bool = False,
    ):
        """
        Initialize the CQL algorithm.

        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param max_action: Maximum value of actions in the action space.
        :param nb_batch_cloning: Number of batch cloning steps for policy learning.
        :param batch_size: Batch size for training.
        :param discount: Discount factor for future rewards.
        :param alpha_multiplier: SScalingcaling factor for the entropy regularization.
        :param policy_hidden_dim: Hidden dimension for policy network.
        :param policy_depth: Depth of policy network.
        :param critic_hidden_dim: Hidden dimension for critic network.
        :param critic_depth: Depth of critic network.
        :param use_automatic_entropy_tuning: Flag for using automatic entropy tuning.
        :param target_entropy: Target entropy for entropy regularization.
        :param policy_lr: Learning rate for policy optimizer.
        :param critic_lr: Learning rate for critic optimizer.
        :param soft_target_update_rate: Rate for soft target network updates.
        :param target_update_period: Number of steps between target network updates.
        :param policy_log_std_multiplier: Multiplier for policy's log standard deviation.
        :param policy_log_std_offset: Offset for policy's log standard deviation.
        :param use_cql: Flag for using CQL.
        :param cql_n_actions: Number of actions for CQL importance sampling.
        :param cql_importance_sample: Flag for using importance sampling in CQL.
        :param cql_lagrange: Flag for using Lagrange multiplier for CQL.
        :param cql_target_action_gap: Target action gap for CQL.
        :param cql_temp: Temperature parameter for CQL.
        :param cql_min_q_weight: Weight for CQL minimum Q-value loss.
        :param cql_max_target_backup: Flag for using maximum target backup in CQL.
        :param cql_clip_diff_min: Minimum value for clipping CQL differences.
        :param cql_clip_diff_max: Maximum value for clipping CQL differences.
        :param orthogonal_init: Flag for using orthogonal initialization in policy network.
        :param backup_entropy: Flag for backing up entropy in Q-values for CQL.
        :param no_tanh: Flag for disabling tanh activation in policy network.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.cql_lagrange = cql_lagrange  # bool
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.cql_n_actions = cql_n_actions
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_temp = cql_temp
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.use_cql = use_cql
        self.target_update_period = target_update_period
        self.soft_target_update_rate = soft_target_update_rate
        self.batch_size = batch_size
        self.nb_batch_cloning = nb_batch_cloning
        self.backup_entropy = backup_entropy
        self.no_tanh = no_tanh

        self.policy = TanhGaussianPolicy(
            state_dim,
            action_dim,
            hidden_dim=policy_hidden_dim,
            depth=policy_depth,
            log_std_multiplier=policy_log_std_multiplier,
            log_std_offset=policy_log_std_offset,
            orthogonal_init=orthogonal_init,
            no_tanh=self.no_tanh,
        ).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            self.policy_lr,
        )

        self.critic = TwinCritic(
            state_dim, action_dim, hidden_dims=[critic_hidden_dim for _ in range(critic_depth)]
        ).to(device)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.critic_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.critic.q1, self.target_critic.q1, soft_target_update_rate)
        soft_target_update(self.critic.q2, self.target_critic.q2, soft_target_update_rate)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action using the CQL policy.

        :param state: Input state tensor.
        :return: Selected action tensor.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = (
            self.policy(state, deterministic=True)[0]  # pylint: disable=E1102
            .cpu()
            .data.numpy()
            .flatten()
        )
        return action.clip(-self.max_action, self.max_action)

    def train_step(self, replay_buffer: ReplayBuffer):
        """
        Perform a training step of the CQL algorithm.

        :param replay_buffer: Replay buffer containing collected experiences.
        :return: Dictionary of training metrics, including:
            - 'log_pi': Mean log probability of selected actions.
            - 'policy_loss': Loss of the policy network.
            - 'qf1_loss': Loss of the first critic network.
            - 'qf2_loss': Loss of the second critic network.
            - 'alpha_loss': Loss of the entropy tuning parameter.
            - 'alpha': Current value of the entropy tuning parameter.
            - 'average_qf1': Average Q-value estimate from the first critic network.
            - 'average_qf2': Average Q-value estimate from the second critic network.
            - 'average_target_q': Average target Q-value.
            - 'total_steps': Total number of training steps.
        """
        observations, actions, next_observations, rewards, not_dones = replay_buffer.sample(
            self.batch_size
        )
        dones = 1.0 - not_dones

        self._total_steps += 1

        new_actions, log_pi = self.policy(observations)  # pylint: disable=E1102

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)

        # Policy loss
        if self.total_steps < self.nb_batch_cloning:
            log_probs = self.policy.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(torch.hstack(self.target_critic(observations, new_actions)))
            policy_loss = (alpha * log_pi - q_new_actions).mean()

        # Q function loss
        q1_pred, q2_pred = self.critic(observations, actions)  # pylint: disable=E1102

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(  # pylint: disable=E1102
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    torch.hstack(self.target_critic(next_observations, new_next_actions)), dim=1
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(
                -1
            )
        else:
            new_next_actions, next_log_pi = self.policy(next_observations)  # pylint: disable=E1102
            target_q_values = torch.min(
                torch.hstack(self.target_critic(next_observations, new_next_actions))
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards + (1.0 - dones) * self.discount * target_q_values
        q1_loss = F.mse_loss(q1_pred, td_target.detach())
        q2_loss = F.mse_loss(q2_pred, td_target.detach())

        ### CQL
        if not self.use_cql:
            qf_loss = q1_loss + q2_loss
        else:
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            cql_random_actions = actions.new_empty(
                (batch_size, self.cql_n_actions, action_dim), requires_grad=False
            ).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(  # pylint: disable=E1102
                observations, repeat=self.cql_n_actions
            )
            cql_next_actions, cql_next_log_pis = self.policy(  # pylint: disable=E1102
                next_observations, repeat=self.cql_n_actions
            )
            cql_current_actions, cql_current_log_pis = (
                cql_current_actions.detach(),
                cql_current_log_pis.detach(),
            )
            cql_next_actions, cql_next_log_pis = (
                cql_next_actions.detach(),
                cql_next_log_pis.detach(),
            )

            observations_interleaved = extend_and_repeat(
                observations, 1, cql_random_actions.shape[1]
            ).reshape(-1, observations.shape[-1])
            cql_random_actions_reshaped = cql_random_actions.reshape(
                -1, cql_random_actions.shape[-1]
            )
            cql_current_actions_reshaped = cql_current_actions.reshape(
                -1, cql_current_actions.shape[-1]
            )
            cql_next_actions_reshaped = cql_next_actions.reshape(-1, cql_next_actions.shape[-1])

            cql_q1_rand = self.critic.Q1(
                observations_interleaved, cql_random_actions_reshaped
            ).reshape(batch_size, -1)
            cql_q2_rand = self.critic.Q2(
                observations_interleaved, cql_random_actions_reshaped
            ).reshape(batch_size, -1)

            # cql_q1_rand = self.critic.q1(observations, cql_random_actions)
            # cql_q2_rand = self.critic.q2(observations, cql_random_actions)

            cql_q1_current_actions = self.critic.Q1(
                observations_interleaved, cql_current_actions_reshaped
            ).reshape(batch_size, -1)
            cql_q2_current_actions = self.critic.Q2(
                observations_interleaved, cql_current_actions_reshaped
            ).reshape(batch_size, -1)
            cql_q1_next_actions = self.critic.Q1(
                observations_interleaved, cql_next_actions_reshaped
            ).reshape(batch_size, -1)
            cql_q2_next_actions = self.critic.Q2(
                observations_interleaved, cql_next_actions_reshaped
            ).reshape(batch_size, -1)

            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand,
                    q1_pred,
                    cql_q1_next_actions,
                    cql_q1_current_actions,
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand,
                    q2_pred,
                    cql_q2_next_actions,
                    cql_q2_current_actions,
                ],
                dim=1,
            )

            if self.cql_importance_sample:
                random_density = np.log(0.5**action_dim)
                cql_cat_q1 = torch.cat(
                    [
                        cql_q1_rand - random_density,
                        cql_q1_next_actions - cql_next_log_pis.detach(),
                        cql_q1_current_actions - cql_current_log_pis.detach(),
                    ],
                    dim=1,
                )
                cql_cat_q2 = torch.cat(
                    [
                        cql_q2_rand - random_density,
                        cql_q2_next_actions - cql_next_log_pis.detach(),
                        cql_q2_current_actions - cql_current_log_pis.detach(),
                    ],
                    dim=1,
                )

            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

            # Subtract the log likelihood of data
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - q1_pred,
                self.cql_clip_diff_min,
                self.cql_clip_diff_max,
            ).mean()
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - q2_pred,
                self.cql_clip_diff_min,
                self.cql_clip_diff_max,
            ).mean()

            if self.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_min_q_weight
                    * (cql_qf1_diff - self.cql_target_action_gap)
                )
                cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_min_q_weight
                    * (cql_qf2_diff - self.cql_target_action_gap)
                )

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)

            qf_loss = q1_loss + q2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        if self.total_steps % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=q1_loss.item(),
            qf2_loss=q2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        return metrics

    @property
    def total_steps(self):
        """
        Get the total number of training steps performed by the CQL agent.

        :return: Total number of training steps.
        """
        return self._total_steps

    def save(self, filename: str):
        """
        Save the CQL agent's model to a file.

        :param filename: Filename for the saved model.
        """

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )

    @staticmethod
    def load(filename: str,  state_dim, action_dim, max_action, **agent_params) -> CQL:
        """
        Load a saved CQL agent from a file.

        :param filename: Filename of the saved model.
        :param agent_params: Dictionary of agent parameters.
        :param env: Gym environment.
        :return: Instantiated CQL agent.
        """
        cql = CQL(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **agent_params,
        )

        checkpoint = torch.load(filename)
        cql.policy.load_state_dict(checkpoint["policy_state_dict"])
        cql.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        cql.critic.load_state_dict(checkpoint["critic_state_dict"])
        cql.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        cql.target_critic = deepcopy(cql.critic)

        return cql
