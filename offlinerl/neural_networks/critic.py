import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype.typing import List, Tuple

from offlinerl.neural_networks.activations import AvgL1Norm
from offlinerl.neural_networks.q_network import QNetwork


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """Implements the Critic module of TD3 (twin critics Q1 and Q2).

        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.observation_space.shape[0]
        """
        super(TwinCritic, self).__init__()

        # Double q networks
        self.q1 = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
        self.q2 = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the Q1(s,a).

        :param state: current state s
        :param action: action a
        :return: Q1(s,a)
        """
        state_action = torch.cat([state, action], 1)
        return self.q1(state_action)

    def Q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the Q1(s,a).

        :param state: current state s
        :param action: action a
        :return: Q1(s,a)
        """
        state_action = torch.cat([state, action], 1)
        return self.q2(state_action)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute the Q1(s,a) and Q2(s,a).

        :param state: current state s
        :param action: action a
        :return: Q1(s,a), Q2(s,a)
        """
        state_action = torch.cat([state, action], 1)
        return self.q1(state_action), self.q2(state_action)


class Critic_TD7(nn.Module):
    """
    Implementation of the critic neural network for the TD7 algorithm.

    The critic network computes Q-values for state-action pairs and latent embeddings.
    """

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        """
        Initialize the Critic_TD7 neural network.

        :param state_dim: Dimension of the input state.
        :param action_dim: Dimension of the input action.
        :param zs_dim: Dimension of the latent state.
        :param hdim: Dimension of the hidden layers.
        :param activ: Activation function for hidden layers.
        """
        super(Critic_TD7, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        """
        Forward pass of the critic neural network.

        :param state: Input state.
        :param action: Input action.
        :param zsa: Latent state-action embedding.
        :param zs: Latent state embedding.
        :return: Computed Q-values.
        """
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)
