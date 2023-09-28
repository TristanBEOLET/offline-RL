from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerl.neural_networks.activations import AvgL1Norm
from offlinerl.neural_networks.mlp import MLP


class Actor_TD3(MLP):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dims: Tuple[int] = (256, 256),
    ):
        """Implements the Actor module of TD3.

        With env a gymnasium environment :
        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        :param max_action: float(env.action_space.high[0])
        """
        super(Actor_TD3, self).__init__(
            input_dim=state_dim, hidden_dims=hidden_dims, output_dim=action_dim
        )
        self.max_action = max_action

    def forward(self, state):  # pylint: disable=W0237
        a = F.relu(self.layers[0](state))
        a = F.relu(self.layers[1](a))
        return self.max_action * torch.tanh(self.layers[-1](a))
        # return self.max_action * torch.clip(self.layers[-1](a), -1.0, 1.0)


class Actor_BCQ(MLP):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, phi: float = 0.05):
        """Implements the actor part of the BCQ.

        With env a gymnasium environment :
        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        :param max_action: float(env.action_space.high[0])
        :param phi: controls exploration - exploitation, defaults to 0.05
        """
        super(Actor_BCQ, self).__init__(
            input_dim=state_dim + action_dim, hidden_dims=[400, 300], output_dim=action_dim
        )

        self.max_action = max_action
        self.phi = phi

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute ???.

        :param state: current state s
        :param action: action a
        :return: action
        """
        a = F.relu(self.layers[0](torch.cat([state, action], 1)))
        a = F.relu(self.layers[1](a))
        a = self.phi * self.max_action * torch.tanh(self.layers[-1](a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Actor_TD7(nn.Module):
    """
    Implementation of the actor neural network for the TD7 algorithm.
    """

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        """
        Initialize the Actor_TD7 neural network.

        :param state_dim: Dimension of the input state.
        :param action_dim: Dimension of the output action.
        :param zs_dim: Dimension of the latent state.
        :param hdim: Dimension of the hidden layers.
        :param activ: Activation function for hidden layers.
        """
        super(Actor_TD7, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        """
        Forward pass of the actor neural network.

        :param state: Input state.
        :param zs: Latent state.
        :return: Computed action.
        """
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))
