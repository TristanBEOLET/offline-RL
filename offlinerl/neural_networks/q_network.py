import torch
import torch.nn.functional as F
from beartype.typing import List

from offlinerl.neural_networks.mlp import MLP


class QNetwork(MLP):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """Implements the Actor module of TD3.

        With env a gymnasium environment :
        :param state_dim: env.observation_space.shape[0]
        :param action_dim: env.action_space.shape[0]
        """
        super(QNetwork, self).__init__(
            input_dim=state_dim + action_dim, hidden_dims=hidden_dims, output_dim=1
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0237
        """Compute state action value

        :param state_action: concatenation of state and action : [s,a]
        :return: Q(s,a)
        """
        q1 = F.relu(self.layers[0](state_action))
        for layer in self.layers[1:-1]:
            q1 = F.relu(layer(q1))
        return self.layers[-1](q1)
