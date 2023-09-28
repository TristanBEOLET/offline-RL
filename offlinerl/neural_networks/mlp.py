"""
MLP model implemented using torch.
"""
import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import List, Optional, Union


class MLP(nn.Module):
    @beartype
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        activations: Optional[Union[str, List[Union[str, None]]]] = None,
    ):
        """
        Multi-Layer Perceptron (MLP) class.

        :param input_dim: Number of features in the input.
        :param hidden_dims: List of integers specifying the number of units in each hidden
         layer.
        :param output_dim: Number of output classes or units.
        :param activations: List of activation function names for each hidden layer. If not
         provided, no activation functions will be added.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activations = activations

        layers = []

        if hidden_dims:
            # Dealing with activation functions
            if isinstance(activations, str):
                self.activations = [activations for _ in range(len(activations) + 1)]

            # Input layer
            layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
            if self.activations is not None:
                activation = self._get_activation(self.activations[0])
                if activation is not None:
                    layers.append(activation)

            # Hidden layer
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
                if self.activations is not None:
                    activation = self._get_activation(self.activations[i])
                    if activation is not None:
                        layers.append(activation)

            # Output layer
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
            if self.activations is not None:
                activation = self._get_activation(self.activations[i + 1])
                if activation is not None:
                    layers.append(activation)

        else:
            # Dealing with activation functions
            if isinstance(activations, str):
                self.activations = [activations]

            layers.append(nn.Linear(self.input_dim, self.output_dim))
            if self.activations is not None:
                activation = self._get_activation(self.activations[0])
                if activation is not None:
                    layers.append(activation)

        self.layers = nn.ModuleList(layers)
        self.model = nn.Sequential(*layers)

    @staticmethod
    @beartype
    def _get_activation(name: Optional[str]) -> Optional[nn.Module]:
        """
        Helper method to get the activation function based on its name.

        :param name: Name of the activation function.
        :returns nn.Module: Activation function module.
        :raises ValueError: If the activation function name is not supported.
        """
        if name is None:
            return None
        elif name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        :param x: Input tensor.
        :returns: Output tensor.
        """
        return self.model(x)

    def save(self, path):
        """
        Save the model at the given path.

        :param path: The path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model from the given path.

        :param path: The path where the model is saved.
        """
        self.load_state_dict(torch.load(path))
        self.eval()
