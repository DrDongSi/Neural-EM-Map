"""PyTorch implementation of the SIREN model and component sinusoidal layer.

This module is adapted from Sitzmann et. al.,
https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb.
Docstrings and type annotations were added, and unnecessary functionality was
removed. In addition, slight stylistic changes were made.
"""
from typing import Tuple

import numpy as np
import torch


class SineLayer(torch.nn.Module):
    """Implementation of a layer with sinusoidal activation."""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 is_first: bool = False,
                 omega_0: float = 30.) -> None:
        """Construct a sinusoidal activation network layer.

        Args:
            in_features: Number of input connections to this layer.
            out_features: Number of output connections from this layer.
            bias: Optional; Flag to allow layer to learn an additive bias.
            is_first: Optional; Indicates if the layer is the first in the
                network. The weight initialization is different for the first
                SineLayer.
            omega_0: Optional; Additional weight to layer. Empirically
                determined that 30 is the optimal value to increase network
                stability.
        """
        super().__init__()
        self._omega_0 = omega_0
        self._is_first = is_first
        self._in_features = in_features
        self._linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the layer."""
        with torch.no_grad():
            if self._is_first:
                self._linear.weight.uniform_(-1 / self._in_features,
                                             1 / self._in_features)
            else:
                self._linear.weight.uniform_(
                    -np.sqrt(6 / self._in_features) / self._omega_0,
                    np.sqrt(6 / self._in_features) / self._omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through layer

        Args:
            input: Tensor for layer activation.

        Returns:
            Tensor after having been passed through sinusoidal activation.
        """
        return torch.sin(self._omega_0 * self._linear(input))


class Siren(torch.nn.Module):
    """Implementation of the SIREN model with variable architecture."""
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 outermost_linear: bool = True,
                 first_omega_0: float = 30.,
                 hidden_omega_0: float = 30.) -> None:
        """Construct SIREN model.

        Much of the success of using layers with sinusoidal activation
        functions is the omega_0 weight for each layer. It contributes to the
        initial weights of network layers, and its presence during training
        improved network stability. The default value of 30 was determined by
        Sitzmann et. al. to be optimal for their applications.

        Args:
            in_features: Number of input dimensions to the network.
            hidden_features: Number of features per hidden layer.
            hidden_layers: Number of hidden layers.
            out_features: Number of output dimensions from the network.
            outermost_linear: Optional; Use a non-sinusoidal linear layer as
                the final layer of the network.
            first_omega_0: Optional; Omega_0 weight of the first layer. For
                most applications, the default value of 30 is optimal.
            hidden_omega_0: Optional; Omega_0 weight of the hidden layers. For
                most applications, the default value of 30 is optimal.
        """
        super().__init__()
        self._net = list()
        # Construct first layer
        self._net.append(SineLayer(in_features,
                                   hidden_features,
                                   is_first=True,
                                   omega_0=first_omega_0))
        # Construct hidden layers
        for i in range(hidden_layers):
            self._net.append(SineLayer(hidden_features,
                                       hidden_features,
                                       is_first=False,
                                       omega_0=hidden_omega_0))
        # Construct the final layer
        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0)
            self._net.append(final_linear)
        else:
            self._net.append(SineLayer(hidden_features,
                                       out_features,
                                       is_first=False,
                                       omega_0=hidden_omega_0))
        # Create feedforward network from layers
        self._net = torch.nn.Sequential(*self._net)

    def forward(self, model_input: torch.Tensor) -> Tuple[torch.Tensor,
                                                          torch.Tensor]:
        """Perform forward pass through network.

        Args:
            model_input: Tensor of the model's input.

        Returns:
            Tuple of Tensors, respectively the network output and a copy of
            the network input for using with PyTorch autograd to get the first
            derivative.
        """
        # Save copy of model input so we can get derivitives later with
        # respect to the network input
        model_input = model_input.clone().detach().requires_grad_(True)

        # Perform forward pass of network
        model_output = self._net(model_input)

        return (model_output, model_input)
