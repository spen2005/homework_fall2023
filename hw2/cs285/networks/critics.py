import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs) # The critic network is a simple MLP, so we can just return the output of the network
        

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        # q_values are obtained from the reward-to-go calculation
        obs = ptu.from_numpy(obs) # Convert the observations to a PyTorch tensor
        q_values = ptu.from_numpy(q_values) # Convert the q_values to a PyTorch tensor

        # TODO: update the critic using the observations and q_values
        loss = F.mse_loss(self.network(obs).squeeze(), q_values) # 1. Sqeeze the output of the network to remove the extra dimension 2. Compute the mean squared error loss between the output of the network and the q_values
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward() # Compute the loss
        self.optimizer.step() # Update the parameters of the critic``

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }