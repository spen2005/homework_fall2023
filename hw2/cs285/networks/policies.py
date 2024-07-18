import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = ptu.from_numpy(obs)

        if self.discrete:
            logits = self.logits_net(obs_tensor)
            action_probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1)
        else:
            mean = self.mean_net(obs_tensor)
            std = torch.exp(self.logstd)
            normal_dist = distributions.Normal(mean, std)
            action = normal_dist.sample()

        action = ptu.to_numpy(action.squeeze())

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        obs_tensor = ptu.from_numpy(obs)

        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs_tensor)
            action_probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs_tensor)
            std = torch.exp(self.logstd)
            normal_dist = distributions.Normal(mean, std)
            action = normal_dist.sample()

        action = ptu.to_numpy(action.squeeze())
        return action

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""

        # TODO: implement the policy gradient actor update.
        obs_tensor = ptu.from_numpy(obs)
        actions_tensor = ptu.from_numpy(actions)
        advantages_tensor = ptu.from_numpy(advantages).squeeze()

        if self.discrete:
            logits = self.logits_net(obs_tensor)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_actions = log_probs[range(len(actions)), actions]
            loss = -torch.mean(log_probs_actions * advantages_tensor)
        else:
            mean = self.mean_net(obs_tensor)
            std = torch.exp(self.logstd)
            normal_dist = torch.distributions.Normal(mean, std)
            log_probs = normal_dist.log_prob(actions_tensor)
            log_probs = log_probs.sum(dim=-1)  # Summing over the action dimensions to match the batch size
            loss = -torch.mean(log_probs * advantages_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
