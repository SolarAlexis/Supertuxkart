try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run
    assert run(["pip", "install", "easypip"]).returncode == 0, "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils").setup()

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, constraints
from torch.distributions.transforms import Transform
from bbrl.agents import Agent
from bbrl_utils.nn import build_ortho_mlp, build_mlp
import gymnasium as gym
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List

class FeatureFilterWrapper(gym.ObservationWrapper):
    def __init__(self, env, index):
        super(FeatureFilterWrapper, self).__init__(env)
        self.index = index
        
        continuous_space = self.env.observation_space["continuous"]
        low = np.delete(continuous_space.low, index)
        high = np.delete(continuous_space.high, index)
        filtered_continuous_space = gym.spaces.Box(low=low, high=high, dtype=continuous_space.dtype)
        self.observation_space = gym.spaces.Dict({"continuous": filtered_continuous_space})

    def observation(self, obs):
        # Assume obs is a dictionary with "continuous" key
        # Delete specified indices from "continuous" and return a new dictionary
        partial_obs = np.delete(obs["continuous"], self.index)
        return {"continuous": partial_obs}

class MyActionRescaleWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1,  1], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, agent_action):
        """
        Transforme l'action de l'agent (dans [-1,1] pour chacune des 2 composantes)
        en une action valide pour l'environnement ([0,1] pour la première composante
        et [-1,1] pour la deuxième).
        """
        
        env_action = np.array(agent_action, copy=True, dtype=np.float32)

        env_action[..., 0] = (env_action[..., 0] + 1.0) / 2.0

        return env_action

# DQN part ---------------------------------------------------------------

class DiscreteQAgent(Agent):
    """BBRL agent (discrete actions) based on a MLP"""

    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.action_sizes = [size.n for size in action_dim]
        output_size = math.prod(self.action_sizes)
        
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [output_size], activation=nn.ReLU()
        )

    def forward(self, t: int, **kwargs):
        """An Agent can use self.workspace"""

        # Retrieves the observation from the environment at time t
        continuous = self.get(("env/env_obs/continuous", t))
        discrete = self.get(("env/env_obs/discrete", t))
        x = torch.cat([continuous, discrete], dim=1)

        # Computes the critic (Q) values for the observation
        q_values = self.model(x)
        # ... and sets the q-values (one for each possible action)
        self.set((f"{self.prefix}q_values", t), q_values)
        
class ArgmaxActionSelector(Agent):
    """BBRL agent that selects the best action based on Q(s,a)"""

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        flat_action = q_values.argmax(-1)
        action_sizes = [5, 2, 2, 2, 2, 2, 7]
        action = torch.tensor([np.unravel_index(idx, action_sizes) for idx in flat_action])
        self.set(("action", t), action)
        
class EGreedyActionSelector(Agent):
    def __init__(self, epsilon, action_dim):
        super().__init__()
        self.epsilon = epsilon
        self.action_sizes = [size.n for size in action_dim]

    def forward(self, t: int, **kwargs):
        # Récupérer les q_values
        q_values: torch.Tensor = self.get(("q_values", t))
        size, _ = q_values.shape

        # Calculer les actions avec argmax (indices plats)
        flat_max_action = q_values.argmax(-1) 

        # Convertir les indices plats en actions MultiDiscrete
        max_action = torch.tensor(
            [np.unravel_index(idx, self.action_sizes) for idx in flat_max_action]
        )

        # Générer des actions aléatoires MultiDiscrete
        random_action = torch.stack(
            [torch.randint(dim_size, size=(size,)) for dim_size in self.action_sizes],
            dim=-1,
        )

        # Décider aléatoirement entre random et argmax
        is_random = torch.rand(size) < self.epsilon  # Taille (size,)
        is_random = is_random.unsqueeze(-1)  # Taille (size, 1) pour broadcast

        action = torch.where(is_random, random_action, max_action)
        
        self.set(("action", t), action)
        
        
# ddpg part ---------------------------------------------------------------
        
class ContinuousDeterministicActor(Agent):
    def __init__(self, obs_space, hidden_layers, action_space):
        super().__init__()
        
        self.action_space = action_space
        obs_space_size = obs_space["continuous"].shape[0]
        action_space_size = action_space.shape[0]
        
        layers = [obs_space_size] + list(hidden_layers) + [action_space_size]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs/continuous", t))
        action = self.model(obs)
        
        # low = torch.tensor(self.action_space.low, dtype=action.dtype)
        # high = torch.tensor(self.action_space.high, dtype=action.dtype)
        # scaled_action = low + (high - low) * (action + 1) / 2
        
        assert torch.all(action >= -1) and torch.all(action <= 1), "Actions out of bounds!"
        self.set(("action", t), action)
        
class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        sigma = torch.full_like(act, self.sigma)
        dist = Normal(act, sigma)
        action = dist.sample()
        
        self.set(("action", t), action)
        
class AddOUNoise(Agent):
    """
    Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)
        
# SAC part ---------------------------------------------------------------

class SquashedGaussianActor(Agent):
    def __init__(self, obs_space, hidden_layers, action_space, min_std=1e-4):
        """Creates a new Squashed Gaussian actor

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-2 (1e-4)
        """
        super().__init__()
        
        obs_space_size = obs_space["continuous"].shape[0]
        action_space_size = action_space.shape[0]
        
        self.min_std = min_std
        backbone_dim = [obs_space_size] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_space_size)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_space_size)
        self.softplus = nn.Softplus()
        
        # cache_size avoids numerical infinites or NaNs when
        # computing log probabilities
        self.tanh_transform = TanhTransform(cache_size=1)

    def normal_dist(self, obs: torch.Tensor):
        """Compute normal distribution given observation(s)"""
        
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = torch.clamp(self.softplus(std_out) + self.min_std, max=10.0)
        # Independent ensures that we have a multivariate
        # Gaussian with a diagonal covariance matrix (given as
        # a vector `std`)
        return Independent(Normal(mean, std), 1)
        
    def forward(self, t, stochastic=True):
        """Computes the action a_t and its log-probability p(a_t| s_t)

        :param stochastic: True when sampling
        """
        normal_dist = self.normal_dist(self.get(("env/env_obs/continuous", t)))
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        if stochastic:
            # Uses the re-parametrization trick
            action = action_dist.rsample()
        else:
            # Directly uses the mode of the distribution
            action = self.tanh_transform(normal_dist.mode)

        action = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        log_prob = torch.clamp(action_dist.log_prob(action), max=0)
        # This line allows to deepcopy the actor...
        self.tanh_transform._cached_x_y = [None, None]
        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)
        
        
class ContinuousQAgent(Agent):
    def __init__(self, obs_space, hidden_layers: list[int], action_space):
        """Creates a new critic agent $Q(s, a)$

        :param state_dim: The number of dimensions for the observations
        :param hidden_layers: The list of hidden layers for the NN
        :param action_dim: The numer of dimensions for actions
        """
        super().__init__()
        
        obs_space_size = obs_space["continuous"].shape[0]
        action_space_size = action_space.shape[0]
        
        self.is_q_function = True
        self.model = build_mlp(
            [obs_space_size + action_space_size] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs/continuous", t))
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)
        

# PPO part ---------------------------------------------------------------

def build_ortho_mlp(sizes, activation, output_activation=nn.Identity(), use_layer_norm=True):
    """
    Construit un MLP avec initialisation orthogonale et option de normalisation par couche (LayerNorm)
    sur les couches cachées.
    """
    layers = []
    for j in range(len(sizes) - 1):
        linear = nn.Linear(sizes[j], sizes[j+1])
        # Initialisation orthogonale
        gain = nn.init.calculate_gain(activation.__class__.__name__.lower()) if j < len(sizes)-2 else 1.0
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.constant_(linear.bias, 0)
        layers.append(linear)
        # Ajouter LayerNorm seulement pour les couches cachées
        if use_layer_norm and j < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[j+1]))
        # Appliquer l'activation (output_activation sur la dernière couche)
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act)
    return nn.Sequential(*layers)

class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, name="critic"):
        super().__init__(name)
        self.is_q_function = False
        # On combine les dimensions des observations continues et discrètes
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [1],
            activation=nn.ReLU(),
            use_layer_norm=True 
        )

    def forward(self, t, **kwargs):
        continuous = self.get(("env/env_obs/continuous", t))
        discrete = self.get(("env/env_obs/discrete", t))
        observation = torch.cat([continuous, discrete], dim=1)
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)
        
class DiscretePolicy(Agent):
    def __init__(self, state_dim, hidden_size, n_actions, name="policy"):
        super().__init__(name=name)
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        action_space_size = n_actions.n
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_size) + [action_space_size],
            activation=nn.ReLU(),
            use_layer_norm=True
        )

    def dist(self, obs):
        scores = self.model(obs)
        assert torch.isfinite(scores).all(), "Scores contain NaN or Inf"
        probs = torch.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)
        assert (probs >= 0).all() and (probs <= 1).all(), "Invalid probabilities"
        return torch.distributions.Categorical(probs)

    def forward(
        self,
        t,
        *,
        stochastic=True,
        predict_proba=False,
        compute_entropy=False,
        **kwargs,
    ):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        """
        continuous = self.get(("env/env_obs/continuous", t))
        discrete = self.get(("env/env_obs/discrete", t))
        observation = torch.cat([continuous, discrete], dim=1)
        assert torch.isfinite(observation).all(), "Observation contains NaN or Inf"
        scores = self.model(observation)
        probs = torch.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)
        assert (probs >= 0).all() and (probs <= 1).all(), "Invalid probabilities"

        if predict_proba:
            action = self.get(("action", t))
            log_probs = (probs + 1e-8).log()[torch.arange(probs.size(0)), action]
            assert torch.isfinite(log_probs).all(), "Log_probs contain NaN or Inf"
            self.set((f"{self.prefix}logprob_predict", t), log_probs)
        else:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = scores.argmax(1) 
            self.set(("action", t), action)

        if compute_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            assert torch.isfinite(entropy).all(), "Entropy contains NaN or Inf"
            self.set((f"{self.prefix}entropy", t), entropy)

# TQC part ---------------------------------------------------------------

class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(
        self, *args, **kwargs
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(
            torch.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def make_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)

class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "SquashedDiagGaussianDistribution":
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(
        self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        raise Exception("Call to entropy in squashed Diag Gaussian distribution")
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob

    def get_ten_samples(self) -> List:
        action_list = []
        for i in range(10):
            action = self.sample()
            action_list.append(action)
        return action_list

class BaseActor(Agent):
    """ Generic class to centralize copy_parameters"""

    def copy_parameters(self, other):
        """Copy parameters from other agent"""
        for self_p, other_p in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(other_p)

def build_mlp(sizes, activation, output_activation=nn.Identity(), use_layer_norm=True):
    """Construit un MLP avec option de normalisation par couche sur les couches cachées."""
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        # On ajoute la normalisation seulement pour les couches cachées
        if use_layer_norm and j < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[j + 1]))
        # L'activation est appliquée après normalisation (sauf sur la dernière couche)
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act)
    return nn.Sequential(*layers)

def build_backbone(sizes, activation, use_layer_norm=True):
    """Construit la partie 'backbone' d'un réseau avec option de normalisation par couche."""
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(sizes[j + 1]))
        layers.append(activation)
    return layers

class SquashedGaussianActorTQC(BaseActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        
        obs_space_size = state_dim["continuous"].shape[0]
        action_space_size = action_dim.shape[0]
        
        backbone_dim = [obs_space_size] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.Tanh())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_space_size)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_space_size)
        self.action_dist = SquashedDiagGaussianDistribution(action_space_size)

    def get_distribution(self, obs: torch.Tensor):
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)

        std_out = std_out.clamp(-5, 2)  # as in the official code
        std = torch.exp(std_out)
        return self.action_dist.make_distribution(mean, std)

    def forward(self, t, stochastic=False, predict_proba=False, **kwargs):
        action_dist = self.get_distribution(self.get(("env/env_obs/continuous", t)))
        if predict_proba:
            action = self.get(("action", t))
            log_prob = action_dist.log_prob(action)
            self.set(("logprob_predict", t), log_prob)
        else:
            if stochastic:
                action = action_dist.sample()
            else:
                action = action_dist.mode()
            log_prob = action_dist.log_prob(action)
            self.set(("action", t), action)
            self.set(("action_logprobs", t), log_prob)

    def predict_action(self, obs, stochastic=False):
        """Predict just one action (without using the workspace)"""
        action_dist = self.get_distribution(obs)
        return action_dist.sample() if stochastic else action_dist.mode()

class TruncatedQuantileNetwork(Agent):
    def __init__(self, state_dim, hidden_layers, n_nets, action_dim, n_quantiles):
        super().__init__()
        
        obs_space_size = state_dim["continuous"].shape[0]
        action_space_size = action_dim.shape[0]
        
        self.is_q_function = True
        self.nets = []
        for i in range(n_nets):
            net = build_mlp([obs_space_size + action_space_size] + list(hidden_layers) + [n_quantiles], activation=nn.ReLU())
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, t):
        obs = self.get(("env/env_obs/continuous", t))
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        self.set(("quantiles", t), quantiles)
        return quantiles

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        return quantiles