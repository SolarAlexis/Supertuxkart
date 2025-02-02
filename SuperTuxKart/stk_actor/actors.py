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
from bbrl.utils.distributions import SquashedDiagGaussianDistribution

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
        
        low = torch.tensor(self.action_space.low, dtype=action.dtype)
        high = torch.tensor(self.action_space.high, dtype=action.dtype)
        scaled_action = low + (high - low) * (action + 1) / 2
        
        assert torch.all(scaled_action >= low) and torch.all(scaled_action <= high), "Actions out of bounds!"
        print("action", scaled_action)
        self.set(("action", t), scaled_action)
        
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
        print(action)
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

class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, name="critic"):
        super().__init__(name)
        self.is_q_function = False
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [1], activation=nn.ReLU()
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
            [input_size] + list(hidden_size) + [action_space_size], activation=nn.ReLU()
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
            print(log_probs)
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

        std_out = std_out.clamp(-20, 2)  # as in the official code
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