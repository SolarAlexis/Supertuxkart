try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run
    assert run(["pip", "install", "easypip"]).returncode == 0, "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils").setup()

import copy
import os

import math
import numpy as np
import torch
import torch.nn as nn
from bbrl.agents import Agent, Agents, KWAgentWrapper, TemporalAgent
from bbrl_utils.algorithms import EpisodicAlgo, iter_partial_episodes
from bbrl_utils.nn import build_ortho_mlp, setup_optimizer
from bbrl_utils.notebook import setup_tensorboard
from omegaconf import OmegaConf

from bbrl_utils.nn import copy_parameters

import gymnasium as gym
from bbrl.agents import Agent
import torch


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, state_dim, hidden_size, action_size, name="policy"):
        super().__init__(name=name)
        
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.action_sizes = [size.n for size in action_size]
        output_size = sum(self.action_sizes)
        
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_size) + [output_size], activation=nn.ReLU()
        )
        
    def dist(self, obs):
        
        continuous = torch.tensor(obs["continuous"])
        discrete = torch.tensor(obs["discrete"])
        x = torch.cat([continuous, discrete], dim=0)
        
        logits = self.model(x)
        split_logits = list(torch.split(logits, self.action_sizes, dim=0))
        for split in split_logits:
            print(torch.softmax(split, dim=-1))
        probs = [torch.distributions.Categorical(torch.softmax(split, dim=-1)) for split in split_logits]
        print(probs, "ici")
        return probs
        
    def forward(
        self,
        t,
        *,
        stochastic=True,
        predict_proba=False,
        compute_entropy=False,
        **kwargs,
    ):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)

        if predict_proba:
            action = self.get(("action", t))
            log_probs = probs[torch.arange(probs.size()[0]), action].log()
            self.set((f"{self.prefix}logprob_predict", t), log_probs)
        else:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = scores.argmax(1)
            self.set(("action", t), action)

        if compute_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            self.set((f"{self.prefix}entropy", t), entropy)

class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, name="critic"):
        super().__init__(name)
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)

class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass

class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))







class DiscretePolicy(Agent):
    def __init__(self, state_dim, hidden_size, action_size, name="policy"):
        super().__init__(name=name)
        
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.action_sizes = [size.n for size in action_size]
        output_size = sum(self.action_sizes)
        
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_size) + [output_size], activation=nn.ReLU()
        )

    def dist(self, obs):
        
        continuous = torch.tensor(obs["continuous"])
        discrete = torch.tensor(obs["discrete"])
        x = torch.cat([continuous, discrete], dim=0)
        
        logits = self.model(x)
        split_logits = list(torch.split(logits, self.action_sizes, dim=0))
        probs = [torch.distributions.Categorical(torch.softmax(split, dim=-1)) for split in split_logits]
        return probs

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
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)

        if predict_proba:
            action = self.get(("action", t))
            log_probs = probs[torch.arange(probs.size()[0]), action].log()
            self.set((f"{self.prefix}logprob_predict", t), log_probs)
        else:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = scores.argmax(1)
            self.set(("action", t), action)

        if compute_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            self.set((f"{self.prefix}entropy", t), entropy)
            
class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, name="critic"):
        super().__init__(name)
        
        input_size = state_dim["continuous"].shape[0] + len(state_dim["discrete"])
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)
        

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
        self.set(("q_values", t), q_values)
        
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