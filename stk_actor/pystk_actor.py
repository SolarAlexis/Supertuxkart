import os
import copy

# Ajoutez le chemin contenant les DLLs
os.add_dll_directory(r"D:\\projet_SuperTuxKart\\stk-code\\build\\bin\\Debug")

import torch
from typing import List, Callable
from bbrl.agents import Agents, Agent, KWAgentWrapper, TemporalAgent
import gymnasium as gym
from bbrl_utils.nn import setup_optimizer
from bbrl_utils.algorithms import EpisodicAlgo
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from pystk2_gymnasium import AgentSpec
from functools import partial

# Imports our Actor class
# IMPORTANT: note the relative import
from actors import Actor, MyWrapper, ArgmaxActor, SamplingActor, VAgent, DiscretePolicy, DiscreteQAgent, EGreedyActionSelector, ArgmaxActionSelector
from utils import compute_critic_loss_dqn


#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name
player_name = "SupramaXx"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = Actor(observation_space, action_space)

    # Returns a dummy actor
    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: MyWrapper(env, option="1")
    ]

class PPOClip(EpisodicAlgo):
    def __init__(self, cfg):
        super().__init__(cfg, env_wrappers=get_wrappers(), autoreset=True)
        
        obs_space_size = self.train_env.envs[0].observation_space
        action_space_size = self.train_env.envs[0].action_space
        
        print("\nObservation space: ", obs_space_size)
        print("\naction space: ", action_space_size)
        
        self.train_policy = globals()[cfg.algorithm.policy_type](
            obs_space_size,
            cfg.algorithm.architecture.actor_hidden_size,
            action_space_size,
        ).with_prefix("current_policy/")
        
        self.old_policy = copy.deepcopy(self.train_policy).with_prefix("old_policy/")

        # self.eval_policy = KWAgentWrapper(
        #     self.train_policy, 
        #     stochastic=False,
        #     predict_proba=False,
        #     compute_entropy=False,
        # )

        # self.critic_agent = VAgent(
        #     obs_size, cfg.algorithm.architecture.critic_hidden_size
        # ).with_prefix("critic/")
        # self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")

        # self.old_policy = copy.deepcopy(self.train_policy).with_prefix("old_policy/")

        # self.policy_optimizer = setup_optimizer(
        #     cfg.optimizer, self.train_policy
        # )
        # self.critic_optimizer = setup_optimizer(
        #     cfg.optimizer, self.critic_agent
        # )

class EpisodicDQN(EpisodicAlgo):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, env_wrappers=get_wrappers(), autoreset=False)

        make_stkenv = partial(
            make_env,
            env_name,
            wrappers=get_wrappers(),
            render_mode=render_mode,
            autoreset=False,
            agent=AgentSpec(use_ai=False, name=player_name),
        )
        self.train_env = ParallelGymAgent(make_stkenv, cfg.algorithm.n_envs)
        
        obs_space_size = self.train_env.envs[0].observation_space
        action_space_size = self.train_env.envs[0].action_space
        
        # Our discrete Q-Agent
        self.q_agent = DiscreteQAgent(
            obs_space_size, cfg.algorithm.architecture.hidden_size, action_space_size
        )

        # The e-greedy strategy (when training)
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon, action_space_size)

        # The training agent combines the Q agent
        self.train_policy = Agents(self.q_agent, explorer)

        # The optimizer for the Q-Agent parameters
        self.optimizer = setup_optimizer(self.cfg.optimizer, self.q_agent)

        # ...and the evaluation policy (select the most likely action)
        self.eval_policy = Agents(self.q_agent, ArgmaxActionSelector())
        
    def run(self):
        
        print("check init")
        for train_workspace in self.iter_episodes():
            print("check début iter")
            q_values, terminated, done, reward, action = train_workspace[
                "q_values", "env/terminated", "env/done", "env/reward", "action"
            ]
            must_bootstrap = ~terminated
            critic_loss = compute_critic_loss_dqn(
                self.cfg, reward, must_bootstrap, done, q_values, action
            )
            self.logger.add_log("critic_loss", critic_loss, self.nb_steps)
            self.logger.add_log("q_values/min", q_values.max(-1).values.min(), self.nb_steps)
            self.logger.add_log("q_values/max", q_values.max(-1).values.max(), self.nb_steps)
            self.logger.add_log("q_values/mean", q_values.max(-1).values.mean(), self.nb_steps)

            self.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.q_agent.parameters(), self.cfg.algorithm.max_grad_norm
            )
            self.optimizer.step()
            
            print("check avant eval()")
            
            self.evaluate()
            
            print("check avant reset")
            for env in self.train_env.envs:
                env.reset()
            
            print("check fin iter")