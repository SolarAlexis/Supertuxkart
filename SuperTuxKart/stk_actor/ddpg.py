import copy
from pathlib import Path
import inspect

import torch
import torch.nn as nn
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer, soft_update_params
from bbrl.visu.plot_policies import plot_policy
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from pystk2_gymnasium import AgentSpec
from functools import partial

from .actors import ContinuousQAgent, ContinuousDeterministicActor, AddGaussianNoise
from .pystk_actor import get_wrappers, player_name
from .config import params_td3

env_name = "supertuxkart/flattened_continuous_actions-v0"

mse = nn.MSELoss()

def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, target_q_values: torch.Tensor, obs: torch.Tensor):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference

    # Apply the must_bootstrap flag to only propagate the Q-values where needed
    target = reward[1] + cfg.algorithm.discount_factor * target_q_values[1] * must_bootstrap[1].float() - 0.5 * torch.abs(obs[:, 3])
    # Get the predicted Q-values for the actions actually taken in the current step
    qvals = q_values[0]
    
    critic_loss = mse(qvals, target)

    return critic_loss

def compute_actor_loss(q_values, actions, coef_steer_penalty=params_td3["coef_steer_penalty"], coef_extreme_steer_penalty=params_td3["coef_extreme_steer_penalty"], coef_acceleration_penalty=params_td3["coef_acceleration_penalty"]):
    # Extraire la dimension "steer" (deuxième colonne)
    steer = actions[:, 1] 
    acceleration = actions[:, 0] 
    
    # Pénalité pour les valeurs de "steer" proches de -1 ou 1
    steer_penalty = torch.mean(torch.abs(steer))  # Moyenne de |steer|
    extreme_steer_penalty = torch.mean(torch.where(torch.abs(steer) > 0.9, torch.abs(steer), torch.tensor(0.0)))
    acceleration_penalty = 1 - torch.mean(torch.where(acceleration < 0.5, acceleration, torch.tensor(1.0)))
    
    # Perte de l'acteur avec pénalité sur "steer" et "acceleration"
    actor_loss = -torch.mean(q_values) + coef_steer_penalty * steer_penalty + coef_extreme_steer_penalty * extreme_steer_penalty + coef_acceleration_penalty * acceleration_penalty
    return actor_loss

class DDPG(EpochBasedAlgo):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, env_wrappers=get_wrappers())

        make_stkenv = partial(
            make_env,
            env_name,
            wrappers=get_wrappers(),
            render_mode=render_mode,
            autoreset=True,
            agent=AgentSpec(use_ai=False, name=player_name),
        )
        self.train_env = ParallelGymAgent(make_stkenv, cfg.algorithm.n_envs)
        
        obs_space = self.train_env.envs[0].observation_space
        action_space = self.train_env.envs[0].action_space
        
        self.critic = ContinuousQAgent(
            obs_space, cfg.algorithm.architecture.critic_hidden_size, action_space
        ).with_prefix("critic/")
        self.target_critic = copy.deepcopy(self.critic).with_prefix("target-critic/")

        self.actor = ContinuousDeterministicActor(
            obs_space, cfg.algorithm.architecture.actor_hidden_size, action_space
        )

        # As an alternative, you can use `AddOUNoise`
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        # Define agents over time
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic = TemporalAgent(self.critic)
        self.t_target_critic = TemporalAgent(self.target_critic)

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)
        
def run_ddpg(ddpg: DDPG, compute_critic_loss, compute_actor_loss):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    i = 1
    
    for rb in ddpg.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(ddpg.cfg.algorithm.batch_size)

        # Compute the critic loss
        ddpg.critic(rb_workspace, t=0)
        ddpg.actor(rb_workspace, t=1)
        ddpg.target_critic(rb_workspace,t=1)
        # Critic update
        # Compute critic loss

        q_values, terminated, reward, target_q_values = rb_workspace[
                "critic/q_value", "env/terminated", "env/reward", "target-critic/q_value"]
        must_bootstrap = ~terminated
        critic_loss = compute_critic_loss(ddpg.cfg, reward, must_bootstrap, q_values, target_q_values)

        # Gradient step (critic)
        ddpg.logger.add_log("critic_loss", critic_loss, ddpg.nb_steps)
        ddpg.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.critic.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.critic_optimizer.step()

        # Compute the actor loss
        ddpg.actor(rb_workspace, t=0)
        ddpg.critic(rb_workspace,t=0)
        q_values = rb_workspace["critic/q_value"]
        actor_loss = compute_actor_loss(q_values)


        # Gradient step (actor)
        ddpg.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.actor.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            ddpg.critic, ddpg.target_critic, ddpg.cfg.algorithm.tau_target
        )

        # Evaluate the actor if needed
        if ddpg.evaluate():
            if ddpg.cfg.plot_agents:
                plot_policy(
                    ddpg.actor,
                    ddpg.eval_env,
                    ddpg.best_reward,
                    str(ddpg.base_dir / "plots"),
                    ddpg.cfg.gym_env.env_name,
                    stochastic=False,
                )

                
        if i == 20:
            torch.save(ddpg.eval_policy.state_dict(), mod_path / "pystk_actor.pth")
            i = 0
        i+=1
                
    torch.save(ddpg.eval_policy.state_dict(), mod_path / "pystk_actor.pth")
    
    
class TD3(EpochBasedAlgo):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, env_wrappers=get_wrappers())

        make_stkenv = partial(
            make_env,
            env_name,
            wrappers=get_wrappers(),
            render_mode=render_mode,
            autoreset=True,
            agent=AgentSpec(use_ai=False, name=player_name),
        )
        self.train_env = ParallelGymAgent(make_stkenv, cfg.algorithm.n_envs)
        
        obs_space = self.train_env.envs[0].observation_space
        action_space = self.train_env.envs[0].action_space
        
        self.critic_1 = ContinuousQAgent(
            obs_space, cfg.algorithm.architecture.critic_hidden_size, action_space
        ).with_prefix("critic_1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")
        
        self.critic_2 = ContinuousQAgent(
            obs_space, cfg.algorithm.architecture.critic_hidden_size, action_space
        ).with_prefix("critic_2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic_2/")

        self.actor = ContinuousDeterministicActor(
            obs_space, cfg.algorithm.architecture.actor_hidden_size, action_space
        )
        
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor
        
        # Define agents over time
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)
        
        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_1_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_2_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_2)
        
def run_td3(td3: TD3, compute_critic_loss, compute_actor_loss):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    i = 1
    
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)
        # Implement the learning loop

        td3.critic_1(rb_workspace, t=0)
        td3.critic_2(rb_workspace, t=0)
        td3.actor(rb_workspace, t=1)
        td3.target_critic_1(rb_workspace,t=1)
        td3.target_critic_2(rb_workspace,t=1)
        
        q_values_1, q_values_2, terminated, reward, target_q_values_1, target_q_values_2, obs = rb_workspace[
                "critic_1/q_value", "critic_2/q_value", "env/terminated", "env/reward", "target-critic_1/q_value", "target-critic_2/q_value", "env/env_obs/continuous"]
        must_bootstrap = ~terminated
        
        # target_q_values: The Q-values computed by the target critic (shape 2xB)
        target_q_values = torch.min(target_q_values_1, target_q_values_2)
        
        critic_loss_1 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q_values_1, target_q_values, obs[1])
        critic_loss_2 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q_values_2, target_q_values, obs[1])
        
        # Gradient step (critic_1)
        td3.logger.add_log("critic_loss_1", critic_loss_1, td3.nb_steps)
        td3.critic_1_optimizer.zero_grad()
        critic_loss_1.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_1_optimizer.step()
        
        # Gradient step (critic_2)
        td3.logger.add_log("critic_loss_2", critic_loss_2, td3.nb_steps)
        td3.critic_2_optimizer.zero_grad()
        critic_loss_2.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_2_optimizer.step()
        
        # Compute the actor loss
        td3.actor(rb_workspace, t=0)
        td3.critic_1(rb_workspace,t=0)
        q_values = rb_workspace["critic_1/q_value"]
        actions = rb_workspace["action"][0]
        actor_loss = compute_actor_loss(q_values, actions)
        td3.logger.add_log("actor_loss", actor_loss, td3.nb_steps)
        
        # Gradient step (actor)
        td3.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.actor_optimizer.step()
        
        # Soft update of target q function
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )
        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )
        
        # Evaluate the actor if needed
        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )
        
        if i == 20:
            torch.save(td3.eval_policy.state_dict(), mod_path / "pystk_actor.pth")
            i = 0
        i+=1
                
    torch.save(td3.eval_policy.state_dict(), mod_path / "pystk_actor.pth")