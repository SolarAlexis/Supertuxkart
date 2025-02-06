import copy
from pathlib import Path
import inspect
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer, soft_update_params
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from pystk2_gymnasium import AgentSpec
from functools import partial

from .actors import SquashedGaussianActor, ContinuousQAgent
from .pystk_actor import get_wrappers, player_name

env_name = "supertuxkart/flattened_continuous_actions-v0"

class SACAlgo(EpochBasedAlgo):
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

        # We need an actor
        self.actor = SquashedGaussianActor(
            obs_space, cfg.algorithm.architecture.actor_hidden_size, action_space
        )

        # Builds the critics
        self.critic_1 = ContinuousQAgent(
            obs_space,
            cfg.algorithm.architecture.critic_hidden_size,
            action_space,
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic-1/"
        )

        self.critic_2 = ContinuousQAgent(
            obs_space,
            cfg.algorithm.architecture.critic_hidden_size,
            action_space,
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix(
            "target-critic-2/"
        )

        # Train and evaluation policies
        self.train_policy = self.actor
        self.eval_policy = KWAgentWrapper(self.actor, stochastic=False)
        
def setup_entropy_optimizers(cfg):
    if cfg.algorithm.entropy_mode == "auto":
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = nn.Parameter(
            torch.log(torch.ones(1) * cfg.algorithm.init_entropy_coef)
        )
        entropy_coef_optimizer = setup_optimizer(
            cfg.entropy_coef_optimizer, log_entropy_coef
        )
        return entropy_coef_optimizer, log_entropy_coef
    else:
        return None, None
    
mse = nn.MSELoss()

def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    t_actor: TemporalAgent,
    t_q_agents: TemporalAgent,
    t_target_q_agents: TemporalAgent,
    rb_workspace: Workspace,
    ent_coef: torch.Tensor,
):
    r"""Computes the critic loss for a set of $S$ transition samples"""

    # Run critics on the workspace to get Q-values at t=0
    t_q_agents(rb_workspace, t=0, n_steps=1)
    
    # Obtain Q-values and target values at t=1
    t_actor(rb_workspace, t=1, n_steps=1)
    action_logprobs = rb_workspace["action_logprobs"]

    t_target_q_agents(rb_workspace, t=1, n_steps=1)
    q_values_1, q_values_2, target_q_values_1, target_q_values_2 = rb_workspace[
        "critic-1/q_value", "critic-2/q_value", "target-critic-1/q_value", "target-critic-2/q_value"
    ]

    # Compute the minimum target Q-value and the value function for the actor
    q_next = torch.minimum(target_q_values_1[1], target_q_values_2[1])
    v_phi = q_next - ent_coef * action_logprobs[1]
    
    # Compute the target using the reward and bootstrapping conditions
    target = reward[1] + cfg.algorithm.discount_factor * v_phi * must_bootstrap[1].int()
    
    # Calculate MSE loss for both critics
    critic_loss_1 = mse(q_values_1[0], target)
    critic_loss_2 = mse(q_values_2[0], target)

    return critic_loss_1, critic_loss_2

def compute_actor_loss(
    ent_coef, t_actor: TemporalAgent, t_q_agents: TemporalAgent, rb_workspace: Workspace
):
    r"""
    Actor loss computation
    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param t_q_agents: The critics (as temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """

    # Recompute the action with the current actor (at $a_t$)

    t_actor(rb_workspace, t=0, n_steps=1)
    action_logprobs = rb_workspace["action_logprobs"]
    action = rb_workspace["action"][0]
    steer = action[:, 1] 

    # Compute Q-values

    t_q_agents(rb_workspace, t=0, n_steps=1)
    q_values_1, q_values_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]

    current_q_values = torch.min(q_values_1, q_values_2)

    # Compute the actor loss

    # actor_loss =
    actor_loss = ent_coef * action_logprobs[0] - current_q_values[0] + torch.mean(torch.where(torch.abs(steer) >= 0.95, torch.abs(steer), torch.tensor(0.0)))


    return actor_loss.mean()

def behavioral_cloning_pretraining(actor, optimizer, demo_data, logger, num_iterations=10000, batch_size=256):
    """
    Pré-entraînement de l'acteur par imitation (behavioral cloning) en utilisant
    les démonstrations collectées.

    :param actor: l'acteur à pré-entraîner (de type SquashedGaussianActor)
    :param optimizer: l'optimizer utilisé pour l'acteur (par exemple Adam)
    :param demo_data: la liste des transitions (chargée depuis le pickle)
                      Chaque élément est un dict contenant par exemple :
                      {'obs': ..., 'action': ..., 'reward': ..., 'next_obs': ..., 'done': ...}
                      On suppose que obs est un dict avec la clé "continuous".
    :param logger: l’instance de logger (ex: Salina Logger)
    :param num_iterations: nombre d'itérations de pré-entraînement
    :param batch_size: taille du mini-batch
    """
    actor.train()
    device = next(actor.parameters()).device  # Récupère le device où se trouve l’acteur

    for it in range(num_iterations):
        # Sélection aléatoire d'un mini-batch
        indices = np.random.choice(len(demo_data), batch_size, replace=True)
        obs_batch = [demo_data[i]['obs'] for i in indices]
        actions_batch = [demo_data[i]['action'] for i in indices]
        
        # On suppose que chaque observation est un dict avec la clé "continuous"
        obs_continuous = np.stack([obs['continuous'] for obs in obs_batch], axis=0)
        actions = np.stack(actions_batch, axis=0)
        
        # Conversion en tenseurs (et transfert sur device)
        obs_tensor = torch.tensor(obs_continuous, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
        
        # Reconstruit la distribution (Normal -> Tanh)
        # 'normal_dist' est déjà implémenté dans l'acteur via actor.normal_dist()
        normal_dist = actor.normal_dist(obs_tensor)
        tanh_dist = TransformedDistribution(normal_dist, [actor.tanh_transform])

        # On peut clamper légèrement les actions si elles peuvent être exactement -1 ou +1
        actions_tensor_clamped = torch.clamp(actions_tensor, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Calcul de la log-probabilité des actions démonstration
        log_probs = tanh_dist.log_prob(actions_tensor_clamped)

        # Perte = - log-likelihood moyenne
        loss = -log_probs.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logger
        logger.add_log("pretraining_actor_loss", loss, it)
        
        # Print pour debug
        if it % 1000 == 0:
            print(f"[Behavioral Cloning] it {it}/{num_iterations}, loss: {loss.item():.4f}")

def run_sac(sac: SACAlgo, compute_critic_loss, compute_actor_loss, setup_entropy_optimizers):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    i = 1
    
    cfg = sac.cfg
    logger = sac.logger

    
    # init_entropy_coef is the initial value of the entropy coef alpha.
    ent_coef = cfg.algorithm.init_entropy_coef
    tau = cfg.algorithm.tau_target

    # Creates the temporal actors
    t_actor = TemporalAgent(sac.train_policy)
    t_q_agents = TemporalAgent(Agents(sac.critic_1, sac.critic_2))
    t_target_q_agents = TemporalAgent(Agents(sac.target_critic_1, sac.target_critic_2))

    # Configure the optimizer
    actor_optimizer = setup_optimizer(cfg.actor_optimizer, sac.actor)
    critic_optimizer = setup_optimizer(cfg.critic_optimizer, sac.critic_1, sac.critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)

    # --- Pré-entraînement par imitation ---
    demo_data_path = "/home/alexis/SuperTuxKart/stk_actor/demo_data.pkl"
    with open(demo_data_path, "rb") as f:
        demo_data = pickle.load(f)
    
    print("Lancement du pré-entraînement (Behavioral Cloning) sur l'acteur...")
    behavioral_cloning_pretraining(sac.actor, actor_optimizer, demo_data, logger, num_iterations=200000, batch_size=cfg.algorithm.batch_size)
    print("Pré-entraînement terminé.")
    torch.save(sac.actor.state_dict(), mod_path / "pystk_actor.pth")
    
    # If entropy_mode is not auto, the entropy coefficient ent_coef remains
    # fixed. Otherwise, computes the target entropy
    if cfg.algorithm.entropy_mode == "auto":
        # target_entropy is \mathcal{H}_0 in the SAC and aplications paper.
        target_entropy = -np.prod(sac.train_env.action_space.shape).astype(np.float32)

    # Loops over successive replay buffers
    for rb in sac.iter_replay_buffers():
        # Implement the SAC algorithm
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        terminated, reward = rb_workspace["env/terminated", "env/reward"]
        if entropy_coef_optimizer is not None:
            ent_coef = torch.exp(log_entropy_coef.detach())
        
        
        # Critic update part #############################
        
        critic_loss_1, critic_loss_2 = compute_critic_loss(
            cfg,
            reward,
            ~terminated,
            t_actor,
            t_q_agents,
            t_target_q_agents,
            rb_workspace,
            ent_coef
        )
        
        logger.add_log("critic_loss_1", critic_loss_1, sac.nb_steps)
        logger.add_log("critic_loss_2", critic_loss_1, sac.nb_steps)
        critic_loss = critic_loss_1 + critic_loss_2
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            sac.critic_1.parameters(), cfg.algorithm.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            sac.critic_2.parameters(), cfg.algorithm.max_grad_norm
        )
        critic_optimizer.step()
        
        # Actor update part #############################
        
        actor_optimizer.zero_grad()
        actor_loss = compute_actor_loss(ent_coef, t_actor, t_q_agents, rb_workspace)
        logger.add_log("actor_loss", actor_loss, sac.nb_steps)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.actor.parameters(), cfg.algorithm.max_grad_norm
        )
        actor_optimizer.step()
        
        # Entropy optimizer part
        if entropy_coef_optimizer is not None:
            # See Eq. (17) of the SAC and Applications paper. The log
            # probabilities *must* have been computed when computing the actor
            # loss.
            action_logprobs_rb = rb_workspace["action_logprobs"].detach()
            entropy_coef_loss = -(
                log_entropy_coef.exp() * (action_logprobs_rb + target_entropy)
            ).mean()
            entropy_coef_optimizer.zero_grad()
            entropy_coef_loss.backward()
            entropy_coef_optimizer.step()
            logger.add_log("entropy_coef_loss", entropy_coef_loss, sac.nb_steps)
            logger.add_log("entropy_coef", ent_coef, sac.nb_steps)

        ####################################################

        # Soft update of target q function
        soft_update_params(sac.critic_1, sac.target_critic_1, tau)
        soft_update_params(sac.critic_2, sac.target_critic_2, tau)

        sac.evaluate()
        
        if i == 20:
            torch.save(sac.actor.state_dict(), mod_path / "pystk_actor.pth")
            i = 0
        i+=1
    
    torch.save(sac.actor.state_dict(), mod_path / "pystk_actor.pth")