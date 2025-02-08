import copy
from pathlib import Path
import inspect
from typing import Tuple, Optional, Iterator
import pickle

import numpy as np
import torch
import torch.nn as nn
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env, GymAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from pystk2_gymnasium import AgentSpec
from functools import partial
from tqdm import tqdm
from omegaconf import OmegaConf

from .actors import SquashedGaussianActorTQC, TruncatedQuantileNetwork
from .pystk_actor import get_wrappers, player_name
from .config import params_TQC

class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string: float, loss: float, steps: int):
        self.logger.add_scalar(log_string, loss, steps)

    # A specific function for RL algorithms having a critic, an actor and an
    # entropy losses
    def log_losses(
        self, critic_loss: float, entropy_loss: float, actor_loss: float, steps: int
    ):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards: torch.Tensor, nb_steps):
        self.add_log("reward/mean", rewards.mean().item(), nb_steps)
        self.add_log("reward/max", rewards.max().item(), nb_steps)
        self.add_log("reward/min", rewards.min().item(), nb_steps)
        self.add_log("reward/median", rewards.median().item(), nb_steps)

# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer

def setup_entropy_optimizers(cfg):
    if cfg.algorithm.target_entropy == "auto":
        entropy_coef_optimizer_args = get_arguments(cfg.entropy_coef_optimizer)
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = torch.log(
            torch.ones(1) * cfg.algorithm.entropy_coef
        ).requires_grad_(True)
        entropy_coef_optimizer = get_class(cfg.entropy_coef_optimizer)(
            [log_entropy_coef], **entropy_coef_optimizer_args
        )
    else:
        log_entropy_coef = 0
        entropy_coef_optimizer = None
    return entropy_coef_optimizer, log_entropy_coef

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
def get_env_agents(cfg, *, autoreset=True, include_last_state=True) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, wrappers=get_wrappers(), render_mode=None,
                agent=AgentSpec(use_ai=False, name=player_name), autoreset=autoreset),
        cfg.algorithm.n_envs, 
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, wrappers=get_wrappers(), render_mode=None,
                agent=AgentSpec(use_ai=False, name=player_name), autoreset=autoreset),
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent

def compute_actor_loss(ent_coef, t_actor, q_agent, rb_workspace):
    """Actor loss computation

    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param q_agent: The critic (temporal agent) (n net of m quantiles)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """
    # Recompute the quantiles from the current policy, not from the actions in the buffer

    t_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
    action_logprobs_new = rb_workspace["action_logprobs"]

    q_agent(rb_workspace, t=0, n_steps=1)
    quantiles = rb_workspace["quantiles"][0]

    actor_loss = (ent_coef * action_logprobs_new[0] - quantiles.mean(2).mean(1))

    return actor_loss.mean()

def compute_critic_loss(
        cfg, reward, must_bootstrap,
        t_actor,
        q_agent,
        target_q_agent,
        rb_workspace,
        ent_coef
):
    # Compute quantiles from critic with the actions present in the buffer:
    # at t, we have Qu  ntiles(s,a) from the (s,a) in the RB
    q_agent(rb_workspace, t=0, n_steps=1)
    quantiles = rb_workspace["quantiles"].squeeze()

    with torch.no_grad():
        # Replay the current actor on the replay buffer to get actions of the
        # current policy
        t_actor(rb_workspace, t=1, n_steps=1, stochastic=True)
        action_logprobs_next = rb_workspace["action_logprobs"]

        # Compute target quantiles from the target critic: at t+1, we have
        # Quantiles(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB

        target_q_agent(rb_workspace, t=1, n_steps=1)
        post_quantiles = rb_workspace["quantiles"][1]

        sorted_quantiles, _ = torch.sort(post_quantiles.reshape(quantiles.shape[0], -1))
        quantiles_to_drop_total = cfg.algorithm.top_quantiles_to_drop * cfg.algorithm.architecture.n_nets
        truncated_sorted_quantiles = sorted_quantiles[:,
                                     :quantiles.size(-1) * quantiles.size(-2) - quantiles_to_drop_total]

        # compute the target
        logprobs = ent_coef * action_logprobs_next[1]
        y = reward[0].unsqueeze(-1) + must_bootstrap.int().unsqueeze(-1) * cfg.algorithm.discount_factor * (
                    truncated_sorted_quantiles - logprobs.unsqueeze(-1))

    # computing the Huber loss
    pairwise_delta = y[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples

    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

def create_tqc_agent(cfg, train_env_agent, eval_env_agent):

    obs_space = train_env_agent.envs[0].observation_space
    action_space = train_env_agent.envs[0].action_space
    
    assert (
        train_env_agent.is_continuous_action()
    ), "TQC code dedicated to continuous actions"

    # Actor
    actor = SquashedGaussianActorTQC(
        obs_space, cfg.algorithm.architecture.actor_hidden_size, action_space
    )

    # Train/Test agents
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)

    # Builds the critics
    critic = TruncatedQuantileNetwork(
        obs_space, cfg.algorithm.architecture.critic_hidden_size,
        cfg.algorithm.architecture.n_nets, action_space,
        cfg.algorithm.architecture.n_quantiles
    )
    target_critic = copy.deepcopy(critic)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    #train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    )

def behavioral_cloning_pretraining(actor, optimizer, demo_data, logger, num_iterations=10000, batch_size=256):
    """
    Pré-entraînement de l'acteur par imitation (behavioral cloning) en utilisant
    les démonstrations collectées.

    :param actor: l'acteur à pré-entraîner (de type SquashedGaussianActorTQC)
    :param optimizer: l'optimizer utilisé pour l'acteur (par exemple Adam)
    :param demo_data: la liste des transitions (chargée depuis le pickle)
                      Chaque élément est un dictionnaire contenant par exemple :
                      {'obs': ..., 'action': ..., 'reward': ..., 'next_obs': ..., 'done': ...}
                      On suppose que obs est un dictionnaire avec la clé "continuous".
    :param num_iterations: nombre d'itérations de pré-entraînement
    :param batch_size: taille du mini-batch
    """
    actor.train()
    for it in range(num_iterations):
        # Sélection aléatoire d'un mini-batch
        indices = np.random.choice(len(demo_data), batch_size, replace=True)
        obs_batch = [demo_data[i]['obs'] for i in indices]
        actions_batch = [demo_data[i]['action'] for i in indices]
        
        # Ici, on suppose que chaque observation est un dictionnaire avec la clé "continuous"
        # On crée un batch d'observations (attention aux dimensions attendues par l'acteur)
        obs_continuous = np.stack([obs['continuous'] for obs in obs_batch], axis=0)
        actions = np.stack(actions_batch, axis=0)
        
        # Conversion en tenseurs (et transfert sur le même device que l'acteur)
        device = next(actor.parameters()).device
        obs_tensor = torch.tensor(obs_continuous, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
        
        # Passage dans l'acteur pour obtenir la distribution
        dist = actor.get_distribution(obs_tensor)
        # Calcul de la log-vraisemblance de l'action démonstration
        log_probs = dist.log_prob(actions_tensor)
        loss = - log_probs.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.add_log("pretraining_actor_loss", loss.item(), it)
        
        if it % 1000 == 0:
            print(f"Behavioral Cloning it {it}/{num_iterations}, loss: {loss.item():.4f}")

def run_tqc(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = float('-inf')
    ent_coef = cfg.algorithm.entropy_coef
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    ) = create_tqc_agent(cfg, train_env_agent, eval_env_agent)
    
    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)

    # --- Pré-entraînement par imitation ---
    demo_data_path = "/home/alexis/SuperTuxKart/stk_actor/demo_data.pkl"
    with open(demo_data_path, "rb") as f:
        demo_data = pickle.load(f)
    
    print("Lancement du pré-entraînement (Behavioral Cloning) sur l'acteur...")
    behavioral_cloning_pretraining(actor, actor_optimizer, demo_data, logger, num_iterations=200000, batch_size=cfg.algorithm.batch_size)
    print("Pré-entraînement terminé.")
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
    
    t_actor = TemporalAgent(actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    train_workspace = Workspace()

    # Creates a replay buffer
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    nb_steps = 0
    tmp_steps = 0

    # Initial value of the entropy coef alpha. If target_entropy is not auto,
    # will remain fixed
    if cfg.algorithm.target_entropy == "auto":
        target_entropy = -np.prod(train_env_agent.action_space.shape).astype(np.float32)
    else:
        target_entropy = cfg.algorithm.target_entropy

    mean = np.NINF
    
    # Training loop
    pbar = tqdm(range(cfg.algorithm.max_epochs))
    for epoch in pbar:
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps - 1,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        if nb_steps > cfg.algorithm.learning_starts:
            
            # Accumulateurs pour sommer les métriques sur n_updates
            total_critic_loss = 0.0
            total_actor_loss = 0.0
            total_entropy_coef_loss = 0.0
            total_mean_q_value = 0.0
            total_entropy_coef = 0.0
            
            for _ in range(cfg.algorithm.n_updates):
            
                # Get a sample from the workspace
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

                done, truncated, reward, action_logprobs_rb = rb_workspace[
                    "env/done", "env/truncated", "env/reward", "action_logprobs"
                ]

                # Determines whether values of the critic should be propagated
                # True if the episode reached a time limit or if the task was not done
                # See https://github.com/osigaud/bbrl/blob/master/docs/time_limits.md
                must_bootstrap = ~done[1]

                critic_loss = compute_critic_loss(cfg, reward, must_bootstrap,
                                                t_actor, q_agent, target_q_agent,
                                                rb_workspace, ent_coef)

                total_critic_loss += critic_loss.item()

                actor_loss = compute_actor_loss(
                    ent_coef, t_actor, q_agent, rb_workspace
                )
                total_actor_loss += actor_loss.item()

                # Entropy coef update part ########################
                if entropy_coef_optimizer is not None:
                    # Important: detach the variable from the graph
                    # so that we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = torch.exp(log_entropy_coef.detach())
                    entropy_coef_loss = -(
                            log_entropy_coef * (action_logprobs_rb + target_entropy)
                    ).mean()
                    entropy_coef_optimizer.zero_grad()
                    # We need to retain the graph because we reuse the
                    # action_logprobs are used to compute both the actor loss and
                    # the critic loss
                    entropy_coef_loss.backward(retain_graph=True)
                    entropy_coef_optimizer.step()
                    total_entropy_coef_loss += entropy_coef_loss.item()
                    total_entropy_coef += ent_coef.item()

                # Actor update part ###############################
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), cfg.algorithm.max_grad_norm
                )
                actor_optimizer.step()

                # Critic update part ###############################
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), cfg.algorithm.max_grad_norm
                )
                critic_optimizer.step()
                
                with torch.no_grad():
                    q_agent(rb_workspace, t=0, n_steps=1)
                    quantiles = rb_workspace["quantiles"].squeeze()
                    total_mean_q_value += quantiles.mean().item()
                ####################################################

                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(critic, target_critic, tau)

            # Calcul de la moyenne sur les n_updates
            avg_critic_loss = total_critic_loss / cfg.algorithm.n_updates
            avg_actor_loss = total_actor_loss / cfg.algorithm.n_updates
            if entropy_coef_optimizer is not None:
                avg_entropy_coef_loss = total_entropy_coef_loss / cfg.algorithm.n_updates
                avg_entropy_coef = total_entropy_coef / cfg.algorithm.n_updates
            avg_mean_q_value = total_mean_q_value / cfg.algorithm.n_updates

            # Enregistrement des métriques moyennées
            logger.add_log("critic_loss", avg_critic_loss, nb_steps)
            logger.add_log("actor_loss", avg_actor_loss, nb_steps)
            if entropy_coef_optimizer is not None:
                logger.add_log("entropy_coef_loss", avg_entropy_coef_loss, nb_steps)
                logger.add_log("entropy_coef", avg_entropy_coef, nb_steps)
            logger.add_log("critic/mean_q_value", avg_mean_q_value, nb_steps)
        
        pbar.set_description(f"nb_steps: {nb_steps}, reward: {mean:.3f}")
        
        # Evaluate ###########################################
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                n_steps= cfg.algorithm.eval_steps,
                stochastic=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)

            if cfg.save_best and mean > best_reward:
                best_reward = mean
                torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
            
            torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
            torch.save(critic.state_dict(), mod_path / "pystk_critic.pth")
            
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
    torch.save(critic.state_dict(), mod_path / "pystk_critic.pth")