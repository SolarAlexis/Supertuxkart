import copy
from pathlib import Path
import inspect
import pickle

import torch
import torch.nn as nn
import numpy as np
from bbrl_utils.algorithms import EpisodicAlgo, iter_partial_episodes
from bbrl.agents import TemporalAgent, KWAgentWrapper
from bbrl.workspace import Workspace
from bbrl_utils.nn import setup_optimizer, copy_parameters
from bbrl.utils.functional import gae
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from pystk2_gymnasium import AgentSpec
from functools import partial

from .actors import VAgent, DiscretePolicy
from .pystk_actor import get_wrappers, player_name

env_name = "supertuxkart/flattened_discrete-v0"

class PPOClip(EpisodicAlgo):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, env_wrappers=get_wrappers(), autoreset=True)

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

        self.train_policy = DiscretePolicy(
            obs_space,
            cfg.algorithm.architecture.actor_hidden_size,
            action_space,
        ).with_prefix("current_policy/")

        self.eval_policy = KWAgentWrapper(
            self.train_policy, 
            stochastic=False,
            predict_proba=False,
            compute_entropy=False,
        )

        self.critic_agent = VAgent(
            obs_space, cfg.algorithm.architecture.critic_hidden_size
        ).with_prefix("critic/")
        self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")

        self.old_policy = copy.deepcopy(self.train_policy)
        self.old_policy.with_prefix("old_policy/")

        self.policy_optimizer = setup_optimizer(
            cfg.optimizer, self.train_policy
        )
        self.critic_optimizer = setup_optimizer(
            cfg.optimizer, self.critic_agent
        )
        
def pretraining_ppo(actor, critic, actor_optimizer, critic_optimizer, demo_data, logger,
                    num_iterations=100000, batch_size=256, discount=0.98):
    """
    Pré-entraînement de l'acteur et du critic PPO à partir de démonstrations.

    Pour l'acteur, on effectue du behavioral cloning en minimisant la cross-entropy
    entre les logits prédits et l'action démonstrative.
    
    Pour le critic, on effectue une régression TD en ajustant V(s) vers la target :
         target = reward + discount * V(next_obs) * (1 - done)

    On utilise le Workspace et TemporalAgent pour alimenter les agents avec les observations.
    
    :param actor: l'agent policy (ex: instance de DiscretePolicy)
    :param critic: l'agent critic (ex: instance de VAgent)
    :param actor_optimizer: optimizer pour l'acteur
    :param critic_optimizer: optimizer pour le critic
    :param demo_data: liste des transitions de démonstration.
                      Chaque élément doit être un dict avec :
                        - "obs": dict avec "continuous" et "discrete"
                        - "action": action démonstrative (entier)
                        - "reward": reward (float)
                        - "next_obs": dict avec "continuous" et "discrete"
                        - "done": bool indiquant la fin d'épisode
    :param logger: objet logger (doit posséder une méthode add_log)
    :param num_iterations: nombre total d'itérations de pré-entraînement
    :param batch_size: taille du mini-batch
    :param discount: facteur de discount pour le critic
    """
    actor.train()
    critic.train()
    ce_loss = nn.CrossEntropyLoss()  # pour l'acteur
    mse_loss = nn.MSELoss()            # pour le critic
    device = next(actor.parameters()).device

    t_actor = TemporalAgent(actor)
    t_critic = TemporalAgent(critic)

    for it in range(num_iterations):
        # Sélection aléatoire d'un mini-batch
        indices = np.random.choice(len(demo_data), batch_size, replace=True)
        obs_batch = [demo_data[i]['obs'] for i in indices]
        actions_batch = [demo_data[i]['action'] for i in indices]
        rewards_batch = [demo_data[i]['reward'] for i in indices]
        next_obs_batch = [demo_data[i]['next_obs'] for i in indices]
        dones_batch = [1.0 if demo_data[i]['done'] else 0.0 for i in indices]

        # Construction des tenseurs pour les observations (continuous & discrete)
        obs_continuous = np.stack([obs["continuous"] for obs in obs_batch], axis=0)
        obs_discrete = np.stack([obs["discrete"] for obs in obs_batch], axis=0)
        next_obs_continuous = np.stack([obs["continuous"] for obs in next_obs_batch], axis=0)
        next_obs_discrete = np.stack([obs["discrete"] for obs in next_obs_batch], axis=0)
        
        obs_cont_tensor = torch.tensor(obs_continuous, dtype=torch.float32, device=device)
        obs_disc_tensor = torch.tensor(obs_discrete, dtype=torch.float32, device=device)
        next_obs_cont_tensor = torch.tensor(next_obs_continuous, dtype=torch.float32, device=device)
        next_obs_disc_tensor = torch.tensor(next_obs_discrete, dtype=torch.float32, device=device)
        
        # Actions (pour l'acteur, on suppose des indices entiers)
        actions_tensor = torch.tensor(actions_batch, dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.float32, device=device)
        
        # --- Pré-entraînement de l'acteur par Behavioral Cloning ---
        # Remplissage du workspace avec les observations de l'état courant.
        ws = Workspace()
        ws.set("env/env_obs/continuous", 0, obs_cont_tensor)
        ws.set("env/env_obs/discrete", 0, obs_disc_tensor)
        
        # Passage de l'agent acteur dans le workspace
        t_actor(ws, t=0, n_steps=1)
        # L'acteur doit mettre dans ws la clé "action" à l'instant 0.
        predicted_actions = ws["action"][0]  # actions prédite(s) (indices)
        # Pour le BC, on préfère utiliser directement les logits de l'acteur.
        # On reconstruit l'observation comme concaténation.
        observation = torch.cat([obs_cont_tensor, obs_disc_tensor], dim=1)
        logits = actor.model(observation)  # [batch_size, n_actions]
        actor_loss = ce_loss(logits, actions_tensor)
        
        # --- Pré-entraînement du critic par régression TD ---
        # Calcul de V(s) pour l'état courant.
        ws.set("env/env_obs/continuous", 0, obs_cont_tensor)
        ws.set("env/env_obs/discrete", 0, obs_disc_tensor)
        t_critic(ws, t=0, n_steps=1)
        v_pred = ws["critic/v_values"][0]  # [batch_size]
        
        # Calcul de V(next_obs)
        ws_next = Workspace()
        ws_next.set("env/env_obs/continuous", 0, next_obs_cont_tensor)
        ws_next.set("env/env_obs/discrete", 0, next_obs_disc_tensor)
        t_critic(ws_next, t=0, n_steps=1)
        v_next = ws_next["critic/v_values"][0].detach()
        
        # Calcul de la target TD
        target = rewards_tensor + discount * v_next * (1 - dones_tensor)
        critic_loss = mse_loss(v_pred, target)
        
        # --- Optimisation ---
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        
        logger.add_log("pretraining_actor_loss", actor_loss, it)
        logger.add_log("pretraining_critic_loss", critic_loss, it)
        
        if it % 1000 == 0:
            print(f"Pretraining it {it}/{num_iterations}, actor_loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}")

def run(ppo_clip: PPOClip):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    i = 1
    
    cfg = ppo_clip.cfg
    

    t_policy = TemporalAgent(ppo_clip.train_policy)
    t_old_policy = TemporalAgent(ppo_clip.old_policy)
    t_critic = TemporalAgent(ppo_clip.critic_agent)
    t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)
    
    if cfg.pretraining:
        # --- Pré-entraînement par imitation ---
        demo_data_path = "/home/alexis/SuperTuxKart/stk_actor/combined_demo_data.pkl"
        with open(demo_data_path, "rb") as f:
            demo_data = pickle.load(f)
        
        print("Lancement du pré-entraînement (Behavioral Cloning) sur l'acteur...")
        pretraining_ppo(ppo_clip.train_policy, ppo_clip.critic_agent, ppo_clip.policy_optimizer, ppo_clip.critic_optimizer, demo_data, ppo_clip.logger, num_iterations=200000, batch_size=cfg.algorithm.batch_size)
        print("Pré-entraînement terminé.")
        torch.save(ppo_clip.train_policy.state_dict(), mod_path / "pystk_actor.pth")
    
    if cfg.load_model:
        ppo_clip.train_policy.load_state_dict(torch.load(mod_path / "pystk_actor.pth", weights_only=True))
        ppo_clip.critic_agent.load_state_dict(torch.load(mod_path / "pystk_critic.pth", weights_only=True))

    for train_workspace in iter_partial_episodes(
        ppo_clip, cfg.algorithm.n_steps
    ):
        # Run the current policy and evaluate the proba of its action according
        # to the old policy The old_policy can be run after the train_agent on
        # the same workspace because it writes a logprob_predict and not an
        # action. That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given
        # its own probabilities

        with torch.no_grad():
            t_old_policy(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                # Just computes the probability of the old policy's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        t_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)
        with torch.no_grad():
            t_old_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
            "env/terminated",
            "env/reward",
            "critic/v_values",
            "old_critic/v_values",
        ]

        # the critic values are clamped to move not too far away from the values of the previous critic
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            ws_v_value = ws_old_v_value + torch.clamp(
                ws_v_value - ws_old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # Compute the advantage using the (clamped) critic values
        with torch.no_grad():
            advantage = gae(
                ws_reward[1:],
                ws_v_value[1:],
                ~ws_terminated[1:],
                ws_v_value[:-1],
                cfg.algorithm.discount_factor,
                cfg.algorithm.gae,
            )

        ppo_clip.critic_optimizer.zero_grad()
        target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
        critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target) * cfg.algorithm.critic_coef
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ppo_clip.critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        ppo_clip.critic_optimizer.step()

        # We store the advantage into the transition_workspace
        if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        train_workspace.set_full("advantage", torch.cat(
            (advantage, torch.zeros(1, advantage.shape[1]))
        ))
        transition_workspace = train_workspace.get_transitions()

        # Inner optimization loop: we sample transitions and use them to learn
        # the policy
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            # Compute the policy loss

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            # Note that the policy is not wrapped into a TemporalAgent, but we use a single step
            # Compute the ratio of action probabilities
            # Compute the policy loss
            # (using cfg.algorithm.clip_range and torch.clamp)
            
            old_probs = torch.exp(sample_workspace["old_policy/logprob_predict"][0])
            
            ppo_clip.train_policy(sample_workspace, t=0, predict_proba=True, compute_entropy=True)
            current_probs = torch.exp(sample_workspace["current_policy/logprob_predict"][0])

            policy_advantage = sample_workspace["advantage"][0]
            
            eps = cfg.algorithm.clip_range
            policy_loss = torch.mean(torch.min(current_probs / old_probs * policy_advantage, torch.clamp(current_probs / old_probs, 1-eps, 1+eps) * policy_advantage))

            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration Note that the standard PPO
            # algorithms do not have an entropy term, they don't need it because
            # the KL term is supposed to deal with exploration So, to run the
            # standard PPO algorithm, you should set
            # cfg.algorithm.entropy_coef=0
            
            entropy = sample_workspace["current_policy/entropy"]
            
            assert len(entropy) == 1, f"{entropy.shape}"
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            ppo_clip.logger.log_losses(
                critic_loss, entropy_loss, policy_loss, ppo_clip.nb_steps
            )
            ppo_clip.logger.add_log(
                "advantage", policy_advantage[0].mean(), ppo_clip.nb_steps
            )

            loss = loss_policy + loss_entropy

            ppo_clip.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ppo_clip.train_policy.parameters(), cfg.algorithm.max_grad_norm
            )
            ppo_clip.policy_optimizer.step()

        # Copy parameters
        copy_parameters(ppo_clip.train_policy, ppo_clip.old_policy)
        copy_parameters(ppo_clip.critic_agent, ppo_clip.old_critic_agent)

        # Evaluates our current algorithm if needed
        ppo_clip.evaluate()
        
        if i == 20:
            torch.save(ppo_clip.train_policy.state_dict(), mod_path / "pystk_actor.pth")
            torch.save(ppo_clip.critic_agent.state_dict(), mod_path / "pystk_critic.pth")
            i = 0
        i+=1
    
    torch.save(ppo_clip.train_policy.state_dict(), mod_path / "pystk_actor.pth")
    torch.save(ppo_clip.critic_agent.state_dict(), mod_path / "pystk_critic.pth")