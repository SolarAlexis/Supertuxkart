
import copy
from pathlib import Path
import inspect

import torch
import torch.nn as nn
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl.agents import Agents, TemporalAgent
from bbrl_utils.nn import setup_optimizer, copy_parameters
from pystk2_gymnasium import AgentSpec
from functools import partial

from .actors import DiscreteQAgent, EGreedyActionSelector, ArgmaxActionSelector
from .pystk_actor import get_wrappers, player_name

env_name = "supertuxkart/flattened_multidiscrete-v0"

class DQN(EpochBasedAlgo):
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
        
        obs_space_size = self.train_env.envs[0].observation_space
        action_space_size = self.train_env.envs[0].action_space

        # Get the two agents (critic and target critic)
        critic = DiscreteQAgent(
            obs_space_size, cfg.algorithm.architecture.hidden_size, action_space_size
        )
        target_critic = copy.deepcopy(critic).with_prefix("target/")
        
        # Builds the train agent that will produce transitions
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon, action_space_size)
        self.train_policy = Agents(critic, explorer)
        
        self.eval_policy = Agents(critic, ArgmaxActionSelector())

        # Creates two temporal agents just for "replaying" some parts
        # of the transition buffer
        self.t_q_agent = TemporalAgent(critic)
        self.t_target_q_agent = TemporalAgent(target_critic)

        # Get an agent that is executed on a complete workspace
        self.optimizer = setup_optimizer(cfg.optimizer, self.t_q_agent)

        self.last_critic_update_step = 0
        

def run_dqn(dqn: DQN, compute_critic_loss):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    
    for rb in dqn.iter_replay_buffers():
        for _ in range(dqn.cfg.algorithm.n_updates):
            rb_workspace = rb.get_shuffled(dqn.cfg.algorithm.batch_size)

            # The q agent needs to be executed on the rb_workspace workspace
            dqn.t_q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)
            with torch.no_grad():
                dqn.t_target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)

            q_values, terminated, reward, action, target_q_values = rb_workspace[
                "q_values", "env/terminated", "env/reward", "action", "target/q_values"
            ]

            # Determines whether values of the critic should be propagated
            must_bootstrap = ~terminated

            # Compute critic loss
            critic_loss = compute_critic_loss(
                dqn.cfg, reward, must_bootstrap, q_values, target_q_values, action
            )
            # Store the loss for tensorboard display
            dqn.logger.add_log("critic_loss", critic_loss, dqn.nb_steps)
            dqn.logger.add_log("q_values/min", q_values.max(-1).values.min(), dqn.nb_steps)
            dqn.logger.add_log("q_values/max", q_values.max(-1).values.max(), dqn.nb_steps)
            dqn.logger.add_log("q_values/mean", q_values.max(-1).values.mean(), dqn.nb_steps)
            
            dqn.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dqn.t_q_agent.parameters(), dqn.cfg.algorithm.max_grad_norm
            )
            dqn.optimizer.step()

            # Update target
            if (
                dqn.nb_steps - dqn.last_critic_update_step
                > dqn.cfg.algorithm.target_critic_update
            ):
                dqn.last_critic_update_step = dqn.nb_steps
                copy_parameters(dqn.t_q_agent, dqn.t_target_q_agent)
                torch.save(dqn.eval_policy.agents[0].state_dict(), mod_path / "pystk_actor.pth")

            dqn.evaluate()
            
    torch.save(dqn.eval_policy.agents[0].state_dict(), mod_path / "pystk_actor.pth")
    
    
def dqn_compute_critic_loss_autoreset_multidiscrete(
    cfg,
    reward: torch.Tensor,           # shape (2, B)
    must_bootstrap: torch.Tensor,   # shape (2, B)
    q_values: torch.Tensor,         # shape (2, B, A)
    target_q_values: torch.Tensor,  # shape (2, B, A)
    action: torch.LongTensor        # shape (2, B, n_dims)
):
    """
    Calcule la loss critique pour un DQN avec actions MultiDiscrete
    en mode auto-reset (on ne gère plus explicitement le done).

    :param cfg: Configuration (contient notamment le discount factor)
    :param reward: Tensor de récompenses, shape (2, B)
                   - reward[0] = r_t
                   - reward[1] = r_{t+1}
    :param must_bootstrap: Tensor indiquant s'il faut bootstrap,
                           shape (2, B) => must_bootstrap[1] pour s_{t+1}
    :param q_values: Q-values "online", shape (2, B, A)
                     - q_values[0] = Q(s_t, .)
    :param target_q_values: Q-values "cibles", shape (2, B, A)
                            - target_q_values[1] = Q^target(s_{t+1}, .)
    :param action: Action multi-discrète, shape (2, B, n_dims)
                   - action[0] = a_t
                   - action[1] = a_{t+1} (pour Double DQN, éventuellement)
    :return: Scalaire (loss critique)
    """

    # Par exemple pour SuperTuxKart, 7 dimensions d'action :
    action_sizes = [5, 2, 2, 2, 2, 2, 7]

    # Construction "à l’envers" des coefficients pour aplatir l'action multi-discrète en un seul entier
    # Exemple:
    #   action_sizes = [5, 2, 2, 2, 2, 2, 7]
    #   => coefficients = [2*2*2*2*2*7, 2*2*2*2*7, 2*2*2*7, 2*2*7, 2*7, 7, 1]
    # de sorte que flat_action = sum( action[i] * coefficients[i] )
    coeffs = [1] * len(action_sizes)
    for i in reversed(range(len(action_sizes) - 1)):
        coeffs[i] = coeffs[i + 1] * action_sizes[i + 1]

    coefficients = torch.tensor(coeffs, device=action.device, dtype=torch.long)

    # On "aplati" l'action de la frame t (action[0]) => shape (B,)
    # action[0] est de shape (B, n_dims)
    flat_action_0 = (action[0] * coefficients).sum(dim=-1)

    # Calcul de Q^target(s_{t+1}, .)
    max_q_tp1 = target_q_values[1].max(dim=-1).values.detach()  # shape (B,)

    # Construction de la cible DQN : r_{t+1} + gamma * max Q^target(s_{t+1}, .) * must_bootstrap[1]
    gamma = cfg.algorithm.discount_factor
    target = reward[1] + gamma * max_q_tp1 * must_bootstrap[1].float()  # shape (B,)

    # On récupère Q(s_t, a_t) => shape (B,)
    q_s_a = q_values[0].gather(1, flat_action_0.unsqueeze(-1)).squeeze(-1)

    # MSELoss entre la cible et la prédiction
    mse = nn.MSELoss()
    critic_loss = mse(target, q_s_a)

    return critic_loss

def ddqn_compute_critic_loss_autoreset_multidiscrete(
    cfg,
    reward: torch.Tensor,           # shape (2, B)
    must_bootstrap: torch.Tensor,   # shape (2, B)
    q_values: torch.Tensor,         # shape (2, B, A)
    target_q_values: torch.Tensor,  # shape (2, B, A)
    action: torch.LongTensor        # shape (2, B, n_dims)
):
    """
    Calcule la loss critique pour un Double DQN avec actions MultiDiscrete
    en mode auto-reset (on ne gère plus explicitement le done).

    :param cfg: Configuration (contient notamment le discount factor)
    :param reward: Tensor de récompenses, shape (2, B)
                   - reward[0] = r_t
                   - reward[1] = r_{t+1}
    :param must_bootstrap: Tensor indiquant s'il faut bootstrap,
                           shape (2, B) => must_bootstrap[1] pour s_{t+1}
    :param q_values: Q-values "online", shape (2, B, A)
                     - q_values[0] = Q(s_t, .)
                     - q_values[1] = Q(s_{t+1}, .) (réseau online)
    :param target_q_values: Q-values "cibles", shape (2, B, A)
                            - target_q_values[1] = Q^target(s_{t+1}, .)
    :param action: Action multi-discrète, shape (2, B, n_dims)
                   - action[0] = a_t
                   - action[1] = a_{t+1} (pour info, mais pas forcément utilisée ici)
    :return: Scalaire (loss critique)
    """

    # Exemple de tailles pour 7 dimensions d'action :
    action_sizes = [5, 2, 2, 2, 2, 2, 7]

    # Construction des coefficients pour aplatir l'action multi-discrète en un entier
    coeffs = [1] * len(action_sizes)
    for i in reversed(range(len(action_sizes) - 1)):
        coeffs[i] = coeffs[i + 1] * action_sizes[i + 1]

    coefficients = torch.tensor(coeffs, device=action.device, dtype=torch.long)

    # On "aplati" l'action de la frame t (action[0]) => shape (B,)
    flat_action_0 = (action[0] * coefficients).sum(dim=-1)

    # 1) Sélection de la meilleure action à t+1 via le réseau online (q_values[1])
    #    argmax sur la dimension des actions flatten (A).
    argmax_actions_tp1 = q_values[1].argmax(dim=-1)  # shape (B,)

    # 2) Récupération de la Q-value correspondante dans le réseau cible (target_q_values[1])
    max_q_tp1 = target_q_values[1].gather(1, argmax_actions_tp1.unsqueeze(-1)).squeeze(-1)  # shape (B,)

    # 3) Construction de la cible Double DQN :
    #    r_{t+1} + gamma * Q^target(s_{t+1}, argmax_a) * must_bootstrap[1]
    gamma = cfg.algorithm.discount_factor
    target = reward[1] + gamma * max_q_tp1 * must_bootstrap[1].float()  # shape (B,)

    # 4) Récupération de la Q-value (online) pour l'action réellement prise à t
    q_s_a = q_values[0].gather(1, flat_action_0.unsqueeze(-1)).squeeze(-1)  # shape (B,)

    # 5) Calcul de la perte (MSE)
    mse = nn.MSELoss()
    critic_loss = mse(target, q_s_a)

    return critic_loss