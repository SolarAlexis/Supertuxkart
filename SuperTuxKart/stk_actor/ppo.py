import copy
from pathlib import Path
import inspect

import torch
from bbrl_utils.algorithms import EpisodicAlgo, iter_partial_episodes
from bbrl.agents import TemporalAgent, KWAgentWrapper
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

        self.train_policy = globals()[cfg.algorithm.policy_type](
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
        

def run(ppo_clip: PPOClip):
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    i = 1
    
    cfg = ppo_clip.cfg
    

    t_policy = TemporalAgent(ppo_clip.train_policy)
    t_old_policy = TemporalAgent(ppo_clip.old_policy)
    t_critic = TemporalAgent(ppo_clip.critic_agent)
    t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)

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
            torch.save(ppo_clip.eval_policy.state_dict(), mod_path / "pystk_actor.pth")
            torch.save(ppo_clip.critic_agent.state_dict(), mod_path / "pystk_critic.pth")
            i = 0
        i+=1
    
    torch.save(ppo_clip.eval_policy.state_dict(), mod_path / "pystk_actor.pth")
    torch.save(ppo_clip.critic_agent.state_dict(), mod_path / "pystk_critic.pth")