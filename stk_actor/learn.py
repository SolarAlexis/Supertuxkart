import os

# Ajoutez le chemin contenant les DLLs
os.add_dll_directory(r"D:\\projet_SuperTuxKart\\stk-code\\build\\bin\\Debug")

from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.utils.functional import gae
from bbrl_utils.nn import setup_optimizer
from bbrl.workspace import Workspace
from bbrl.agents import Agent, Agents, KWAgentWrapper, TemporalAgent
import copy
from omegaconf import OmegaConf


# Note the use of relative imports
from actors import Actor, VAgent
from pystk_actor import env_name, get_wrappers, player_name, PPOClip, EpisodicDQN
from config import params_ppo, params_dqn

if __name__ == "__main__":
    dqn = EpisodicDQN(OmegaConf.create(params_dqn))
    dqn.run()
    # Setup the environment
    # make_stkenv = partial(
    #     make_env,
    #     env_name,
    #     wrappers=get_wrappers(),
    #     render_mode=None,
    #     autoreset=True,
    #     agent=AgentSpec(use_ai=False, name=player_name),
    # )
    # env_agent = ParallelGymAgent(make_stkenv, 1)
    # env = env_agent.envs[0]
    # workspace = Workspace()
    # env_agent(workspace, t=0)
    # print(workspace)
    
    # (2) Learn
    
    # ppo_clip = PPOClip(OmegaConf.create(params_ppo))
    # cfg = ppo_clip.cfg
    
    # print(ppo_clip.train_policy)
    
    # t_policy = TemporalAgent(ppo_clip.train_policy)
    # t_old_policy = TemporalAgent(ppo_clip.old_policy)
    #t_critic = TemporalAgent(ppo_clip.critic_agent)
    #t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)
    
    
    # ix = 0
    # done = False
    # state, *_ = env.reset()
    
    # train_policy = Actor(env.observation_space, params["actor_hidden_size"], env.action_space).with_prefix("current_policy/")
    # old_policy = copy.deepcopy(train_policy)
    # critic = VAgent(
    #         env.observation_space, params["critic_hidden_size"]
    # ).with_prefix("critic/")
    # old_critic = copy.deepcopy(critic).with_prefix("old_critic/")
    
    # t_policy = TemporalAgent(train_policy)
    # t_old_policy = TemporalAgent(old_policy)
    # t_critic = TemporalAgent(critic)
    # t_old_critic = TemporalAgent(old_critic)
    
    # policy_optimizer = setup_optimizer(
    #     params["optimizer"], train_policy
    # )
    # critic_optimizer = setup_optimizer(
    #     params["optimizer"], critic
    # )
    
    # print(workspace)
    # print("\nObservation space: ", env.observation_space)
    # print("\naction space: ", env.action_space)
    # print("\nactor model: ", train_policy.model)
    
    # (3) Save the actor sate
    
    # mod_path = Path(inspect.getfile(get_wrappers)).parent
    # print(mod_path)
    # torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
