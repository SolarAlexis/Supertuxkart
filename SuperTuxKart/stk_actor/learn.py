# import os

# Ajoutez le chemin contenant les DLLs
# os.add_dll_directory(r"D:\\projet_SuperTuxKart\\stk-code\\build\\bin\\Debug")

# import logging
# logging.basicConfig(level=logging.DEBUG)

from pystk2_gymnasium import AgentSpec
from functools import partial
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from omegaconf import OmegaConf


# Note the use of relative imports
from .pystk_actor import env_name, get_wrappers, player_name
from .config import params_dqn, params_ddpg, params_SAC, params_td3, params_ppo, params_TQC
from .dqn import DQN, run_dqn, dqn_compute_critic_loss_autoreset_multidiscrete, ddqn_compute_critic_loss_autoreset_multidiscrete
from .ddpg import DDPG, run_ddpg, compute_actor_loss, compute_critic_loss, TD3, run_td3
# from .SAC import SACAlgo, run_sac, compute_critic_loss, compute_actor_loss, setup_entropy_optimizers
from .ppo import PPOClip, run
from .TQC import run_tqc

if __name__ == "__main__":
    
    run_tqc(OmegaConf.create(params_TQC))
    
    # ppo_clip = PPOClip(OmegaConf.create(params_ppo))
    # run(ppo_clip)
    
    # td3 = TD3(OmegaConf.create(params_td3))
    # run_td3(td3, compute_critic_loss, compute_actor_loss)
    
    # sac = SACAlgo(OmegaConf.create(params_SAC))
    # run_sac(sac, compute_critic_loss, compute_actor_loss, setup_entropy_optimizers)
    
    #ddpg = DDPG(OmegaConf.create(params_ddpg))
    #run_ddpg(ddpg, compute_critic_loss, compute_actor_loss)
    
    
    # #dqn
    # dqn = DQN(OmegaConf.create(params_dqn))
    # run_dqn(dqn, dqn_compute_critic_loss_autoreset_multidiscrete)

    # #dqn
    # ddqn = DQN(OmegaConf.create(params_dqn))
    # run_dqn(ddqn, ddqn_compute_critic_loss_autoreset_multidiscrete)
    
    
    
    # # Setup the environment
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
    
    # ix = 0
    # done = False
    # state, *_ = env.reset()
    
    # print(workspace)
    # print("\nObservation space: ", env.observation_space)
    # print("\naction space: ", env.action_space)
