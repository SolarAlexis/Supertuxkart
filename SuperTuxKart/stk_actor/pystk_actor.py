
from typing import List, Callable
from bbrl.agents import Agents, Agent, KWAgentWrapper
import gymnasium as gym

from .actors import DiscreteQAgent, ArgmaxActionSelector, ContinuousDeterministicActor, SquashedGaussianActor, FeatureFilterWrapper, MyActionRescaleWrapper, DiscretePolicy

#: The base environment name
env_name = "supertuxkart/flattened_discrete-v0"

#: Player name
player_name = "SupramaXx"


# def get_actor(
#     state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
# ) -> Agent:
#     actor = ContinuousDeterministicActor(
#             observation_space, [256, 128], action_space
#         )

#     if state is None:
#         raise ValueError("No state available")

#     actor.load_state_dict(state)
#     return actor

# def get_actor(
#     state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
# ) -> Agent:
#     actor = SquashedGaussianActor(
#             observation_space, [256, 256], action_space
#         )

#     if state is None:
#         raise ValueError("No state available")

#     actor.load_state_dict(state)
#     return KWAgentWrapper(actor, stochastic=True)

def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = DiscretePolicy(
            observation_space,
            [256, 256],
            action_space,
        )

    if state is None:
        raise ValueError("No state available")

    actor = KWAgentWrapper(
            actor, 
            stochastic=False,
            predict_proba=False,
            compute_entropy=False,
        )
    actor.load_state_dict(state)
    return actor

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        #lambda env: MyWrapper(env, option="1")
        # lambda env: FeatureFilterWrapper(env, [0, 
        # 1, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        # 20, 21, 22, 23, 24, 25, 87, 88]),
        #lambda env: MyActionRescaleWrapper(env)
    ]

# class PPOClip(EpisodicAlgo):
#     def __init__(self, cfg):
#         super().__init__(cfg, env_wrappers=get_wrappers(), autoreset=True)
        
#         obs_space_size = self.train_env.envs[0].observation_space
#         action_space_size = self.train_env.envs[0].action_space
        
#         print("\nObservation space: ", obs_space_size)
#         print("\naction space: ", action_space_size)
        
#         self.train_policy = globals()[cfg.algorithm.policy_type](
#             obs_space_size,
#             cfg.algorithm.architecture.actor_hidden_size,
#             action_space_size,
#         ).with_prefix("current_policy/")
        
#         self.old_policy = copy.deepcopy(self.train_policy).with_prefix("old_policy/")

#         self.eval_policy = KWAgentWrapper(
#             self.train_policy, 
#             stochastic=False,
#             predict_proba=False,
#             compute_entropy=False,
#         )

#         self.critic_agent = VAgent(
#             obs_size, cfg.algorithm.architecture.critic_hidden_size
#         ).with_prefix("critic/")
#         self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")

#         self.old_policy = copy.deepcopy(self.train_policy).with_prefix("old_policy/")

#         self.policy_optimizer = setup_optimizer(
#             cfg.optimizer, self.train_policy
#         )
#         self.critic_optimizer = setup_optimizer(
#             cfg.optimizer, self.critic_agent
#         )

if __name__ == "__main__":
    
    from pystk2_gymnasium import AgentSpec
    from functools import partial
    from bbrl.agents.gymnasium import ParallelGymAgent, make_env
    from bbrl.workspace import Workspace
    import torch
    
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]
    workspace = Workspace()
    env_agent(workspace, t=0)
    
    ix = 0
    done = False
    state, *_ = env.reset()
    
    print(env.action_space)
    print(env.observation_space)
    print(workspace)
    
    # discrete = workspace["env/env_obs/discrete"][0]
    # continuous =  workspace["env/env_obs/continuous"][0]
    # obs = torch.cat([continuous, discrete], dim=1)
    # print(obs)
    # model = get_actor(torch.load("/home/alexis/SuperTuxKart/stk_actor/pystk_actor.pth", weights_only=True), env.observation_space, env.action_space)
    # print(model)
    # model(workspace, t=0)
    # print(workspace["action"])