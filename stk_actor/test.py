import os

# Ajoutez le chemin contenant les DLLs
os.add_dll_directory(r"D:\\projet_SuperTuxKart\\stk-code\\build\\bin\\Debug")

import gymnasium as gym
from pystk2_gymnasium import AgentSpec


# STK gymnasium uses one process
if __name__ == '__main__':
  # Use a a flattened version of the observation and action spaces
  # In both case, this corresponds to a dictionary with two keys:
  # - `continuous` is a vector corresponding to the continuous observations
  # - `discrete` is a vector (of integers) corresponding to discrete observations
  env_name = "supertuxkart/flattened_multidiscrete-v0"
  env = gym.make(env_name, render_mode="human", agent=AgentSpec(use_ai=False))

  ix = 0
  done = False
  state, *_ = env.reset()

  while not done and ix<100:
    
      action = env.action_space.sample()
      state, reward, terminated, truncated, _ = env.step(action)
      done = truncated or terminated
      ix += 1
      
      if ix == 1 and env_name == "supertuxkart/flattened-v0":
        
        print(f"Actions continues : {action['continuous'].shape}")
        print(f"Actions discretes : {action['discrete'].shape}")
        print(f"State continues : {state['continuous'].shape}")
        print(f"State discrets : {state['discrete'].shape}")
        print(f"Reward : {reward}")
        print(f"Done : {done}")
      
      if ix == 1 and (env_name == "supertuxkart/flattened_multidiscrete-v0" or "supertuxkart/flattened_continuous_actions-v0"):
        
        print(f"Actions {action}")
        print(f"State {state}")
        print(f"Reward : {reward}")
        print(f"Done : {done}")
        print(env.observation_space["discrete"].shape)
        print(env.observation_space["continuous"].shape)
        print(env.action_space)
      
  env.close()