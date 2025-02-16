import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import pygame
import pickle
import time
import numpy as np
from typing import List, Callable

# =============================================================================
# Vos Wrappers
# =============================================================================

class FeatureFilterWrapper(gym.ObservationWrapper):
    def __init__(self, env, index):
        super(FeatureFilterWrapper, self).__init__(env)
        self.index = index
        
        continuous_space = self.env.observation_space["continuous"]
        low = np.delete(continuous_space.low, index)
        high = np.delete(continuous_space.high, index)
        filtered_continuous_space = gym.spaces.Box(low=low, high=high, dtype=continuous_space.dtype)
        self.observation_space = gym.spaces.Dict({"continuous": filtered_continuous_space})

    def observation(self, obs):
        # On part du principe que obs est un dictionnaire avec la clé "continuous"
        partial_obs = np.delete(obs["continuous"], self.index)
        return {"continuous": partial_obs}

class MyActionRescaleWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1,  1], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, agent_action):
        """
        Transforme l'action de l'agent (dans [-1,1] pour chacune des 2 composantes)
        en une action valide pour l'environnement ([0,1] pour la première composante
        et [-1,1] pour la deuxième).
        """
        env_action = np.array(agent_action, copy=True, dtype=np.float32)
        env_action[..., 0] = (env_action[..., 0] + 1.0) / 2.0
        return env_action

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Retourne la liste de wrappers à appliquer à l'environnement."""
    return [
        lambda env: FeatureFilterWrapper(env, [0, 1, 7, 87, 88]),
        lambda env: MyActionRescaleWrapper(env)
    ]

# =============================================================================
# Création de l'environnement avec wrappers
# =============================================================================

env_name = "supertuxkart/flattened_continuous_actions-v0"
env = gym.make(env_name, render_mode="human", agent=AgentSpec(use_ai=False))

# Appliquer les wrappers
for wrapper in get_wrappers():
    env = wrapper(env)

obs, info = env.reset()

# Afficher la structure de l'observation pour vérifier l'effet des wrappers
if isinstance(obs, dict):
    print("Clés de l'observation :", list(obs.keys()))
    print("Forme de l'observation 'continuous' :", np.array(obs["continuous"]).shape)
else:
    print("Observation de type", type(obs), "de forme", np.array(obs).shape)

# =============================================================================
# Configuration de pygame pour la capture des entrées clavier
# =============================================================================

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Conduite manuelle - SuperTuxKart")
clock = pygame.time.Clock()

# =============================================================================
# Collecte des démonstrations
# =============================================================================

demo_data = []  # Liste pour stocker les transitions
episode = 0
max_episodes = 1  # Nombre d'épisodes de démonstration à collecter

print("Contrôlez le kart avec les flèches :")
print(" - Haut : accélérer doucement")
print(" - Bas : freiner doucement")
print(" - Gauche/Droite : tourner doucement")
time.sleep(2)  # Temps pour se préparer

running = True
while running and episode < max_episodes:
    # Gestion des événements pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Lecture de l'état du clavier
    keys = pygame.key.get_pressed()
    acc = -0.6
    steer = 0.0
    
    if keys[pygame.K_z]:
        acc = -0.6    # accélération réduite pour une conduite plus douce
    if keys[pygame.K_m]:
        acc = 1     # boost
    if keys[pygame.K_s]:
        acc = -1   
    if keys[pygame.K_q]:
        steer = -0.7
    if keys[pygame.K_d]:
        steer = 0.7
    
    # Construire l'action (vecteur à 2 composantes)
    action = np.array([acc, steer])
    
    # Effectuer un pas dans l'environnement
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Enregistrer la transition
    # Ici, on enregistre directement l'observation renvoyée par l'environnement (avec wrappers)
    demo_data.append({
        'obs': obs,
        'action': action,
        'reward': reward,
        'done': done
    })
    
    # Mise à jour de l'observation
    obs = next_obs

    # Rafraîchir l'affichage et limiter à 30 fps
    pygame.display.flip()
    clock.tick(30)

    # Si l'épisode est terminé, réinitialiser l'environnement
    if done:
        episode += 1
        print(f"Episode {episode}/{max_episodes} terminé.")
        obs, info = env.reset()
        time.sleep(1)

env.close()
pygame.quit()

# =============================================================================
# Sauvegarde des démonstrations
# =============================================================================

save_path = "/home/alexis/SuperTuxKart/stk_actor/demo_data53.pkl"
with open(save_path, "wb") as f:
    pickle.dump(demo_data, f)

print(f"Enregistrement terminé : {len(demo_data)} transitions sauvegardées dans {save_path}.")