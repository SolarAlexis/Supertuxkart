import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import pygame
import pickle
import time
import numpy as np
from typing import List, Callable

# =============================================================================
# Création de l'environnement
# =============================================================================

env_name = "supertuxkart/flattened_discrete-v0"
env = gym.make(env_name, render_mode="human", agent=AgentSpec(use_ai=False))

obs, info = env.reset()

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
max_episodes = 10  # Nombre d'épisodes de démonstration à collecter
time.sleep(2)  # Temps pour se préparer

running = True
while running and episode < max_episodes:
    # Gestion des événements pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Lecture de l'état du clavier
    keys = pygame.key.get_pressed()
    action = 601
    
    if keys[pygame.K_z]:
        action = 601 
    if keys[pygame.K_m]:
        action = 604    # boost
    if keys[pygame.K_s]:
        action = 605
    if keys[pygame.K_q]:
        action = 201  
    if keys[pygame.K_d]:
        action = 801  # tourner à droite en douceur
    
    # Effectuer un pas dans l'environnement
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Enregistrer la transition
    # Ici, on enregistre directement l'observation renvoyée par l'environnement (avec wrappers)
    demo_data.append({
        'obs': obs,
        'action': action,
        'reward': reward,
        'next_obs': next_obs,
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

save_path = "/home/alexis/SuperTuxKart/stk_actor/demo_data11.pkl"
with open(save_path, "wb") as f:
    pickle.dump(demo_data, f)

print(f"Enregistrement terminé : {len(demo_data)} transitions sauvegardées dans {save_path}.")