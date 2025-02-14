import gymnasium as gym
import numpy as np
import pygame
import pickle
import time
from pystk2_gymnasium import AgentSpec

# Création de l'environnement
env_name = "supertuxkart/flattened_discrete-v0"
env = gym.make(env_name, render_mode="human", agent=AgentSpec(use_ai=False))
obs, info = env.reset()

# Initialisation de pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Manual Control - SuperTuxKart (Discrete)")
clock = pygame.time.Clock()

demo_data = []
max_episodes = 25
episode = 0

print("Contrôles :")
print("  Z = accélérer, M = boost")
print("  S = frein")
print("  Q = tourner à gauche, D = tourner à droite")
print("  LSHIFT = drift, SPACE = tir, N = nitro, R = rescue")
print("Par défaut, aucune touche => le kart est neutre (avance tout droit).")
time.sleep(2)

def encode_action():
    """
    Construit et encode l'action discrète attendue.
    L'ordre des composantes est :
       [acceleration, steer, brake, drift, fire, nitro, rescue]
    avec nvec = [5, 7, 2, 2, 2, 2, 2].

    Mapping choisi :
      - Acceleration (5 valeurs) :
          • Aucune touche → 2 (neutre)
          • Touche Z → 3
          • Touche M → 4
      - Steer (7 valeurs) :
          • Aucune touche → 3 (neutre)
          • Touche Q seule → 0 (gauche)
          • Touche D seule → 6 (droite)
      - Les autres actions : 1 si la touche est pressée, sinon 0.
    """
    # Mettre à jour l'état des événements
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    # Acceleration
    if keys[pygame.K_m]:
        acc = 4
    elif keys[pygame.K_z]:
        acc = 3
    else:
        acc = 2  # Par défaut : neutre (milieu de l'échelle 0-4)
    
    # Steer
    if keys[pygame.K_q] and not keys[pygame.K_d]:
        steer = 0
    elif keys[pygame.K_d] and not keys[pygame.K_q]:
        steer = 6
    else:
        steer = 3  # Par défaut : neutre (milieu de l'échelle 0-6)
    
    # Autres actions binaires
    brake  = int(keys[pygame.K_s])
    drift  = int(keys[pygame.K_LSHIFT])
    fire   = int(keys[pygame.K_SPACE])
    nitro  = int(keys[pygame.K_n])
    rescue = int(keys[pygame.K_r])
    
    # Vecteur d'action
    components = [acc, steer, brake, drift, fire, nitro, rescue]
    nvec = [5, 7, 2, 2, 2, 2, 2]
    
    # Flattening : encoder le vecteur en un entier
    flat_action = 0
    multiplier = 1
    for comp, base in zip(components, nvec):
        flat_action += comp * multiplier
        multiplier *= base
    return 987

running = True
while running and episode < max_episodes:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = encode_action()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    demo_data.append({
        'obs': obs,
        'action': action,
        'reward': reward,
        'next_obs': next_obs,
        'done': done
    })
    obs = next_obs

    env.render()
    pygame.display.flip()
    clock.tick(30)

    if done:
        episode += 1
        print(f"Episode {episode}/{max_episodes} terminé.")
        obs, info = env.reset()
        time.sleep(1)

env.close()
pygame.quit()

with open("demo_data.pkl", "wb") as f:
    pickle.dump(demo_data, f)

print(f"Enregistrement terminé : {len(demo_data)} transitions sauvegardées dans demo_data.pkl")
