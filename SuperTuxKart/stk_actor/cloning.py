import numpy as np
import torch
from bbrl.agents import Agents
from bbrl.workspace import Workspace
from pystk2_gymnasium import AgentSpec
from functools import partial
from gymnasium import make

from .pystk_actor import player_name, get_wrappers

def collect_demos(cfg, nb_episodes=50, demo_path="demos.npy"):
    # Créez un environnement pour la collecte en mode non stochastique
    env_agent, _ = get_env_agents(cfg, autoreset=True, include_last_state=True)
    
    # On utilise un agent "expert" pour piloter.
    # Par exemple, vous pouvez utiliser une policy experte (ici, c'est à vous de définir
    # comment piloter l'agent manuellement ou via un contrôleur pré-défini).
    # Ici, nous allons supposer que vous avez une fonction expert_policy(obs) qui renvoie
    # une action (un vecteur de 2 valeurs dans [-1,1]).
    def expert_policy(obs):
        # Exemple très simple : toujours aller tout droit sans tourner.
        # Remplacez ceci par votre code pour piloter manuellement ou par une policy experte.
        return np.array([1.0, 0.0], dtype=np.float32)
    
    demos = []
    for ep in range(nb_episodes):
        workspace = Workspace()
        # Remarquez qu'on utilise stochastic=False pour que l'environnement
        # ne rajoute pas de bruit et que l'agent exécute exactement la policy que l'on donne.
        env_agent(workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=False)
        
        # Récupérer les transitions
        transitions = workspace.get_transitions()
        # On peut par exemple récupérer les observations et actions
        # Ici, j'utilise des clés génériques, adaptez selon ce que renvoie votre Workspace.
        demo_episode = {
            "observations": transitions["env/obs"],  # ou la clé correspondante
            "actions": transitions["action"],
        }
        demos.append(demo_episode)
        print(f"Episode {ep+1}/{nb_episodes} collecté.")
    
    # Sauvegarde des démonstrations
    np.save(demo_path, demos)
    print(f"Démonstrations sauvegardées dans {demo_path}")

# Exemple d'utilisation :
if __name__ == "__main__":
    # Chargez votre configuration (ici, cfg) comme dans votre code principal
    from your_config_module import cfg  # adaptez cet import
    collect_demos(cfg, nb_episodes=50, demo_path="demos.npy")
