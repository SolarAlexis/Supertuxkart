import torch
import numpy as np

def compute_critic_loss_dqn(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    done: torch.Tensor,
    q_values: torch.Tensor,
    action: torch.LongTensor,
) -> torch.Tensor:
    """
    Compute the temporal difference (TD) loss for DQN with MultiDiscrete actions.

    :param cfg: Configuration object with algorithm parameters
    :param reward: Tensor of shape (T, B) containing rewards
    :param must_bootstrap: Tensor of shape (T, B) indicating transitions to bootstrap
    :param done: Tensor of shape (T, B) indicating if episodes are terminated
    :param q_values: Tensor of shape (T, B, A) containing Q-values for all actions
    :param action: Tensor of shape (T, B, n_dims) containing chosen actions in MultiDiscrete format
    :return: Scalar Tensor representing the critic loss
    """
    # Dimensions MultiDiscrete
    action_sizes = [5, 2, 2, 2, 2, 2, 7]

    # Calculer les coefficients pour convertir MultiDiscrete -> indices plats
    coefficients = torch.tensor(
        [torch.prod(torch.tensor(action_sizes[i + 1 :])) for i in range(len(action_sizes))]
    )

    # Conversion des actions MultiDiscrete en indices plats
    flat_action = (action * coefficients).sum(dim=-1).long()  # Ensure dtype is int64

    # Calcul des Q-valeurs maximums pour les actions suivantes
    max_q = q_values.max(-1).values.detach()  # Shape: (T, B)

    # Calcul de la cible TD : R + γ * max(Q(s', a')) * must_bootstrap
    target = reward[1:] + cfg.algorithm.discount_factor * max_q[1:] * must_bootstrap[1:].int()

    # Récupérer Q(s, a) pour les actions choisies
    qsa_t = q_values.gather(2, flat_action.unsqueeze(-1)).squeeze(-1)  # Shape: (T, B)

    # Masque pour les transitions valides (non terminées)
    not_done = (~done[:-1]).int()  # Shape: (T-1, B)

    # Calcul de la perte TD
    td = (target - qsa_t[:-1]) ** 2 * not_done  # Shape: (T-1, B)

    # Moyenne pondérée sur les transitions valides
    critic_loss = td.sum() / not_done.sum()

    return critic_loss