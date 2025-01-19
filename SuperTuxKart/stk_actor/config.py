params_dqn = {
    "base_dir": "${gym_env.env_name}/dqn-S${algorithm.seed}_${current_time:}",
    # `collect_stats` is True: we keep the cumulated reward for all
    # evaluation episodes
    "collect_stats": True,
    "save_best": True,
    "algorithm": {
        "seed": 2,
        "max_grad_norm": 1,
        "epsilon": 0.05,
        "n_envs": 8,
        "n_steps": 64,
        "n_updates": 64,
        "eval_interval": 10_000,
        "learning_starts": 5_000,
        "nb_evals": 7,
        "buffer_size": 20_000,
        "batch_size": 256,
        "target_critic_update": 1_000,
        "max_epochs": 206,
        "discount_factor": 0.99,
        "architecture": {"hidden_size": [128, 256, 512]},
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_multidiscrete-v0",
    },
    "optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-3,
    },
}

params_ddpg = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/ddpg-S${algorithm.seed}_${current_time:}",
    "collect_stats": True,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "algorithm": {
        "seed": 2,
        "max_grad_norm": 1,
        "epsilon": 0.02,
        "n_envs": 8,
        "n_steps": 100,
        "nb_evals": 7,
        "discount_factor": 0.98,
        "buffer_size": 2e5,
        "batch_size": 64,
        "tau_target": 0.05,
        "eval_interval": 10_000,
        "max_epochs": 250,
        # Minimum number of transitions before learning starts
        "learning_starts": 10000,
        "action_noise": 0.1,
        "architecture": {
            "actor_hidden_size": [128, 64, 32],
            "critic_hidden_size": [128, 64, 32],
        },
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_continuous_actions-v0",
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-3,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-3,
    },
}

params_td3 = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": True,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": False,
    "coef_steer_penalty": 10,
    "coef_extreme_steer_penalty": 20,
    "coef_acceleration_penalty": 5,
    "algorithm": {
        "seed": 1,
        "max_grad_norm": 0.5,
        "epsilon": 0.02,
        "n_envs": 64,
        "n_steps": 2,
        "nb_evals": 7,
        "discount_factor": 0.99,
        "buffer_size": 260_000, #1e6
        "batch_size": 512,
        "tau_target": 0.001,
        "eval_interval": 60_000,
        "max_epochs": 2000,
        # Minimum number of transitions before learning starts
        "learning_starts": 15_000,
        "action_noise": 0.4,
        "architecture": {
            "actor_hidden_size": [256, 128],
            "critic_hidden_size": [256, 256],
        },
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_continuous_actions-v0",
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-4,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 5e-4,
    },
}

params_SAC = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/sac-S${algorithm.seed}_${current_time:}",
    "algorithm": {
        "seed": 1,
        "n_envs": 8,
        "n_steps": 32,
        "buffer_size": 1e6, #20%
        "batch_size": 256,
        "max_grad_norm": 1, #0.5
        "nb_evals": 7, #7
        "eval_interval": 100_000, #20 000
        "learning_starts": 2_000, #10 000
        "max_epochs": 2000, #250
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 1,
        "tau_target": 0.01,
        "architecture": {
            "actor_hidden_size": [256, 256],
            "critic_hidden_size": [256, 256],
        },
    },
    "gym_env": {"env_name": "supertuxkart/flattened_continuous_actions-v0"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4, #3e-4
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
}

params_ppo = {
    "base_dir": "${gym_env.env_name}/ppo-clip-S${algorithm.seed}_${current_time:}",
     "save_best": False,
    "logger": {
        "classname": "bbrl.utils.logger.TFLogger",
        "cache_size": 10000,
        "every_n_seconds": 10,
        "verbose": False,
    },
    "algorithm": {
        "seed": 12,
        "max_grad_norm": 0.5,
        "n_envs": 8,
        "n_steps": 128,
        "eval_interval": 40_000,
        "nb_evals": 7,
        "gae": 0.95,
        "discount_factor": 0.99,
        "normalize_advantage": True,
        "max_epochs": 320,
        "opt_epochs": 10,
        "batch_size": 256,
        "clip_range": 0.2,
        "clip_range_vf": 0,
        "entropy_coef": 0.1, #2e-7
        "policy_coef": 1,
        "critic_coef": 1.0,
        "policy_type": "DiscretePolicy",
        "architecture": {
            "actor_hidden_size": [256, 256],
            "critic_hidden_size": [256, 256],
        },
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_discrete-v0",
    },
    "optimizer": {
        "classname": "torch.optim.AdamW",
        "lr": 3e-4,
        "eps": 1e-5,
    },
}