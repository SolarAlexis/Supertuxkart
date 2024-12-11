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
        "n_steps": 32,
        "eval_interval": 1000,
        "nb_evals": 10,
        "gae": 0.8,
        "discount_factor": 0.98,
        "normalize_advantage": False,
        "max_epochs": 5_000,
        "opt_epochs": 10,
        "batch_size": 256,
        "clip_range": 0.2,
        "clip_range_vf": 0,
        "entropy_coef": 2e-7,
        "policy_coef": 1,
        "critic_coef": 1.0,
        "policy_type": "DiscretePolicy",
        "architecture": {
            "actor_hidden_size": [64, 64],
            "critic_hidden_size": [64, 64],
        },
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_multidiscrete-v0",
    },
    "optimizer": {
        "classname": "torch.optim.AdamW",
        "lr": 1e-3,
        "eps": 1e-5,
    },
}

params_dqn = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/dqn-simple-S${algorithm.seed}_${current_time:}",
    "collect_stats": True,
    "algorithm": {
        "seed": 3,
        "max_grad_norm": 0.5,
        "epsilon": 0.1,
        "n_envs": 1,
        "eval_interval": 500,
        "max_epochs": 10,
        "nb_evals": 1,
        "discount_factor": 0.99,
        "architecture": {"hidden_size": [256, 512]},
    },
    "gym_env": {
        "env_name": "supertuxkart/flattened_multidiscrete-v0",
    },
    "optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 2e-3,
    },
}

