"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
import random
import gym
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.virel import Virel
from rlkit.torch.networks import FlattenMlp
import sys

def experiment(variant):
    env = NormalizedBoxEnv(gym.make(variant['env']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = Virel(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":    
    env_name = 'car_racing-v4'
    epochs=10
    reward_scale = 3.0
    logger_name = "-"
    init_seed = 1
    beta_scale = 0.004
    variant = dict(
        algo_params=dict(
            num_epochs=int(epochs),
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,
            reward_scale=float(reward_scale),

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
        env=env_name,
        algo_name="virel",
        algo_seed=int(init_seed),
    )
    seed=int(1)
    random.seed(seed)
    np.random.seed(seed)
    name = "virel_" + "_" + str(env_name) + "_" + str(init_seed) + "_" + str(reward_scale) + "_" + str(beta_scale)
    setup_logger(name, variant=variant)
    ptu.set_gpu_mode(True)
    experiment(variant)
