"""
Run PyTorch Soft Actor Critic on Pendulum-v0.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
import random
import gym
import copy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.virel import Virel
from rlkit.torch.networks import FlattenMlp
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
import sys

def experiment(variant):
    eval_env = gym.make(variant['env'])
    expl_env = gym.make(variant['env'])
    obs_dim = int(np.prod(eval_env.observation_space.shape))
    action_dim = int(np.prod(eval_env.action_space.shape))

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
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)
    eval_path_collector = MdpPathCollector(eval_env, policy)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=EpsilonGreedy(action_space=expl_env.action_space),
        policy=policy,
    )
    expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)
    
    trainer = Virel(
        env=eval_env,
        qf=qf,
        vf=vf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":    
    #env_name = 'Pendulum-v0'
    env_name = 'MountainCarContinuous-v0'
    #env_name = 'BipedalWalker-v2' ## Can't pickle SwigPyObject objects
    epochs=100
    reward_scale = 1.0 ##1.0 is better than 3
    logger_name = "-"
    init_seed = 1
    beta_scale = 0.004
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=int(epochs),
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=1000,
            batch_size=128,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            reward_scale=float(reward_scale),
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        replay_buffer_size=int(1E6),
        net_size=300,
        env=env_name,
        #algo_seed=int(init_seed),
    )
    seed=int(1)
    random.seed(seed)
    np.random.seed(seed)
    name = "virel_" + "_" + str(env_name) + "_" + str(init_seed) + "_" + str(reward_scale) + "_" + str(beta_scale)
    setup_logger(name, variant=variant)
    ptu.set_gpu_mode(True)
    experiment(variant)
    
    