from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import copy
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class Virel(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,
            
            discount=0.99,
            reward_scale=1.0,
            
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=False, ## No reparametrization in discrete
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
            **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super().__init__()
        
        self.env = env
        self.policy = policy
        self.discount=discount
        self.reward_scale = reward_scale
        
        self.beta = 1.0
        self.beta_batch_size = 4096
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.target_vf = copy.deepcopy(vf)
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        

    def _update_beta(self):
        if self.replay_buffer._size > self.beta_batch_size:
            batch_beta = self.get_batch_custom(self.beta_batch_size)
            rewards_beta = batch_beta['rewards']
            terminals_beta = batch_beta['terminals']
            obs_beta = batch_beta['observations']
            actions_beta = batch_beta['actions']
            next_obs_beta = batch_beta['next_observations']
            with torch.no_grad():
                q_pred_beta = self.qf(obs_beta, actions_beta)
                v_pred_beta = self.vf(next_obs_beta)
                q_target_beta = rewards_beta + (1. - terminals_beta) * self.discount * v_pred_beta
                self.beta = self.qf_criterion(q_pred_beta, q_target_beta)
        else:
            self.beta = 1.0

    def train_from_torch(self,batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions'] ## For the first time, this comes from experience replay buffer
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions) ## send only observations as input?
        q_pred = q_pred.gather(dim=1, index=actions.max(-1)[1].unsqueeze(dim=1)) ## Get the action from Q-Function that is respective to the actions 
        
        v_pred = self.vf(obs)
        policy_outputs = self.policy(
                obs,
                reparameterize=self.train_policy_with_reparameterization,
                return_log_prob=True,
        )
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        

        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf(obs, new_actions) ## Should be able to remove new_actions, since in discrete mode the input becomes (obs, new_actions) --> (obs). But if we do so V _target is exactly the same a s Q in above
        q_new_actions = q_new_actions.gather(dim=1, index=new_actions.max(-1)[1].unsqueeze(dim=1)) ## Get the action from Q-Function that is respective to the log_pi (i.e., new action)
        v_target = q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        Looks like alpha is similar scaling factor to beta
        """
        if self.train_policy_with_reparameterization:
            policy_loss = (self.beta*log_pi - q_new_actions).mean()
        else:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_pi * (self.beta*log_pi - log_policy_target).detach()
            ).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        ## No need for pre_tan_value in discrete space
        ##pre_tanh_value = policy_outputs[-1]
        ##pre_activation_reg_loss = self.policy_pre_activation_weight * (
        ##    (pre_tanh_value**2).sum(dim=1).mean()
        ##)
        policy_reg_loss = mean_reg_loss ##+ std_reg_loss ##+ pre_activation_reg_loss REMEMBER to decide about the std_reg_loss 
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        
    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )
        return snapshot
