import os
import torch
from torch.optim import Adam

from model.utils import *
from model.networks import QuantileMlp, GaussianPolicy


class DSAC(object):
    def __init__(self, 
                 num_inputs, 
                 action_space,
                 # SAC params
                 gamma = 0.99,
                 alpha = 0.2,
                 soft_target_tau = 5e-3,
                 target_update_period = 1,
                 use_automatic_entropy_tuning = False,
                 target_entropy = -3,
                 hidden_size = 256,
                 critic_lr = 0.0003,
                 actor_lr = 3e-5,
                 num_random = 10,
                 min_z_weight = 10.0,
                 # Distributional params
                 num_quantiles = 32,
                 risk_type = 'cvar',
                 risk_param = 0.1,
                 tau_type = 'iqn'):

        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.risk_type = risk_type
        self.risk_param = risk_param

        self.tau_type = tau_type
        self.soft_target_tau = soft_target_tau
        self.target_update_period= target_update_period
        self.num_quantiles = num_quantiles
        self.num_random = num_random
        self.min_z_weight = min_z_weight

        # Distributional critic
        self.zf1 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                               output_size = 1,
                               num_quantiles = num_quantiles,
                               hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.zf2 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                               output_size = 1,
                               num_quantiles = num_quantiles,
                               hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.target_zf1 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                                      output_size = 1,
                                      num_quantiles = num_quantiles,
                                      hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.target_zf2 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                                      output_size = 1,
                                      num_quantiles = num_quantiles,
                                      hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.zf_criterion = quantile_regression_loss
        self.zf1_optimizer = Adam(self.zf1.parameters(), lr = critic_lr)
        self.zf2_optimizer = Adam(self.zf2.parameters(), lr = critic_lr)

        # Gaussian actor
        self.policy = GaussianPolicy(num_inputs = num_inputs, 
                                     num_actions = action_space.shape[0], 
                                     hidden_dim = hidden_size, 
                                     action_space = action_space).to(self.device)
        self.target_policy = GaussianPolicy(num_inputs = num_inputs, 
                                            num_actions = action_space.shape[0], 
                                            hidden_dim = hidden_size,
                                            action_space = action_space).to(self.device)
        self.actor_optimizer = Adam(self.policy.parameters(), lr=actor_lr)

        # Entropy temperature
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            if self.target_entropy != 'auto':
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=actor_lr)
        else:
            self.alpha = alpha

    def _get_tensor_values(self, obs, actions, tau):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        tau_temp = tau.unsqueeze(1).repeat(1, num_repeat, 1).view(tau.shape[0] * num_repeat, tau.shape[1])

        pred1 = self.zf1(obs_temp, actions, tau_temp)
        pred2 = self.zf2(obs_temp, actions, tau_temp)
        pred1 = pred1.view(obs.shape[0], num_repeat, -1)
        pred2 = pred2.view(obs.shape[0], num_repeat, -1)
        return pred1, pred2

    def get_tau(self, actions):
        """
        This method calculates:
            the cumulative probability of quantiles tau, 
            the center of quantiles tau-hat,
            the original cumulative probability density function presum_tau according to the given distribution type
        """
        if self.tau_type == 'fix':
            presum_tau = torch.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':
            # normalized random tensor: (num_action , num_quantiles)
            presum_tau = torch.rand(len(actions), self.num_quantiles) + 0.1    # add 0.1 to prevent tau getting too close
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)                # normalize

        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.ones_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.  
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def update_parameters(self, memory, batch_size, updates):
        """
        Get samples
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state = torch.FloatTensor(state_batch).to(self.device)
        next_state = torch.FloatTensor(next_state_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done = 1 - torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        """
        Update ZF 
        """
        with torch.no_grad():
            new_next_actions, next_log_pi, _ = self.target_policy.sample(next_state)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(new_next_actions)
            target_z1_values = self.target_zf1(next_state, new_next_actions, next_tau_hat)
            target_z2_values = self.target_zf2(next_state, new_next_actions, next_tau_hat)
            target_z_values = torch.min(target_z1_values, target_z2_values) - self.alpha * next_log_pi
            z_target = reward + (1. - done) * self.gamma * target_z_values

        tau, tau_hat, presum_tau = self.get_tau(action)
        z1_pred = self.zf1(state, action, tau_hat)
        z2_pred = self.zf2(state, action, tau_hat)
        
        self.zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        self.zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)

        self.zf1_optimizer.zero_grad()
        self.zf1_loss.backward(retain_graph=True)
        self.zf1_optimizer.step()

        self.zf2_optimizer.zero_grad()
        self.zf2_loss.backward(retain_graph=True)
        self.zf2_optimizer.step()

        """
        Update Policy
        """
        new_actions, log_pi, _ = self.policy.sample(state)
        with torch.no_grad():
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(state, new_actions)
        z1_new_actions = self.zf1(state, new_actions, new_tau_hat)
        z2_new_actions = self.zf2(state, new_actions, new_tau_hat)
        with torch.no_grad():
            risk_weights = distortion_de(new_tau_hat, self.risk_type, self.risk_param)
        q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
        q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        self.actor_loss = (self.alpha * log_pi - q_new_actions).mean()
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

        """
        Update Alpha
        """
        new_actions, log_pi, _ = self.policy.sample(state)
        if self.use_automatic_entropy_tuning:
            self.alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            self.alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            self.alpha_loss = torch.tensor(0.)
            alpha = self.alpha
            alpha_tlogs = torch.tensor(self.alpha)

        """
        Update Target Networks
        """
        if updates % self.target_update_period == 0:
            self.soft_update(self.policy, self.target_policy, self.soft_target_tau)
            self.soft_update(self.zf1, self.target_zf1, self.soft_target_tau)
            self.soft_update(self.zf2, self.target_zf2, self.soft_target_tau)

        return self.zf1_loss.item(), self.zf2_loss.item(), self.actor_loss.item(), self.alpha_loss.item(), alpha_tlogs.item()

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_model(self, path):
        if not os.path.exists('saved_policies/'):
            os.makedirs('saved_policies/')

        actor_path = path+'-actor.pt'
        critic1_path = path + '-critic1.pt'
        critic2_path = path + '-critic2.pt'
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.zf1.state_dict(), critic1_path)
        torch.save(self.zf2.state_dict(), critic2_path)

    def load_model(self, path):
        actor_path = path+'-actor.pt'
        critic1_path = path + '-critic1.pt'
        critic2_path = path + '-critic2.pt'
        self.policy.load_state_dict(torch.load(actor_path))
        self.zf1.load_state_dict(torch.load(critic1_path))
        self.zf2.load_state_dict(torch.load(critic2_path))