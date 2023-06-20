import os
import torch
from torch.optim import Adam
import numpy as np

from model.utils import *
from model.networks import QuantileMlp, GaussianPolicy


class CODAC(object):
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
                 alpha_lr = 3e-4,
                 critic_lr = 3e-4,
                 actor_lr = 3e-5,
                 num_random = 10,
                 min_z_weight = 10.0,
                 with_lagrange = True,
                 lagrange_thresh = 10.0,
                 # distributional params
                 num_quantiles = 32,
                 tau_type = 'iqn',
                 dist_penalty_type = 'uniform',
                 risk_type = 'cvar',
                 risk_param = 0.1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma                               # discount factor
        self.dist_penalty_type = dist_penalty_type       # distribution penalty type
        self.risk_type = risk_type                       # risk type
        self.risk_param = risk_param                     # risk parameter

        self.num_quantiles = num_quantiles              # number of quantiles
        self.num_random = num_random                    # number of random samples 
        self.min_z_weight = min_z_weight                # minimum weight for the minimum quantile
        self.tau_type = tau_type                      

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        self.zf1 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                               output_size = 1,
                               num_quantiles = num_quantiles,
                               hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.zf2 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                               output_size = 1,
                               num_quantiles = num_quantiles,
                               hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.target_zf1 = QuantileMlp(input_size = num_inputs+action_space.shape[0],
                                      output_size = 1,
                                      num_quantiles = num_quantiles,
                                      hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.target_zf2 = QuantileMlp(input_size = num_inputs + action_space.shape[0],
                                      output_size = 1,
                                      num_quantiles = num_quantiles,
                                      hidden_sizes = [hidden_size, hidden_size]).to(self.device)
        self.zf_criterion = quantile_regression_loss
        self.zf1_optimizer = Adam(params = self.zf1.parameters(), lr = critic_lr)
        self.zf2_optimizer = Adam(params = self.zf2.parameters(), lr = critic_lr)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.use_automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            if self.target_entropy != 'auto':
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=actor_lr)
        else:
            self.alpha = alpha

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.target_policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)

        self.optimizer_actor = Adam(self.policy.parameters(), lr=actor_lr)

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_prime_optimizer = Adam([self.log_alpha_prime], lr=alpha_lr)

    def _get_tensor_values(self, obs, actions, tau):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        tau_temp = tau.unsqueeze(1).repeat(1, num_repeat, 1).view(
            tau.shape[0] * num_repeat, tau.shape[1])

        pred1 = self.zf1(obs_temp, actions, tau_temp)
        pred2 = self.zf2(obs_temp, actions, tau_temp)
        pred1 = pred1.view(obs.shape[0], num_repeat, -1)
        pred2 = pred2.view(obs.shape[0], num_repeat, -1)
        return pred1, pred2

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, new_obs_log_pi, _ = network.sample(obs_temp)
        return new_obs_actions.detach(), new_obs_log_pi.view(obs.shape[0], num_actions, 1).detach()

    def select_action(self, state, eval=False, tau=0.1):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

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
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size = batch_size)

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

        # penalty term (similar to CQL)
        if self.dist_penalty_type != 'none':
            random_actions_tensor = torch.FloatTensor(z2_pred.shape[0] * self.num_random, action.shape[-1]).uniform_(-1, 1).to(self.device) # OOD actions
            curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs = state,
                                                                         num_actions = self.num_random,
                                                                         network = self.policy)
            new_curr_actions_tensor, new_log_pis = self._get_policy_actions(obs = next_state,
                                                                            num_actions = self.num_random,
                                                                            network = self.policy)
            penalty_index = np.random.randint(0, self.num_quantiles)
            truncated_tau_hat = tau_hat[:, penalty_index: penalty_index + 1] # truncate tau_hat to be concervative

            # get the values of the random actions, current actions, and next actions
            z1_rand, z2_rand = self._get_tensor_values(state, random_actions_tensor, truncated_tau_hat)   
            z1_curr_actions, z2_curr_actions = self._get_tensor_values(state, curr_actions_tensor, truncated_tau_hat)
            z1_next_actions, z2_next_actions = self._get_tensor_values(state, new_curr_actions_tensor, truncated_tau_hat)

            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_z1 = torch.cat(
                [z1_rand - random_density, 
                 z1_next_actions - new_log_pis.detach(),
                 z1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_z2 = torch.cat(
                [z2_rand - random_density, 
                 z2_next_actions - new_log_pis.detach(),
                 z2_curr_actions - curr_log_pis.detach()], 1
            )

            min_zf1_loss = torch.logsumexp(cat_z1, dim=1, ).mean()
            min_zf2_loss = torch.logsumexp(cat_z2, dim=1, ).mean()

            min_zf1_loss = (min_zf1_loss - z1_pred.mean()) * self.min_z_weight
            min_zf2_loss = (min_zf2_loss - z2_pred.mean()) * self.min_z_weight

            if self.with_lagrange:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                min_zf1_loss = alpha_prime * (min_zf1_loss - self.target_action_gap)
                min_zf2_loss = alpha_prime * (min_zf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_zf1_loss - min_zf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            self.zf1_loss = self.zf1_loss + min_zf1_loss
            self.zf2_loss = self.zf2_loss + min_zf2_loss

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
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(new_actions)

        z1_new_actions = self.zf1(state, new_actions, new_tau_hat)
        z2_new_actions = self.zf2(state, new_actions, new_tau_hat)

        with torch.no_grad():
            risk_weights = distortion_de(new_tau_hat, self.risk_type, self.risk_param)

        q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
        q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        self.actor_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.optimizer_actor.zero_grad()
        self.actor_loss.backward()
        self.optimizer_actor.step()

        """
        Update Alpha
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
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
        Soft Updates
        """
        if updates % self.target_update_period == 0:
            self.soft_update(self.policy, self.target_policy,self.soft_target_tau)
            self.soft_update(self.zf1, self.target_zf1, self.soft_target_tau)
            self.soft_update(self.zf2, self.target_zf2, self.soft_target_tau)

        return self.zf1_loss.item(), self.zf2_loss.item(), self.actor_loss.item(), self.alpha_loss.item(), alpha_tlogs.item()

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists('saved_policies/'):
            os.makedirs('saved_policies/')

        actor_path = path + '-actor.pt'
        critic1_path = path + '-critic1.pt'
        critic2_path = path + '-critic2.pt'
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.zf1.state_dict(), critic1_path)
        torch.save(self.zf2.state_dict(), critic2_path)

    # Load model parameters
    def load_model(self, path):
        actor_path = path + '-actor.pt'
        critic1_path = path + '-critic1.pt'
        critic2_path = path + '-critic2.pt'
        self.policy.load_state_dict(torch.load(actor_path))
        self.zf1.load_state_dict(torch.load(critic1_path))
        self.zf2.load_state_dict(torch.load(critic2_path))
