import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model.networks import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, 
                 num_inputs, 
                 action_space,
                 gamma = 0.99, 
                 tau = 0.005, 
                 alpha = 0.2,
                 policy = 'Gaussian',
                 target_update_interval = 1,
                 automatic_entropy_tuning = True,
                 target_entropy = -3,
                 hidden_size = 256,
                 lr = 0.0003):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                if self.target_entropy != 'auto':
                    self.target_entropy = target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)
            else:
                self.alpha = alpha
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        else:
            self.alpha = alpha
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        """
        Get samples
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        """
        Update QF 
        """
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)           # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)           # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        """
        Update Policy 
        """
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """
        Update Alpha 
        """
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        """
        Update Target Networks
        """
        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists('saved_policies/'):
            os.makedirs('saved_policies/')

        actor_path = path+'-actor.pt'
        critic_path = path + '-critic.pt'
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, path):
        actor_path = path+'-actor.pt'
        critic_path = path + '-critic.pt'
        self.policy.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
