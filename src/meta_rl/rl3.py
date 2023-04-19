import torch
import torch.nn as nn
from torch.distributions import  Normal, Categorical, MixtureSameFamily
from linear import LinearGroupHS, LinearSVDO, LinearGaussian, LinearMoG
import torch.nn.functional as F
from recurrent import ProbabilisticGRUCell
import numpy as np

class RL3(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior):
        super(RL3, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden

        self.initial = nn.Parameter(0.00 * torch.randn(1, self.num_hidden), requires_grad=False)

        self.gru = ProbabilisticGRUCell(num_states, num_hidden, prior)
        if prior == 'grouphs':
            self.mu = LinearGroupHS(num_hidden, num_actions)
        elif prior == 'gaussian':
            self.mu = LinearGaussian(num_hidden, num_actions)
        elif prior == 'svdo':
            self.mu = LinearSVDO(num_hidden, num_actions)

    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])
        return self.mu(hx, zeta[1]), hx

    def act(self, input, hx, zeta):
        outputs, hx = self(input, hx, zeta)
        policy = Categorical(torch.nn.functional.softmax(outputs, dim=-1))
        action = policy.sample()

        return policy, hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu.get_zeta(batch_size))

    def kl_divergence(self):
        return self.gru.kl_divergence() + self.mu.kl_divergence()

class RL3C(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior):
        super(RL3C, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden

        self.initial = nn.Parameter(0.00 * torch.randn(1, self.num_hidden), requires_grad=False)
        self.beta = nn.Parameter(-16 * torch.ones([]), requires_grad=True)

        self.gru = ProbabilisticGRUCell(num_states, num_hidden, prior)
        if prior == 'grouphs':
            self.mu = LinearGroupHS(num_hidden, num_actions)
        elif prior == 'gaussian':
            self.mu = LinearGaussian(num_hidden, num_actions)
            self.prior_mu = nn.Parameter(torch.zeros([]), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.zeros([]), requires_grad=True)
        elif prior == 'svdo':
            self.mu = LinearSVDO(num_hidden, num_actions)
        elif prior == 'mog':
            self.prior_mu = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.Tensor([0, -6]), requires_grad=True)
            self.mixture_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.mu = LinearMoG(num_hidden, num_actions)

        self.prior = prior

    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])
        return self.mu(hx, zeta[1]), hx

    def act(self, input, hx, zeta):
        outputs, hx = self(input, hx, zeta)
        policy = Categorical(torch.nn.functional.softmax(outputs, dim=-1))
        action = policy.sample()

        return policy, hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu.get_zeta(batch_size))

    def kl_divergence(self):
        if self.prior == 'gaussian':
            prior_distribution = Normal(self.prior_mu, self.prior_logscale.exp())
            return self.gru.kl_divergence(prior_distribution) + self.mu.kl_divergence(prior_distribution)
        elif self.prior == 'mog':
            mix = Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = Normal(self.prior_mu, self.prior_logscale.exp())
            prior_distribution = MixtureSameFamily(mix, comp)
            return self.gru.kl_divergence(prior_distribution) + self.mu.kl_divergence(prior_distribution)
        else:
            return self.gru.kl_divergence() + self.mu.kl_divergence()

class RL3Q(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior):
        super(RL3Q, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden

        self.initial = nn.Parameter(0.00 * torch.randn(1, self.num_hidden), requires_grad=False)
        self.beta = nn.Parameter(-16 * torch.ones([]), requires_grad=True)

        self.gru = ProbabilisticGRUCell(num_states, num_hidden, prior)
        if prior == 'grouphs':
            self.mu = LinearGroupHS(num_hidden, num_actions)
        elif prior == 'gaussian':
            self.mu = LinearGaussian(num_hidden, num_actions)
            self.prior_mu = nn.Parameter(torch.zeros([]), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.zeros([]), requires_grad=True)
        elif prior == 'svdo':
            self.mu = LinearSVDO(num_hidden, num_actions)
        elif prior == 'mog':
            self.prior_mu = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.Tensor([0, -6]), requires_grad=True)
            self.mixture_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.mu = LinearMoG(num_hidden, num_actions)

        self.prior = prior

    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])
        return self.mu(hx, zeta[1]), hx

    def act(self, input, hx, zeta):
        outputs, hx = self(input, hx, zeta)
        action = torch.argmax(outputs, -1)

        return outputs, hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu.get_zeta(batch_size))

    def kl_divergence(self):
        if self.prior == 'gaussian':
            prior_distribution = Normal(self.prior_mu, self.prior_logscale.exp())
            return self.gru.kl_divergence(prior_distribution) + self.mu.kl_divergence(prior_distribution)
        elif self.prior == 'mog':
            mix = Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = Normal(self.prior_mu, self.prior_logscale.exp())
            prior_distribution = MixtureSameFamily(mix, comp)
            return self.gru.kl_divergence(prior_distribution) + self.mu.kl_divergence(prior_distribution)
        else:
            return self.gru.kl_divergence() + self.mu.kl_divergence()

class RL3A2C(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior, bias=False):
        super(RL3A2C, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden

        self.initial = nn.Parameter(0.00 * torch.randn(1, self.num_hidden), requires_grad=False)
        self.beta = nn.Parameter(-16 * torch.ones([]), requires_grad=True)

        self.gru = ProbabilisticGRUCell(num_states, num_hidden, prior, bias)
        if prior == 'grouphs':
            self.mu_actor = LinearGroupHS(num_hidden, num_actions, bias)
            self.mu_critic = LinearGroupHS(num_hidden, 1, bias)
        elif prior == 'gaussian':
            self.mu_actor = LinearGaussian(num_hidden, num_actions, bias)
            self.mu_critic = LinearGaussian(num_hidden, 1, bias)
            self.prior_mu = nn.Parameter(torch.zeros([]), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.zeros([]), requires_grad=True)
        elif prior == 'svdo':
            self.mu_actor = LinearSVDO(num_hidden, num_actions, bias)
            self.mu_critic = LinearSVDO(num_hidden, 1, bias)
        elif prior == 'mog':
            self.prior_mu = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.prior_logscale = nn.Parameter(torch.Tensor([0, -6]), requires_grad=True)
            self.mixture_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.mu_actor = LinearMoG(num_hidden, num_actions, bias)
            self.mu_critic = LinearMoG(num_hidden, 1, bias)

        self.prior = prior

    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])
        return self.mu_actor(hx, zeta[1]), self.mu_critic(hx, zeta[2]), hx

    def act(self, input, hx, zeta):
        logits, values, hx = self(input, hx, zeta)
        policy = Categorical(torch.nn.functional.softmax(logits, dim=-1))
        action = policy.sample()

        return policy, values, hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu_actor.get_zeta(batch_size), self.mu_critic.get_zeta(batch_size))

    def kl_divergence(self):
        if self.prior == 'gaussian':
            prior_distribution = Normal(self.prior_mu, self.prior_logscale.exp())
            return self.gru.kl_divergence(prior_distribution) + self.mu_actor.kl_divergence(prior_distribution) + self.mu_critic.kl_divergence(prior_distribution)
        elif self.prior == 'mog':
            mix = Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = Normal(self.prior_mu, self.prior_logscale.exp())
            prior_distribution = MixtureSameFamily(mix, comp)
            return self.gru.kl_divergence(prior_distribution) + self.mu_actor.kl_divergence(prior_distribution) + self.mu_critic.kl_divergence(prior_distribution)
        else:
            return self.gru.kl_divergence() + self.mu_actor.kl_divergence() + self.mu_critic.kl_divergence()
