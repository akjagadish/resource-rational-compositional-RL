import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_rl.agent.linear import LinearGroupHS, LinearSVDO, LinearGaussian, LinearMoG

class ProbabilisticGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, prior, bias=False):
        super(ProbabilisticGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if prior == 'grouphs':
            self.weight_ih = LinearGroupHS(input_size, 3 * hidden_size, bias)
            self.weight_hh = LinearGroupHS(hidden_size, 3 * hidden_size, bias)
        elif prior == 'gaussian':
            self.weight_ih = LinearGaussian(input_size, 3 * hidden_size, bias)
            self.weight_hh = LinearGaussian(hidden_size, 3 * hidden_size, bias)
        elif prior == 'svdo':
            self.weight_ih = LinearSVDO(input_size, 3 * hidden_size, bias)
            self.weight_hh = LinearSVDO(hidden_size, 3 * hidden_size, bias)
        elif prior == 'mog':
            self.weight_ih = LinearMoG(input_size, 3 * hidden_size, bias)
            self.weight_hh = LinearMoG(hidden_size, 3 * hidden_size, bias)

        self.prior = prior

    def forward(self, input, hidden, zeta):
        gi = self.weight_ih(input, zeta[0])
        gh = self.weight_hh(hidden, zeta[1])

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy

    def get_zeta(self, batch_size):
        # first for W_ih, second for W_hh
        return (self.weight_ih.get_zeta(batch_size), self.weight_hh.get_zeta(batch_size))

    def kl_divergence(self, p=None):
        if self.prior == 'gaussian' or self.prior == 'mog':
            return self.weight_ih.kl_divergence(p) + self.weight_hh.kl_divergence(p)
        else:
            return self.weight_ih.kl_divergence() + self.weight_hh.kl_divergence()
