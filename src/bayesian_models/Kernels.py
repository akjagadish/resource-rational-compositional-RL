import matplotlib
from importlib import reload
from matplotlib import pyplot as plt
import gpytorch
import torch
import copy
from gpytorch.constraints import Positive
import numpy as np
from copy import deepcopy
import math

class Kernels():
    '''Convenience getter for gpytorch kernels'''


    #lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 20.) # set a prior which restricts the range of the lengthscale parameter
    Linear = gpytorch.kernels.LinearKernel()
    Periodic = gpytorch.kernels.PeriodicKernel()
    RBF = gpytorch.kernels.RBFKernel()

    RBF.lengthscale = 0.1  # set initial low value to avoid underfitting
    Periodic.lengthscale = 0.1

    @staticmethod
    def get_basic_kernels():
        return [Kernels.Periodic, Kernels.RBF, Kernels.Linear]


class DiagonalKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, **params):

        dist = self.covar_dist(x1, x2, **params)#torch.cdist(x1, x2)
        K = torch.zeros_like(dist)
        eps = 0.01
        idx = torch.where(dist <= eps)
        K[idx] = 1.
        return K




class ChangePointKernel(gpytorch.kernels.Kernel):
    def __init__(self, k1, k2, t, **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.k2 = k2
        self.t = t

    def swap(self):
        clone = deepcopy(self.k1)#self.k1.clone()
        self.k1 = deepcopy(self.k2)
        self.k2 = clone

    def set_threshold(self, t):
        self.t = t

    def forward(self, x1, x2,  diag=False, **params):

        # compute kernel matrices for constituant kernels
        # one could do this without evaluating the whole kernel for all inputs
        # based on the threshold, but this way is much simpler to implement
        kernel1 = self.k1(x1, x2).evaluate()
        kernel2 = self.k2(x1, x2).evaluate()



        prod = torch.cartesian_prod(x1.squeeze(1), x2.squeeze(1)) # compute product tensor

        K = torch.zeros(len(x1), len(x2))  # create placeholder tensor

        k1_weights = torch.zeros(len(x1)* len(x2)) # create individual kernel component placeholders
        k2_weights = torch.zeros(len(x1)* len(x2))

        ## find the indices where the product tensor falls in either category based on threshold
        idx1 = torch.where(torch.logical_and((prod[:, 0] < self.t), (prod[:, 1] < self.t)))
        idx2 = torch.where(torch.logical_and((prod[:, 0] >= self.t), (prod[:, 1] >= self.t)))
        k1_weights[idx1] = 1
        k2_weights[idx2] = 1

        k1_weights = k1_weights.view(len(x1), len(x2))
        k2_weights = k2_weights.view(len(x1), len(x2))

        K += kernel1 * k1_weights
        K += kernel2 * k2_weights

        if diag:
            K = K[0]
        return K


#
#
# x1 = torch.linspace(-1, 1, 40)
# x2 = torch.linspace(-2, 2, 20)
#
# y = torch.sin(x1*3*math.pi)
#
# cp = ChangePointKernel(Kernels.Linear, Kernels.Periodic, t=-0.)
# from GP import GP
# l = gpytorch.likelihoods.GaussianLikelihood()
# gp = GP(x1, y, l, cp)
# p = gp.posterior(x2)
# m = p.mean.detach()
# sd = p.stddev.detach()
#
#
# plt.plot(x2, m)
# plt.fill_between(x2, m-sd, m+sd, alpha= 0.4)
#
# cp.swap()
# gp = GP(x1, y, l, cp)
# p = gp.posterior(x2)
# m = p.mean.detach()
# sd = p.stddev.detach()
#
#
# plt.plot(x2, m)
# plt.fill_between(x2, m-sd, m+sd, alpha= 0.4)
