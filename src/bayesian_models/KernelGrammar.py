import torch
import numpy as np
from matplotlib import pyplot as plt
import gpytorch
import random
from Kernels import Kernels, DiagonalKernel, ChangePointKernel
import math
from GP import GP #from models.grammar.GP
from train import MLE #from models.grammar.train 
import warnings


def additive_composition(k1, k2):
    return k1 + k2

def multiplicative_composition(k1, k2):
    return k1*k2

def change_point_composition(k1, k2, threshold = 0.5):
    change_point_kernel = ChangePointKernel(k1, k2, threshold)
    return change_point_kernel


class KernelGrammar():
    def __init__(self, basis_kernels, complexity_penalty = 0.7, ignore_warnings=True, depth=2, change_point_condition = False):
        self.basis_kernels = basis_kernels
        self.kernel_types = [type(kern) for kern in self.basis_kernels]
        self.complexity_penalty = complexity_penalty

        if change_point_condition:
            self.composition_rules = [additive_composition, multiplicative_composition, change_point_composition]
        else:
            self.composition_rules = [additive_composition, multiplicative_composition]
        self.num_rules = len(self.composition_rules)
        self.depth = depth
        if ignore_warnings:
            # suppress GPyTorch warnings about adding jitter
            warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)



    def grow_kernel_space(self, depth=2):
        # depth is how far beyond the first level in the grammar tree we go.
        # The compositional kernel space grows exponentially fast with depth,
        # so it's recommended to stick with depth=2
        kernel_space = [k for k in self.basis_kernels]
        prior = [1, 1, 1]
        depth_probs = [self.complexity_penalty**(d+1) for d in range(depth)]
        last_level = [k for k in self.basis_kernels] # the kernels which we will grow

        for d in range(depth):
            new_kernels = []
            for k in last_level:
                for k_ in self.basis_kernels:
                    for composition in self.composition_rules:
                        kernel = composition(k, k_)
                        new_kernels.append(kernel)
                        kernel_space.append(kernel)
                        prior.append(depth_probs[d])

            last_level = new_kernels

        prior = np.array(prior)
        prior /= prior.sum()


        for i, kernel in enumerate(kernel_space):
            scale_kernel = gpytorch.kernels.ScaleKernel(kernel)
            kernel_space[i] = scale_kernel


        return kernel_space, prior

    def _print_trainable_parameters(self):
        pass

    def sample_kernel_particles(self, num_samples):
        ## make depth adaptive to complexity_penalty

        self.kernel_space = [kernel for kernel in self.basis_kernels]
        self.prior = np.ones(num_samples)  # the first 3 kernels are
        depth_penalty = self.complexity_penalty ** np.arange(1, self.max_depth)
        depth_probability = depth_penalty / depth_penalty.sum()
        for sample in range(num_samples - len(self.basis_kernels)):
            d = np.random.choice(np.arange(1, self.max_depth), p=depth_probability)
            kernels = np.random.choice(self.basis_kernels, size=d)  # sample N basis kernels
            compositions = np.random.choice(self.composition_rules, size=d-1)  # sample N - 1 composition rules
            kernel = kernels[0]
            for i, kernel_prime in enumerate(kernels[1:]):
                composition = compositions[i]
                kernel =composition(kernel, kernel_prime)
            self.kernel_space.append(gpytorch.kernels.ScaleKernel(kernel))  # put in scale kernel format when composition is done
            self.prior[sample] = self.complexity_penalty**d  # complexity_penalty to the power of depth

        self.prior /= self.prior.sum()  # normalize prior
        return self.kernel_space, self.prior



    def flip(self):
        proposal = random.uniform(0, 1)
        accept = True if proposal > self.complexity_pen else False
        return accept



    def get_base_kernel_types(self, kernel, base_list):
        # recursive method for getting the basis kernels of any given kernel, be it compositional or not
        # appends base kernels to the list passed as base_list
        if hasattr(kernel, "base_kernel"):
            base_list.append(type(kernel.base_kernel))
        else:
            for sub_kern in kernel.kernels:
                self.get_base_kernel_types(sub_kern, base_list)


    def initialize(self):
        self.kernels, self.probabilities = self.grow_kernel_space(depth=self.depth)


    def fit(self, X, y, training_iters):


        num_kernels = len(self.kernels)
        total_iters = training_iters * num_kernels
        gps = []
        mls = np.zeros(len(self.kernels))

        for i, kernel in enumerate(self.kernels):
            l = gpytorch.likelihoods.GaussianLikelihood()
            gp = GP(X, y, l, kernel)

            gp, ml = MLE(gp, X, y, learning_rate=0.01, training_iterations=training_iters)
            gps.append(gp)
            mls[i] = ml
            #print("Percentage completed: ", np.round((i/num_kernels) *100, 2), "%", end="\r")

        self.mls = mls * self.probabilities
        self.posterior = self.mls / self.mls.sum()
        self.map_gp = gps[np.argmax(mls)]
        self.map_kernel = self.kernels[np.argmax(mls)]
