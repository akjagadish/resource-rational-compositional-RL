import math
import torch
import gpytorch
import numpy as np
from gpytorch.means import Mean
from matplotlib import pyplot as plt


class CompositionalMean(Mean):
    ''' A custom mean function which creates a weighted combination of other GPs
        To construct its prior mean function
    '''
    def __init__(self, gps, probabilities, rule='add', linfirst=True):
        super(CompositionalMean, self).__init__()
        self.gps = gps
        self.probabilities = probabilities
        self.rule = rule
        self.linfirst = linfirst
    def forward(self, x):
        mu = torch.zeros(len(x))#, len(self.probabilities))
        if self.rule == 'add':
            for i, gp in enumerate(self.gps):
                mu += gp.posterior(x).mean * self.probabilities[i]
        elif self.rule =='changepoint':
            if self.linfirst:
                mu[:int(len(x)/2)] = self.gps[0].posterior(x).mean[:int(len(x)/2)] * self.probabilities[0]
                mu[int(len(x)/2):] = self.gps[1].posterior(x).mean[int(len(x)/2):] * self.probabilities[1]
            else:
                mu[:int(len(x)/2)] = self.gps[1].posterior(x).mean[:int(len(x)/2)] * self.probabilities[1]
                mu[int(len(x)/2):] = self.gps[0].posterior(x).mean[int(len(x)/2):] * self.probabilities[0]

        return mu




class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = kernel #copy.deepcopy(kernel)
        self.covar_module = self.kernel

        #set noise prior to a reasonable value to avoid underfitting
        self.likelihood.noise_covar.register_prior(
        "noise_std_prior",
        gpytorch.priors.SmoothedBoxPrior(0.001, 10.),
        lambda module: module.noise.sqrt()
        )



        # initialize noise to be low to avoid underfitting
        self.likelihood.noise = 0.001

    def register_mean(self, mean_module):
        self.mean_module = mean_module

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, test_x):
        self.eval()
        self.likelihood.eval()

        return self.likelihood(self(test_x))

    def set_train(self):
        self.train()
        self.likelihood.train()


    def evaluate_ml(self, X, y):

        # self.eval()
        # self.likelihood.eval()
        evaluator = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self(X)
        try:
            log_likelihood = evaluator(output, y).item()

        except RuntimeError:
            print("Could not converge")
            return 0.000001

        return np.exp(log_likelihood)


    def function_posterior(self, test_x):
        self.eval()
        self.likelihood.eval()

        return self(test_x)


    def sample_functions(self, num_functions):
        functions = []
        for i in range(num_functions):
            functions.append(self.function_posterior.sample())

        return functions
