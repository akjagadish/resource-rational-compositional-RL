import torch
import gpytorch
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import copy



def MLE(model, X, y, learning_rate=0.01, training_iterations=150, return_kernel = False):
    ''' Maximum likelihood method for finding kernel hyperparameters
        Given a model, training data and optimization settings, the function
        returns the optimized model and its marginal likelihood
        '''
    model_states = []
    losses = torch.zeros(training_iterations)

    model.set_train()


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll_ml = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll_ml(output, y)
        loss.backward()
        optimizer.step()

        model_states.append({param_name: param.detach() for param_name, param in model.state_dict().items()})
        losses[i] = loss.item()



    idx = torch.argmin(losses)
    best_model_params = model_states[idx]
    model.load_state_dict(best_model_params)
    marginal_likelihood = -torch.min(losses)

    if return_kernel:
        return model, np.exp(marginal_likelihood), model.kernel

    return model, np.exp(marginal_likelihood)
