import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/notebooks/models/grammar/')
from ChoiceModel import GrammarModel, ChoiceModel, RBFModelReset
from KernelGrammar import KernelGrammar
from EpisodicDictionary import EpisodicDictionary
from Kernels import Kernels
sys.path.append('/notebooks/models/metaRL/')
from tasks.sample_data import sample_rewards
from SimulationFit import SimulationFit
import argparse

path = '/notebooks/modelfits/simulated_data_preds/optimal/gp_rbf/'
parser = argparse.ArgumentParser(description='Fit gp_rbf model')
parser.add_argument('--eps', type=int, default=9, help='eps strengths')
parser.add_argument('--rule', type=str, default='add', help='rule')
args = parser.parse_args()

# run eval
n_runs = 100
n_inits = 5
rule = args.rule
condition = 'compositional'
n_trials = 5
eps = args.eps/10
#eps = 0.1
n_funs = 4
n_rules = 1 ## pick up fom rule
n_subtasks = 3 if condition == 'compositional' else 1
seeds = np.random.randint(low=0, high=1000, size=(n_inits,))

# setup the model
depth = 1
grammar = KernelGrammar(basis_kernels = Kernels.get_basic_kernels(), complexity_penalty=0.7, ignore_warnings=True, depth=1)
episodic_dict = EpisodicDictionary(num_features=2)
value_function = ChoiceModel.UCB
num_iters = 100

# setup simulated data fit
simulation_fitter = SimulationFit(condition=condition, num_trials=n_trials, rule=rule)
regrets = np.ones((n_rules, n_funs, n_inits, n_runs, n_subtasks, n_trials))
actions = torch.zeros((n_rules, n_funs, n_inits, n_runs, n_subtasks, n_trials))
counter = 0
for lin in ['linneg', 'linpos']:
    for per in ['perodd', 'pereven']:
        for init_idx in range(n_inits):
            print('running init: {}'.format(init_idx+1))
            # sample data
            block = [lin, per, 'linperiodic']
            description = block if condition == 'compositional' else ['linperiodic']
            _, Y, _ = sample_rewards(n_runs, block, rule, seed_val=seeds[init_idx], noise_per_arm=True, noise_var=0.1)
            Y = Y if condition == 'compositional' else Y[:, [2]] # only pass the composite structure if non-compositional
            regrets_per_init = []
            actions_per_init = []
            for run in range(n_runs):
                if rule == 'add':
                    print('use GP-RBF for additive rule')
                    model = RBFModelReset(num_iters)
                else:
                    print('use GP-RBF for cp rule')
                    model = RBFModelReset(num_iters)
                # regret_per_run = fit_simplegrammar(model, Y[run], description, eps=eps, rule=rule)
                regret_per_run, action = simulation_fitter.fit(model, Y[run], description, params=[eps])
                regrets_per_init.append(regret_per_run)
                actions_per_init.append(action)
            ## store regrets
            regrets[0, counter, init_idx] = np.stack(regrets_per_init)
            actions[0, counter, init_idx, :, n_subtasks-1,:] = torch.stack(actions_per_init)*5
        counter += 1
        # np.save('{}/interim_{}_simple_grammar_simulated_{}'.format(path, counter, rule), regrets)

np.save('{}/regrets_gp_rbf_simulated_greedy_{}_{}_{}'.format(path, condition, rule, eps), regrets)
np.save('{}/actions_gp_rbf_greedy_{}_{}_{}'.format(path, condition, rule, eps), actions)