import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/notebooks/models/bayesian_models/')
from ChoiceModel import MeanTracker, ChoiceModel
from KernelGrammar import KernelGrammar
from EpisodicDictionary import EpisodicDictionary
from Kernels import Kernels
#sys.path.append('/notebooks/models/meta_rl/')
from tasks.sample_data import sample_rewards
from SimulationFit import SimulationFit
import pandas as pd

path = '/notebooks/modelfits/simulated_data_preds/mean_tracker'

# run eval
rule = 'changepoint'
n_runs = 100 # if rule=='changepoint' else 100
condition = 'compositional'
sub_task= 'composed'
n_trials = 5
n_funs = 4
n_rules = 1 ## pick up fom rule
#n_sets = 2 if rule=='changepoint' else 1
n_subtasks = 3 if condition == 'compositional' else 1
seeds = np.random.randint(low=0, high=1000, size=(1,))
return_all=True

# setup the model
num_iters = 50

#load fitted softmax parameter
# MODEL_PATH = f'/notebooks/modelfits/baselines_to_participants/{rule}/{analysis_name}_mean_model_params.npy'
analysis_name = f'mean_tracker_{condition}_all_{sub_task}_all' #'{}_{}_{}_{}_{}'.format('mean_tracker', condition, 'all', 'all', 'all')
nll_model_name = f'/notebooks/modelfits/baselines_to_participants/{rule}/{analysis_name}_mean_model_params.npy'
nll_per_model = np.load(nll_model_name) #pd.read_pickle(nll_model_name)
num_participants = len(nll_per_model) #int(nll_per_model.index.max())

# setup simulated data fit
simulation_fitter = SimulationFit(condition=condition, num_trials=n_trials, rule=rule, policy='sticky_ucb', return_all=return_all)

linear = ['linneg', 'linpos']
periodic = ['perodd', 'pereven']
list_sets = [[linear, periodic]] #[[linear, periodic], [periodic, linear]] if rule == 'changepoint' else [[linear, periodic]]
n_sets = len(list_sets) #if rule=='changepoint' else 1
regrets = np.ones((num_participants, n_sets, n_funs, n_runs, n_subtasks, n_trials))
rewards = np.ones((num_participants, n_sets, n_funs, n_runs, n_subtasks, n_trials))
actions = np.zeros((num_participants, n_sets, n_funs, n_runs, n_subtasks, n_trials))
best_actions = np.zeros((num_participants, n_sets, n_funs, n_runs, n_subtasks, n_trials))


for set_idx, sets in enumerate(list_sets):
    set1, set2 = sets
    counter = 0
    for lin in set1:
        for per in set2:
            # sample data
            block = [lin, per, 'linperiodic']
            description = block if condition == 'compositional' else ['linperiodic']
            _, Y, _ = sample_rewards(n_runs, block, rule, seed_val=seeds[0], noise_per_arm=True, noise_var=0.1)
            Y = Y if condition == 'compositional' else Y[:, [2]] # only pass the composite structure if non-compositional
            for subj_idx in np.arange(num_participants): 
                regrets_per_subj = []
                rewards_per_subj = []
                actions_per_subj = []
                best_actions_per_subj = []
                for run in range(n_runs):
                    beta = nll_per_model[subj_idx, 0] 
                    tau = nll_per_model[subj_idx, 1] 
                    sticky = nll_per_model[subj_idx, 2]  
                    model = MeanTracker(num_iters)
                    regret_per_run, action, best_action, reward_per_run = simulation_fitter.fit(model, Y[run], description, params=[beta, tau, sticky])
                    regrets_per_subj.append(regret_per_run)
                    rewards_per_subj.append(reward_per_run)
                    actions_per_subj.append(action)
                    best_actions_per_subj.append(best_action)

                ## store regretsr
                regrets[subj_idx, set_idx, counter] = np.stack(regrets_per_subj)
                if not return_all:
                    actions[subj_idx, set_idx, counter, :, n_subtasks-1] = (torch.stack(actions_per_subj)*5).numpy()
                else:
                    actions[subj_idx, set_idx, counter] = np.stack(actions_per_subj)
                    best_actions[subj_idx, set_idx, counter] = np.stack(best_actions_per_subj)
                    rewards[subj_idx, set_idx, counter] = np.stack(rewards_per_subj)

            counter += 1
            # np.save('{}/interim_{}_simple_grammar_simulated_{}'.format(path, counter, rule), regrets)

# np.save('{}/regrets_mean_tracker_simulated_{}_{}_{}'.format(path, condition, rule, sub_task), regrets)
# np.save('{}/actions_mean_tracker_simulated_{}_{}_{}'.format(path, condition, rule, sub_task), actions)
np.save('{}/stats_mean_tracker_simulated_{}_{}_{}'.format(path, condition, rule, sub_task), [actions, rewards, regrets, best_actions])