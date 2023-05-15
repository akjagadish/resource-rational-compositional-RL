import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument('-m','--models', nargs='+', help='model name')
parser.add_argument('--changepoint', action='store_true', default=False, help='compute for changepoint')
parser.add_argument('--subtask',  default='all', help='fit subtask')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--trial',  default='all', help='fit subtask')

args =  parser.parse_args()
models = args.models if args.models is not None else ['mean_tracker', 'mean_tracker_compositional', 'rbf_nocontext_nomemory', 'simple_grammar_constrained'] 
rule = 'changepoint' if args.changepoint else 'add'  
fit_subtasks = args.subtask
experiment = 'compositional'
fit_tasks = 'all'
fit_trials = args.trial
seed = args.seed
n_subjs = 109 if rule=='changepoint' else 92
MODEL_PATH = f'/notebooks/scratch/modelfits/baselines_to_participants/{rule}/'

for model_name in models:
    mlls = []
    for subject in np.arange(n_subjs):
        analysis_name = f'{model_name}_{experiment}_{fit_tasks}_{fit_subtasks}_{fit_trials}_{seed}' 
        load_model_name = MODEL_PATH + '{}_model_fits_smc_{}.pkl'.format(analysis_name, subject)
        mlls.append(pd.read_pickle(load_model_name)['mll'])
    save_mlls_name = f'{model_name}_{experiment}_{fit_tasks}_{fit_subtasks}_{fit_trials}' 
    np.save(f'/notebooks/modelfits/baselines_to_participants/{save_mlls_name}_{rule}.npy', np.stack(mlls))