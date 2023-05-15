from logging import raiseExceptions
import numpy as np
import pandas as pd
import pymc3 as pm
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import theano as t
from utils import load_data
import torch
print('Runing on PyMC3 v{}'.format(pm.__version__))
print('Runing on Theano v{}'.format(t.__version__))

NUM_ARMS = 6 

def set_random_seed(seed):
    """
    Sets all random seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fit_hier_params(model_matrix, sample_kwargs=None):
    """ Fits hierarchical models over participants 
        given predicted mean and std dev of rewards

    Args:
        model_matrix ([type]): [description]
        sample_kwargs ([type], optional): [description]. Defaults to None.
    """

    # load the data
    x_mu = model_matrix['x_mu']
    x_sd = model_matrix['x_sd']
    x_sc = model_matrix['x_sc']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=5000, random_seed=0, cores=64) 

    with pm.Model() as hier_model:
        mu_1 = 0.
        mu_2 = 0. 
        mu_3 = 0. 

        sigma_1 = 5. 
        sigma_2 = 5.
        sigma_3 = 5. 

        b_1 = pm.Normal('beta_mean',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_uncertainity', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc', mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = b_1 * x_mu +  b_3 * x_sc + b_2 * x_sd
        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace = pm.sample_smc(**sample_kwargs) 

    return hier_model, trace

def get_loo(model, trace):
    loo = pm.stats.loo(trace, model)
    return dict(LOO=loo.loo, LOO_se=loo.loo_se, p_LOO=loo.p_loo, shape_warn=loo.warning)

def run_save_models(model_matrix, model_name, sample_kwargs=None, subj_idx=None):

    def sample_model(sampler, sampler_args, list_params, name=None, subj_idx=None):
        model, trace = sampler(*sampler_args)
        _loo = pd.DataFrame([get_loo(model, trace)], index=[subj_idx])
        _loo['subj_idx'] = subj_idx
        _loo['mll'] = trace.report.log_marginal_likelihood.mean()
        _params = pm.summary(trace).loc[list_params, :]
        _params['Model'] = name
        _params['subj_idx'] = subj_idx
        return _loo, _params, model, trace

    fit_params = ['beta_mean[0]', 'beta_uncertainity[0]', 'beta_sc[0]'] #
    fit_sampler_args = [model_matrix, sample_kwargs]
    model_loo, model_params, model, trace = sample_model(fit_hier_params, fit_sampler_args, fit_params, model_name, subj_idx)
    
    return model_params, model_loo, model, trace

def construct_sticky_choice(pred_data, n_arms=6):
    x_sc = []
    y = pd.get_dummies(pred_data.loc[:, 'prev_arm'])
    for c in set(range(0, n_arms)):
        if c not in set(y.columns):
            y[c] = np.zeros(len(y), dtype=int)
                
    y = y[np.sort(y.columns.values)].values

    return y

def fit_models(model_name, sample_kwargs, analysis_name, subject, rule, experiment, fit_tasks, fit_subtasks, fit_trials, seed):
    
    set_random_seed(seed)
    print("Fitting bayesian softmax policy for predictions from {} from the {} condition: participant {}".format(model_name, experiment, subject))
    pred_data = load_data(model_name, rule=rule, experiment=experiment, fit_tasks=fit_tasks, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
    pred_data = pred_data[pred_data['subj_idx']==subject]

    subj_idx = subject
    n_subj = 1 
    
    # prep the predictor vectors
    x_mu = np.array([pred_data.loc[:, 'mu_%d' % ii].values for ii in range(NUM_ARMS)]).T
    x_sd = np.array([pred_data.loc[:, 'sigma_%d' % ii].values for ii in range(NUM_ARMS)]).T
    x_sc = construct_sticky_choice(pred_data, NUM_ARMS)  
    y = pred_data['arm'].values 
    

    model_matrix = dict(
        x_mu=x_mu,
        x_sd=x_sd,
        x_sc=x_sc,
        y=y,
        n_subj=n_subj,
        subj_idx=subj_idx
    )
    sample_kwargs = dict(draws=5000, random_seed=seed, cores=64)
    print("Running {} subject for the {} model".format(model_matrix['n_subj'], model_name))
    model_params, model_loo, model, trace = run_save_models(model_matrix, model_name, sample_kwargs=sample_kwargs, subj_idx=subj_idx)

    analysis_name = f"{model_name}_{experiment}_{fit_tasks}_{fit_subtasks}_{fit_trials}_{seed}" if analysis_name is None else analysis_name
    model_params.to_pickle(MODEL_PATH + f'{analysis_name}_model_params_smc_{subject}.pkl')
    model_loo.to_pickle(MODEL_PATH + f'{analysis_name}_model_fits_smc_{subject}.pkl')
    #trace.to_netcdf(MODEL_PATH + f'{analysis_name}_model_trace_smc_{subject}.pkl')

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()  
    parser.add_argument('-m','--model', help='model to fit', required=True)
    parser.add_argument('--subject', type=int, default=1, help='subject to fit')
    parser.add_argument('--changepoint', action='store_true', default=False, help='compute for changepoint')
    parser.add_argument('--subtask',  default='composed', help='fit subtask')
    parser.add_argument('--trial',  default='all', help='fit subtask')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    args =  parser.parse_args()
    model_name = args.model
    subject = args.subject
    rule = 'changepoint' if args.changepoint else 'add'  
    fit_subtask = args.subtask
    fit_trial = args.trial
    seed = args.seed
    n_subjs = 92 if rule == 'add' else 109
    
    MODEL_PATH = f'/notebooks/scratch/modelfits/baselines_to_participants/{rule}/'
    fit_models(model_name, sample_kwargs=None, analysis_name=None, subject=subject, experiment='compositional', rule=rule, fit_tasks='all', fit_subtasks=fit_subtask, fit_trials=fit_trial, seed=seed) 
        