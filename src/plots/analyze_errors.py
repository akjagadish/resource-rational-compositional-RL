import numpy as np
from load_human_behavior import load_behavior, load_percondition_metrics
import pymc3 as pm
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import theano as t
import pandas as pd
import torch
import pickle

NUM_TASKS=20

def return_base_conditions(data_comp, criterion, n_subjs, use_most_rewarding=True):
     
    if use_most_rewarding:
        # most rewariding arm w/ noise
        number_opt_linear = np.stack([data_comp.iloc[subj].optimal_actions[0::3].sum(1)/5 for subj in range(n_subjs)])
        number_opt_periodic = np.stack([data_comp.iloc[subj].optimal_actions[1::3].sum(1)/5 for subj in range(n_subjs)])
    else:
        # most rewarding arm w/o noise
        #number_opt_linear = np.stack([(((np.array(data_comp.iloc[subj].best_actions)[0::3])%5).repeat(5).reshape(20,5)==(data_comp.iloc[subj].actions[0::3])%5).sum(1)/5 for subj in range(n_subjs)])
        number_opt_periodic = np.stack([(((np.array(data_comp.iloc[subj].best_actions)[1::3]+1)%2).repeat(5).reshape(20,5)==(data_comp.iloc[subj].actions[1::3]+1)%2).sum(1)/5 for subj in range(n_subjs)])
        
    lin_opt = (number_opt_linear>criterion)
    per_opt = (number_opt_periodic>criterion)
    linper_opt = (lin_opt)*(per_opt)
    lin_opt_but_not_per = (lin_opt).astype('int')-(linper_opt).astype('int')
    per_opt_but_not_lin = (per_opt).astype('int')-(linper_opt).astype('int')
    linper_nonopt = (number_opt_linear<=criterion)*(number_opt_periodic<=criterion)

    return linper_nonopt, lin_opt_but_not_per, per_opt_but_not_lin, linper_opt

def return_composition_condition(data_comp, criterion, n_subjs):
     number_opt_composed = np.stack([data_comp.iloc[subj].optimal_actions[2::3].sum(1)/5 for subj in range(n_subjs)])
     composed = (number_opt_composed>criterion)
     non_composed = number_opt_composed<=criterion
     return composed, non_composed

def return_subtasks_per_condition(condition, non_composed, non_linper, lin, per, linper):

    condition_lin_only =  condition*lin*non_composed
    condition_per_only =  condition*per*non_composed
    condition_linper = condition*linper*non_composed
    condition_nonlinper = condition*non_linper*non_composed
    condition_non_composed = condition*non_composed
    condition_break_down = np.array([[condition_nonlinper.sum(), condition_lin_only.sum(), condition_per_only.sum(), condition_linper.sum()]]).T
    return  condition_break_down, condition_non_composed

def last_arm_base(data_comp, task_id, n_subjs):
    trial=4
    return np.stack([data_comp.iloc[subj].actions[2::3]==np.repeat([data_comp.iloc[subj].actions[task_id::3][:,trial]], 5, axis=1).reshape(20, 5) for subj in range(n_subjs)])
   
def most_rewarding_arm(data_comp, task_id, n_subjs):
    return np.stack([np.repeat(data_comp.iloc[subj].actions[data_comp.iloc[subj].rewards.max(axis=1).reshape(20,3)[:, task_id].max(1).repeat(15).reshape(60, 5) == data_comp.iloc[subj].rewards],5).reshape(20, 5) == data_comp.iloc[subj].actions[2::3] for subj in range(n_subjs)])

def return_errors(rule):

    n_subjs = 92 if rule=='add' else 109
    rule =  rule#'add'
    data = load_behavior(rule)
    NUM_TASKS = 20
    NUM_TRIALS = 5
    data_comp = data[(data['experiment']=='compositional')]
    use_most_rewarding=False
    non_linper, lin, per, linper = return_base_conditions(data_comp, 0.4, n_subjs, use_most_rewarding)
    composed, non_composed = return_composition_condition(data_comp, 0.6, n_subjs)

    # correctly composed
    break_down_composers = np.array([[(non_linper*composed).sum(), (lin*composed).sum(), (per*composed).sum(), (linper*composed).sum()]]).T/composed.sum()

    # corner arms
    MIN_TIMES = 2 # greater than 2 out of 5
    corner_arms_people = np.stack([(data_comp.iloc[subj].actions[2::3]==0).astype('int')+(data_comp.iloc[subj].actions[2::3]==5).astype('int') for subj in range(n_subjs)])
    corner_arms_picked = (corner_arms_people.sum(2)>MIN_TIMES)
    condition_break_down, _ = return_subtasks_per_condition(corner_arms_picked, non_composed, non_linper, lin, per, linper)
    break_down_corner_arms = condition_break_down/non_composed.sum()

    # stickiness to last arm of periodic
    #TRIAL = 4
    #last_arm_periodic_people = np.stack([data_comp.iloc[subj].actions[2::3]==np.repeat([data_comp.iloc[subj].actions[1::3][:,TRIAL]], NUM_TRIALS, axis=1).reshape(NUM_TASKS, NUM_TRIALS) for subj in range(n_subjs)])
    last_arm_periodic_people = last_arm_base(data_comp, 1, n_subjs)
    MIN_TIMES = 2 # out of 5
    last_arm_periodic_picked = last_arm_periodic_people.sum(2)>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(last_arm_periodic_picked, non_composed, non_linper, lin, per, linper)
    break_down_last_arm_periodic = condition_break_down/non_composed.sum()

    # stickiness to last arm of linear
    #TRIAL = 4
    #last_arm_linear_people = np.stack([data_comp.iloc[subj].actions[2::3]==np.repeat([data_comp.iloc[subj].actions[0::3][:,TRIAL]], NUM_TRIALS, axis=1).reshape(NUM_TASKS, NUM_TRIALS) for subj in range(n_subjs)])
    last_arm_linear_people = last_arm_base(data_comp, 0, n_subjs)
    MIN_TIMES = 2 # out of 5
    last_arm_linear_picked = last_arm_linear_people.sum(2)>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(last_arm_linear_picked, non_composed, non_linper, lin, per, linper)
    break_down_last_arm_linear = condition_break_down/non_composed.sum()

    # pick their most rewarding linear arm 
    most_rewarding_linear_people = most_rewarding_arm(data_comp, [0], n_subjs)
    MIN_TIMES = 2 # out of 5
    most_rewarding_linear = most_rewarding_linear_people.sum(2)>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(most_rewarding_linear, non_composed, non_linper, lin, per, linper)
    break_down_most_rewarding_linear = condition_break_down/non_composed.sum()

    # pick their most rewarding periodic arm 
    most_rewarding_periodic_people = most_rewarding_arm(data_comp, [1], n_subjs)
    MIN_TIMES = 2 # out of 5
    most_rewarding_periodic = most_rewarding_periodic_people.sum(2)>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(most_rewarding_periodic, non_composed, non_linper, lin, per, linper)
    break_down_most_rewarding_periodic = condition_break_down/non_composed.sum()

    # pick their most rewarding arm in base subtasks
    most_rewarding_base_people =  most_rewarding_arm(data_comp, [0, 1], n_subjs)
    MIN_TIMES = 2 # out of 5
    most_rewarding_base_picked = most_rewarding_base_people.sum(2)>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(most_rewarding_base_picked, non_composed, non_linper, lin, per, linper)
    break_down_most_rewarding_base = condition_break_down/non_composed.sum()

    # unique arms
    unique_actions_people = np.stack([np.array([len(np.unique(row)) for row in data_comp.iloc[subj].actions[2::3]]) for subj in range(n_subjs)])
    MIN_TIMES = 2 # out of 5
    unique_actions_picked = unique_actions_people>MIN_TIMES
    condition_break_down, _ = return_subtasks_per_condition(unique_actions_picked, non_composed, non_linper, lin, per, linper)
    break_down_unique_actions = condition_break_down/non_composed.sum()

    # same arms
    same_actions_picked = unique_actions_people==1
    condition_break_down, _ = return_subtasks_per_condition(same_actions_picked, non_composed, non_linper, lin, per, linper)
    break_down_same_actions = condition_break_down/non_composed.sum()

    # misc 
    misc = (corner_arms_picked + same_actions_picked + unique_actions_picked + most_rewarding_linear + most_rewarding_periodic)==0
    condition_break_down, _ = return_subtasks_per_condition(misc, non_composed, non_linper, lin, per, linper)
    break_down_misc = condition_break_down/non_composed.sum()

    #all_break_downs = np.concatenate((break_down_corner_arms, break_down_last_arm_linear, break_down_last_arm_periodic, break_down_most_rewarding_linear,  break_down_most_rewarding_periodic,  break_down_most_rewarding_base, break_down_unique_actions, break_down_same_actions), axis=1) #break_down_number_wrongly_composed
    #all_condition_labels = ['Corner arms', 'Stickinesss (Linear)', 'Stickinesss (Periodic)', 'Most rewarding (Linear)',  'Most rewarding (Periodic)', 'Most rewarding (Base)', 'Over exploration', 'No exploration']#, 'Compose wrong functions']
    condition_labels = ['Corner arms', 'Most rewarding: Linear',  'Most rewarding: Periodic', 'Over exploration', 'No exploration', 'Composed'] #, 'Compose wrong functions']
    break_downs = np.concatenate((break_down_corner_arms, break_down_most_rewarding_linear,  break_down_most_rewarding_periodic, break_down_unique_actions, break_down_same_actions, break_down_composers), axis=1) #break_down_number_wrongly_composed
    
    optimal_action_labels = ['Linear \u2718 Perodic \u2718', 'Linear \u2714 Perodic \u2718', 'Linear \u2718 Perodic \u2714', 'Linear \u2714 Perodic \u2714', ]

    return break_downs, condition_labels, optimal_action_labels

def z_transform(x):
    return (x-x.mean())/x.std()

def most_rewarding_arm(data_comp, task_id, n_subjs):
    return np.stack([np.repeat(data_comp.iloc[subj].actions[data_comp.iloc[subj].rewards.max(axis=1).reshape(20,3)[:, task_id].max(1).repeat(15).reshape(60, 5) == data_comp.iloc[subj].rewards],5).reshape(20, 5) == data_comp.iloc[subj].actions[2::3] for subj in range(n_subjs)])

def return_regrets_and_errors(data, EXPERIMENT, error='corner'):
    
    # regrets
    data_comp = data[(data['experiment']==EXPERIMENT)]
    n_subjs = {}
    n_subjs['compositional'] = len(data_comp)
    #n_subjs['noncompositional'] = len(data_noncomp)
    subj_ids = data_comp.index
    subj_ids = np.repeat(subj_ids, NUM_TASKS)
    lin_regrets = np.stack([data_comp.iloc[idx].regrets[0::3].sum(1) for idx in range(n_subjs['compositional'])]).reshape(-1)
    per_regrets = np.stack([data_comp.iloc[idx].regrets[1::3].sum(1) for idx in range(n_subjs['compositional'])]).reshape(-1)
    
    # corner arms
    corner_arms_people = np.stack([(data_comp.iloc[subj].actions[2::3]==0).astype('int')+(data_comp.iloc[subj].actions[2::3]==5).astype('int') for subj in range(n_subjs['compositional'])])
    MIN_TIMES = 2 # out of 5
    corner_arms_picked = (corner_arms_people.sum(2)>MIN_TIMES).reshape(-1).astype('int')
    
    # unique arms
    unique_actions_people = np.stack([np.array([len(np.unique(row)) for row in data_comp.iloc[subj].actions[2::3]]) for subj in range(n_subjs['compositional'])])
    MIN_TIMES = 2 # out of 5
    unique_actions_picked = (unique_actions_people>MIN_TIMES).reshape(-1).astype('int')
    
    # same arms
    same_actions_picked = (unique_actions_people==1).reshape(-1).astype('int')
    
    # most rewarding periodi
    most_rewarding_periodic_people = most_rewarding_arm(data_comp, [1], n_subjs['compositional'])
    most_rewarding_periodic = (most_rewarding_periodic_people.sum(2)>MIN_TIMES).reshape(-1).astype('int')
    
    # most rewarding periodi
    most_rewarding_linear_people = most_rewarding_arm(data_comp, [0], n_subjs['compositional'])
    most_rewarding_linear = (most_rewarding_linear_people.sum(2)>MIN_TIMES).reshape(-1).astype('int')
    
    # composed or non-composed
    criterion = 0.6
    number_opt_composed = np.stack([data_comp.iloc[subj].optimal_actions[2::3].sum(1)/5 for subj in range(n_subjs['compositional'])])
    composed_first_trial = np.stack([data_comp.iloc[subj].optimal_actions[2::3, 0] for subj in range(n_subjs['compositional'])])
    composed = (number_opt_composed>criterion).reshape(-1)
    non_composed = (number_opt_composed<=criterion).reshape(-1)
    non_composed_first_trial = (composed_first_trial==False).reshape(-1)
    
    condition = non_composed
    if error=='corner':
        behaviour = corner_arms_picked
    elif error =='no_exploration':
        behaviour = same_actions_picked
    elif error == 'over_exploration':
        behaviour = unique_actions_picked
    elif error == 'most_rewarding_periodic':
        behaviour=most_rewarding_periodic
    elif error == 'most_rewarding_linear':
        behaviour=most_rewarding_linear
    elif error=='optimal':
        behaviour = composed
    else:
        raise NotImplementedError
    
    regrets = {'regrets_lin': z_transform(lin_regrets)[condition], 'regrets_per': z_transform(per_regrets)[condition]}#, 'regrets_linper': z_transform(lin_regrets*per_regrets)}
    probabilities = {error: behaviour[condition]}
    return pd.DataFrame(regrets), pd.DataFrame(probabilities) 

def run_logistic_regression(model_matrix, sample_kwargs=None, inference='smc'):

    # load the data
    x_lin = model_matrix['x_lin']
    x_per = model_matrix['x_per']
    y = model_matrix['y']
    n_subj = 1

    n = x_lin.shape
    if sample_kwargs is None:
        if inference == 'smc':
            sample_kwargs = dict(draws=1000, random_seed=0, cores=64) 
        else:
            sample_kwargs = dict(draws=5000, return_inferencedata=True, tune=1000, init='advi+adapt_diag') 

    with pm.Model() as model:

        # set priors means
        mu_1 = 0. # -0.20
        mu_2 = 0. #-0.15
        # set prior stds
        sigma_1 = 1. 
        sigma_2 = 1.

        w_1 = pm.Normal('w_lin',  mu=mu_1, sd=sigma_1, shape=n_subj)
        w_2 = pm.Normal('w_per', mu=mu_2, sd=sigma_2, shape=n_subj)

        rho = w_1 * x_lin + w_2 * x_per

        # Data likelihood
        
        ## softmax model
        # p_hat = softmax(rho)
        # yl = pm.Categorical('yl', p=p_hat, observed=y)
        
        ## Bernoulli random vector with probability of success given by sigmoid function and actual data as observed
        p_hat = pm.invlogit(rho)
        pm.Bernoulli(name='logit', p=p_hat, observed=y)

        # inference!
        if inference=='smc':
            trace = pm.sample_smc(**sample_kwargs) 
        else:
            trace = pm.sample(**sample_kwargs)
    return model, trace

def return_probits_people(data, EXPERIMENT, trial=0):
    
    # regrets
    data_comp = data[(data['experiment']==EXPERIMENT)]
    n_subjs = {}
    n_subjs[EXPERIMENT] = len(data_comp)
    subj_ids = data_comp.index
    subj_ids = np.repeat(subj_ids, NUM_TASKS)
    lin_regrets = np.stack([data_comp.iloc[idx].regrets[0::3].sum(1) for idx in range(n_subjs['compositional'])]).reshape(-1)
    per_regrets = np.stack([data_comp.iloc[idx].regrets[1::3].sum(1) for idx in range(n_subjs['compositional'])]).reshape(-1)
    optimal = np.stack([data_comp.iloc[idx].optimal_actions[2::3] for idx in range(n_subjs['compositional'])]).reshape(-1)
    optimal_first = np.stack([data_comp.iloc[idx].optimal_actions[2::3][:, trial].astype('int') for idx in range(n_subjs['compositional'])]).reshape(-1)
    phase = np.stack([np.array(data_comp.iloc[idx].best_actions)[1::3] for idx in range(n_subjs['compositional'])]).reshape(-1)%2
    even_phase = (phase+1)%2
    odd_phase = ((phase+1)%2==0).astype('int')
                     
    # corner arms
    corner_arms_people = np.stack([(data_comp.iloc[subj].actions[2::3]==0).astype('int')+(data_comp.iloc[subj].actions[2::3]==5).astype('int') for subj in range(n_subjs['compositional'])])
    corner_arms_first = corner_arms_people[...,0].reshape(-1)
    
    #phasic arms
    even_phasic_arms_people = np.stack([(data_comp.iloc[subj].actions[2::3]==0).astype('int')+(data_comp.iloc[subj].actions[2::3]==2)+(data_comp.iloc[subj].actions[2::3]==4).astype('int') for subj in range(n_subjs['compositional'])])
    odd_phasic_arms_people = np.stack([(data_comp.iloc[subj].actions[2::3]==1).astype('int')+(data_comp.iloc[subj].actions[2::3]==3)+(data_comp.iloc[subj].actions[2::3]==5).astype('int') for subj in range(n_subjs['compositional'])])
    even_phasic_arms_first = even_phasic_arms_people[...,trial].reshape(-1)
    odd_phasic_arms_first = odd_phasic_arms_people[...,trial].reshape(-1)
    phase_actions = (np.stack([np.array(data_comp.iloc[idx].actions[2::3][:, trial].astype('int')) for idx in range(n_subjs['compositional'])]).reshape(-1))%2
    phase_actions = (phase_actions==phase).astype('int')

    # most rewarding periodic
    most_rewarding_periodic_people = most_rewarding_arm(data_comp, [1], n_subjs['compositional'])
    most_rewarding_periodic_first = most_rewarding_periodic_people[..., trial].reshape(-1).astype('int')
    
    # most rewarding linear
    most_rewarding_linear_people = most_rewarding_arm(data_comp, [0], n_subjs['compositional'])
    most_rewarding_linear_first = most_rewarding_linear_people[...,trial].reshape(-1).astype('int')
    
    return lin_regrets, per_regrets, corner_arms_first, optimal_first, even_phasic_arms_first, odd_phasic_arms_first, most_rewarding_periodic_first, most_rewarding_linear_first, phase_actions

def fit_logistic_regression_errors(rule, trial,list_errors=None, people=True, model_name=None, sample_kwargs=None, sampler='nuts'):

    data = load_behavior(rule)
    model_per_trial = {}
    trace_per_trial = {}
    if people:
        lin_regrets, per_regrets, corner, optimal, _, _, _, _, phase = return_probits_people(data, 'compositional', trial=trial)
    else:
        lin_regrets, per_regrets, corner, optimal, _, _, phase = return_probits_models(model_name, trial=trial, full=False)

    list_errors = [ 'optimal', 'corner', 'corner_optimal', 'non_corner_optimal',  'corner_non_optimal', 'non_corner_non_optimal', \
                   'non_optimal', 'phasic_optimal', 'phasic_non_optimal', 'neither'] if list_errors is None else list_errors
    
    for error in list_errors:

        if error=='corner':
            behaviour = corner
        elif error=='optimal':
            behaviour = optimal==1
        elif error=='non_optimal':
            behaviour = optimal==0
        elif error == 'corner_optimal':
            behaviour = corner*optimal
        elif error=='non_corner_optimal':
            behaviour = (optimal==1)*(corner==0)
        elif error=='corner_non_optimal':
            behaviour = (optimal==0)*(corner==1)
        elif error=='non_corner_non_optimal':
            behaviour = (optimal==0)*(corner==0)
        elif error=='phasic_optimal':
            behaviour = (optimal==1)*(phase==1)
        elif error=='phasic_non_optimal':
            behaviour = (optimal==0)*(phase==1)
        elif error=='neither':
            behaviour = (optimal==0)*(corner==0)*(phase==0)==0
            

        # load model behaviours
        model_matrix = dict(y=behaviour,
                            x_lin=lin_regrets,
                            x_per=per_regrets)
        
        if sampler == 'nuts':
            sample_kwargs =  dict(draws=5000, return_inferencedata=True, tune=1000, init='advi+adapt_diag', random_seed=0) if sample_kwargs is None else sample_kwargs
        elif sampler == 'smc':
            sampler = dict(draws=1000, random_seed=0, cores=64) if sample_kwargs is None else sample_kwargs
        
        # run logistic regression
        model_per_trial[error], trace_per_trial[error] = run_logistic_regression(model_matrix, sample_kwargs, sampler) 

    return model_per_trial, trace_per_trial, list_errors

def reshape_to_match_rl3(x):
    return torch.tensor(torch.tensor(x.reshape(x.shape[0], np.prod(x.shape[1:4]), np.prod(x.shape[-2:]))))

def return_probits_models(model, trial=0, full=False):
    
    # regrets
    rule = 'add'
    changepoint = False if rule =='add' else True
    if model == 'rl3':
        actions, rewards, contexts, regrets, true_best_action = torch.load(f'../../RL3NeurIPS/simulations/all_stats_changepoint={changepoint}__full={full}_entropyTruejagadish2022curriculum-v0.pth')  
    else:
        actions, rewards, regrets, true_best_action = np.load('../modelfits/simulated_data_preds/{}/stats_{}_simulated_compositional_{}_{}.npy'.format(model, model, rule, 'all' if full else 'composed'))  
        actions = reshape_to_match_rl3(actions) 
        rewards = reshape_to_match_rl3(rewards)
        regrets = reshape_to_match_rl3(regrets)
        true_best_action = reshape_to_match_rl3(true_best_action)

    ## choose x trials
    regrets = regrets[:, :20]
    rewards = rewards[:, :20]
    actions = actions[:, :20]
    true_best_action = true_best_action[:, :20]

    n_subjs = {}
    n_subjs['compositional'] = len(actions)
    
    lin_regrets = regrets[...,:5].sum(2).reshape(-1)
    per_regrets =  regrets[..., 5:10].sum(2).reshape(-1)
    optimal_first = true_best_action[..., 10:][..., trial].reshape(-1)
    phase = true_best_action[..., 5:10]%2
    even_phase = (phase+1)%2
    odd_phase = ((phase+1)%2==0)#.astype('int')
                     
    # corner arms
    corner_arms_people = (actions[..., 10:]==0).to(torch.int) + (actions[..., 10:]==5).to(torch.int)
    corner_arms_first = corner_arms_people[..., 0].reshape(-1)
    
    #phasic arms
    phase_actions = actions[..., 10:]%2
    phase_actions = (phase_actions==phase).to(torch.int)[..., trial].reshape(-1)#.astype('int')

    # most rewarding periodic
    most_rewarding_periodic_people = None #most_rewarding_arm(data_comp, [1], n_subjs['compositional'])
    most_rewarding_periodic_first = None #most_rewarding_periodic_people[..., trial].reshape(-1).astype('int')
    
    # most rewarding linear
    most_rewarding_linear_people = None #most_rewarding_arm(data_comp, [0], n_subjs['compositional'])
    most_rewarding_linear_first = None #most_rewarding_linear_people[...,trial].reshape(-1).astype('int')
    
    return lin_regrets, per_regrets, corner_arms_first, optimal_first, most_rewarding_periodic_first, most_rewarding_linear_first, phase_actions

def save_traces_logistic_regression(rule, trial=0, people=True, model_name=None, errors=[ 'optimal', 'corner_non_optimal', 'phasic_non_optimal', 'neither']):
    model_name = None if people else model_name
    model_first, trace_first, list_errors = fit_logistic_regression_errors(rule, trial=trial, list_errors=errors, people=people, model_name=model_name)
    if people:
        with open(f'../modelfits/analysis_regrets/traces_{rule}.pkl', 'wb') as fp:
            pickle.dump([trace_first, list_errors], fp)
    else:
        with open(f'../modelfits/analysis_regrets/traces_{rule}_{model_name}.pkl', 'wb') as fp:
            pickle.dump([trace_first, list_errors], fp)
        print('dictionary saved successfully to file')
