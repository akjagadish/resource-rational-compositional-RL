import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# declare task settings
NUM_TRIALS = 5
NUM_TASKS = 20 
NUM_SUBTASKS = 3
NUM_ARMS = 6
BASEPATH = '/notebooks/modelfits/pooled_preds'

def load_best_rl3_fitted_to_participants(model_name, rule, experiment, fit_tasks, fit_subtasks, fit_trials, entropy=True):
    best_models = np.load(f'/notebooks/modelfits/rl3_to_participants/best_RL3_models_per_participant_{rule}.npy')
    
    # pool across subjects
    for subject, dl in enumerate(best_models):
        data = load_rl3_fitted_to_participants_per_dl(model_name, subject=subject, dl=dl, rule=rule, experiment=experiment, fit_tasks=fit_tasks, fit_subtasks=fit_subtasks, fit_trials=fit_trials, entropy=entropy)
        #subj_id = pd.DataFrame(np.ones((len(data), 1)) * subject).set_axis(['subj_idx'], axis=1, inplace=False)
        #data = pd.concat([data, subj_id], axis=1)
        pred_data = data if subject==0 else pd.concat((data, pred_data))

    # # choose tasks for fitting
    # NUM_SUBTASKS = 1 if experiment == 'noncompositional' else 3
    # STARTTASK = 0
    # if fit_tasks == 'all':
    #     NUM_TASKS = int((pred_data.index.max()+1)/(NUM_TRIALS*NUM_SUBTASKS))  
    # else:
    #     raise ValueError('only works for all tasks')
    # if experiment == 'compositional' or experiment == 'loocompositional':
    #     if fit_subtasks == 'composed':
    #         choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
    #         # trials + length of task * task_id + SUBTASK_ID * NUM_TRIALS
    #         pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)])]
    #     elif fit_subtasks == 'base':
    #         choose_trials = np.arange(NUM_TRIALS*2) if fit_trials == 'all' else ValueError('only works for all trials')
    #         pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
    #     elif fit_subtasks == 'all':
    #         choose_trials = np.arange(NUM_TRIALS*3) if fit_trials == 'all' else ValueError('only works for all trials')
    #         pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
    #     else:
    #         raise ValueError('fit_subtasks must be composed, base, or all')
    # elif experiment == 'noncompositional':
    #     choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
    #     pred_data = pred_data.loc[np.hstack([choose_trials + 1*NUM_TRIALS*task_idx  for task_idx in range(STARTTASK, NUM_TASKS)])]
    # else:
    #     raise ValueError('experiment must be compositional, loocompositional, or noncompositional')

    return pred_data

def load_data(model_name, rule='add', experiment='compositional', fit_tasks='all', fit_subtasks='all', fit_trials='all'):
    # path for pooled rewards
    POOLED_PATH = '{}/{}/'.format(BASEPATH, rule)

    # read pooled preds
    pred_data = pd.read_csv(POOLED_PATH + 'pooled_{}_{}_preds'.format(experiment, model_name), header=0, index_col=0)
    pred_data['prev_arm'] = np.concatenate((np.zeros((1,)), pred_data.loc[:, 'arm'].values))[:-1]

    # choose tasks for fitting
    STARTTASK = 16 if fit_tasks == 'eval' else 0

    if experiment == 'compositional' or experiment == 'loocompositional':
        if fit_subtasks == 'composed':
            if fit_trials == 'all':
                choose_trials = np.arange(NUM_TRIALS)  
            elif fit_trials =='first':
                choose_trials = np.arange(NUM_TRIALS)[0]  
            elif fit_trials =='second':
                choose_trials = np.arange(NUM_TRIALS)[1]  
            elif fit_trials =='third':
                choose_trials = np.arange(NUM_TRIALS)[2]  
            elif fit_trials =='fourth':
                choose_trials = np.arange(NUM_TRIALS)[3]  
            else: 
                choose_trials = np.arange(NUM_TRIALS)[-1]
            # trials + length of task * task_id + SUBTASK_ID * NUM_TRIALS
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'base':
            choose_trials = np.arange(NUM_TRIALS*2) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        else:
            choose_trials = np.arange(NUM_TRIALS*3) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
    
    else:
        choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
        pred_data = pred_data.loc[np.hstack([choose_trials + 1*NUM_TRIALS*task_idx  for task_idx in range(STARTTASK, NUM_TASKS)])]

    return pred_data


def load_data_rl3(model_name, rule='add', experiment='compositional', fit_tasks='all', fit_subtasks='all', fit_trials='all', policy='zeros'):
    # path for pooled rewards
    POOLED_PATH = '{}/'.format('/notebooks/modelfits/baselines_to_rl3/compile_preds')

    # read pooled preds
    pred_data = pd.read_csv(POOLED_PATH + '{}_{}_{}_{}.csv'.format(rule, model_name, experiment, policy), header=0, index_col=0)
    pred_data['prev_arm'] = np.concatenate((np.ones((1,)), pred_data.loc[:, 'arm'].values))[:-1]
    
    # choose tasks for fitting
    NUM_SUBTASKS = 1 if experiment == 'noncompositional' else 3
    STARTTASK = 0
    if fit_tasks == 'all':
        NUM_TASKS = int((pred_data.index.max()+1)/(NUM_TRIALS*NUM_SUBTASKS))  
    else:
        raise ValueError('only works for all tasks')
    if experiment == 'compositional' or experiment == 'loocompositional':
        if fit_subtasks == 'composed':
            choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
            # trials + length of task * task_id + SUBTASK_ID * NUM_TRIALS
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'base':
            choose_trials = np.arange(NUM_TRIALS*2) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'all':
            choose_trials = np.arange(NUM_TRIALS*3) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        else:
            raise ValueError('fit_subtasks must be composed, base, or all')
    elif experiment == 'noncompositional':
        choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
        pred_data = pred_data.loc[np.hstack([choose_trials + 1*NUM_TRIALS*task_idx  for task_idx in range(STARTTASK, NUM_TASKS)])]
    else:
        raise ValueError('experiment must be compositional, loocompositional, or noncompositional')
    return pred_data

def load_baselines_fitted_to_rl3_per_dl(model_name, dl=None, rule='add', experiment='compositional', fit_tasks='all', fit_subtasks='all', fit_trials='all', policy='zeros', entropy=False):
    # path for pooled rewards
    POOLED_PATH = '{}/{}/'.format('/notebooks/scratch/modelfits/baselines_to_rl3', rule)

    # read pooled preds
    read_file = POOLED_PATH + '{}_{}_{}_entropy{}_{}_largeDLs.csv'.format(model_name, experiment, policy, entropy, dl) if entropy else POOLED_PATH + '{}_{}_{}_{}.csv'.format(model_name, experiment, policy, dl)
    pred_data = pd.read_csv(read_file, header=0, index_col=0)
    pred_data['prev_arm'] = np.concatenate((np.ones((1,)), pred_data.loc[:, 'arm'].values))[:-1]
    
    # choose tasks for fitting
    NUM_SUBTASKS = 1 if experiment == 'noncompositional' else 3
    STARTTASK = 0
    if fit_tasks == 'all':
        NUM_TASKS = int((pred_data.index.max()+1)/(NUM_TRIALS*NUM_SUBTASKS))  
    else:
        raise ValueError('only works for all tasks')
    if experiment == 'compositional' or experiment == 'loocompositional':
        if fit_subtasks == 'composed':
            choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
            # trials + length of task * task_id + SUBTASK_ID * NUM_TRIALS
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'base':
            choose_trials = np.arange(NUM_TRIALS*2) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'all':
            choose_trials = np.arange(NUM_TRIALS*3) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        else:
            raise ValueError('fit_subtasks must be composed, base, or all')
    elif experiment == 'noncompositional':
        choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
        pred_data = pred_data.loc[np.hstack([choose_trials + 1*NUM_TRIALS*task_idx  for task_idx in range(STARTTASK, NUM_TASKS)])]
    else:
        raise ValueError('experiment must be compositional, loocompositional, or noncompositional')
    return pred_data

def load_rl3_fitted_to_participants_per_dl(model_name, subject, dl, rule, experiment, fit_tasks, fit_subtasks, fit_trials, entropy):
    # path for pooled rewards
    POOLED_PATH = '/notebooks/scratch/modelfits/rl3_to_participants/'

    # read pooled preds
    RL3_preds, actions = torch.load(POOLED_PATH+'jagadish_likelihoods_subject={}_prior=svdo_changepoint={}_entropy{}.pth'.format(subject, 'True' if rule == 'changepoint' else 'False', entropy))
    fitted_rewards_df = pd.DataFrame(RL3_preds[dl].squeeze().reshape(RL3_preds[dl].shape[0]*RL3_preds[dl].shape[1], RL3_preds[dl].shape[-1]).numpy())
    actions_df = pd.DataFrame(actions.reshape(-1).numpy()).set_axis(['arm'], axis=1, inplace=False)
    mus = fitted_rewards_df.set_axis(['mu_%d' % i for i in range(6)], axis=1, inplace=False)
    description_len = pd.DataFrame(np.ones((len(mus), 1)) * dl).set_axis(['dl'], axis=1, inplace=False)
    pred_data = pd.concat([mus, description_len, actions_df], axis=1) 
    pred_data['prev_arm'] = np.concatenate((np.ones((1,)), pred_data.loc[:, 'arm'].values))[:-1]
    subj_id = pd.DataFrame(np.ones((len(pred_data), 1)) * subject).set_axis(['subj_idx'], axis=1, inplace=False)
    pred_data = pd.concat([pred_data, subj_id], axis=1)
    
    # choose tasks for fitting
    NUM_SUBTASKS = 1 if experiment == 'noncompositional' else 3
    STARTTASK = 0
    if fit_tasks == 'all':
        NUM_TASKS = 20 #int((pred_data.index.max()+1)/(NUM_TRIALS*NUM_SUBTASKS))  
    else:
        raise ValueError('only works for all tasks')
    if experiment == 'compositional' or experiment == 'loocompositional':
        if fit_subtasks == 'composed':
            choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
            # trials + length of task * task_id + SUBTASK_ID * NUM_TRIALS
            # print(np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)]))
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx + 2*NUM_TRIALS  for task_idx in range(STARTTASK, NUM_TASKS)])]#.loc[]
        elif fit_subtasks == 'base':
            choose_trials = np.arange(NUM_TRIALS*2) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        elif fit_subtasks == 'all':
            choose_trials = np.arange(NUM_TRIALS*3) if fit_trials == 'all' else ValueError('only works for all trials')
            pred_data = pred_data.loc[np.hstack([choose_trials + NUM_SUBTASKS*NUM_TRIALS*task_idx for task_idx in range(STARTTASK, NUM_TASKS)])]
        else:
            raise ValueError('fit_subtasks must be composed, base, or all')
    elif experiment == 'noncompositional':
        choose_trials = np.arange(NUM_TRIALS) if fit_trials == 'all' else (np.arange(NUM_TRIALS)[0] if fit_trials =='first' else np.arange(NUM_TRIALS)[-1]) 
        pred_data = pred_data.loc[np.hstack([choose_trials + 1*NUM_TRIALS*task_idx  for task_idx in range(STARTTASK, NUM_TASKS)])]
    else:
        raise ValueError('experiment must be compositional, loocompositional, or noncompositional')
    #print(pred_data)
    return pred_data

    #return pred_data

def save_best_fitted_RL3_models(experiment, fit_subtasks, fit_tasks, fit_trials, rule, entropy):
    model_name = 'RL3'
    n_subjs = 92 if rule == 'add' else 108
    n_dls = 1000
    MODEL_PATH =  f'/notebooks/scratch/modelfits/rl3_to_participants/{rule}/'

    ## find best fitting RL3 models for each subject
    mlls = []
    for subject in np.arange(n_subjs):
        mll = []
        for dl in np.arange(n_dls):
            analysis_name = '{}_{}_{}_{}_{}'.format(model_name, experiment, fit_tasks, fit_subtasks, fit_trials)
            nll_model_name = MODEL_PATH + '{}_entropy{}_model_fits_smc_{}_{}.pkl'.format(analysis_name, entropy, subject, dl)
            nll_per_model = pd.read_pickle(nll_model_name)
            mll.extend(nll_per_model['mll'])
        mlls.append(mll)
    mlls_np = np.stack(mlls)

    best_fits = np.argmax(mlls_np, 1)   
    np.save(f'/notebooks/modelfits/rl3_to_participants/best_RL3_models_per_participant_{rule}.npy', best_fits)
    
class FitSoftmax(nn.Module):
    
    def __init__(self, num_arms=6):
        super().__init__()
        self.beta = nn.Parameter(torch.rand(1)+0.001, requires_grad=True)
        self.tau = nn.Parameter(torch.rand(1)+0.001, requires_grad=True)
        self.stickiness = nn.Parameter(torch.rand(1)+0.001, requires_grad=True)
        self.eps = nn.Parameter(torch.rand(1)+0.001, requires_grad=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.num_arms = num_arms

    def forward(self, x, y=None, prev_arms=None):
        if prev_arms is None:
           prev_arms = np.zeros_like(y)

        if y is None:
            values = x*self.beta
        else:
            values = x*self.beta + y*self.tau 
        # add stickiness term 
        values += prev_arms*self.stickiness
        # # add randomness term
        # x_rand = torch.ones_like(x)/self.num_arms
        # eps = torch.sigmoid(self.eps)
        return self.logsoftmax(values)


class FitEpsGreedy(nn.Module):
    
    def __init__(self, num_arms=6): #, stickiness=False):
        super().__init__()
        self.eps = nn.Parameter(torch.rand(3)+0.001, requires_grad=True) #if stickiness else nn.Parameter(torch.rand(2)+0.001, requires_grad=True) 
        self.num_arms = num_arms 

    def forward(self, x, prev_arms=None):
        if prev_arms is None:
           prev_arms = np.zeros_like(y)
        
        x_rand = torch.ones_like(x)/self.num_arms
        eps = torch.nn.functional.softmax(self.eps)
        
        values = eps[0] * x + eps[1] * x_rand + eps[2] * prev_arms

        # if self.stickiness:
        #     values = eps[0] * x + eps[1] * x_rand + eps[2] * prev_arms
        # else:
        #     values = eps[0] * x + eps[1] * x_rand

        return torch.log(values)

    