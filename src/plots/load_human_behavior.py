import os
import torch
import numpy as np
import json
import pandas as pd
from operator import itemgetter

NUM_TRIALS = 5
NUM_ARMS = 6

def load_behavior(rule):

    DATAPATH = '../experiment/{}_data/{}_data/'.format(rule,rule)
    files = os.listdir(path=DATAPATH) 
    subjects = []
    for idx, file in enumerate(files):
        if file.endswith('.json'):
            true_rewards, params, optimal_actions, best_actions = [], {'linear':[], 'periodic':[], 'composition':[]}, [], []
            with open(DATAPATH+file) as json_file:
                subjdata = json.load(json_file)
                actions =  np.asarray(subjdata['actions'])
                envs =  np.asarray(subjdata['envs'])
                envs = envs[:20] if (len(envs) == 21) or (len(envs) == 22) else envs
                envs = envs[:60] if (len(envs) == 61) or (len(envs) == 62) else envs
                temp = []
                for env_idx, env_path in enumerate(envs):
                    env_path = '../experiment/' + env_path
                    with open(env_path) as json_file:
                        jsondata = json.load(json_file)
                        true_rewards.append(jsondata['y'])
                        best_action = np.argmax(jsondata['y'])
                        subj_optimal = actions[env_idx]==best_action
                        optimal_actions.append(subj_optimal)   
                        best_actions.append(best_action)
                        # load params
                        if (subjdata['experiment'] == 'compositional') or (subjdata['experiment'] == 'loocompositional'):
                            if (env_idx+1)%3==1:
                                params['linear'].append(jsondata['params'])
                            elif (env_idx+1)%3==2:
                                params['periodic'].append(jsondata['params'])
                            elif (env_idx+1)%3==0:
                                params['composition'].append(jsondata['params'])
                        else:
                            params['composition'].append(jsondata['params'])


                true_rewards = np.stack(true_rewards)
                optimal_actions = np.stack(optimal_actions)
                one_subject = {'condition': np.array(subjdata['condition']).reshape(-1), 
            'actions': actions, 'rewards': np.asarray(subjdata['rewards']), 'envs': envs, 
            'experiment': np.asarray(subjdata['experiment']), 'money': subjdata['money'], 'times': np.asarray(subjdata['times']),  
            'instcounter': np.asarray(subjdata['instcounter']), 'gender':np.asarray(subjdata['gender']), 'true_rewards': true_rewards, 
            'optimal_actions': optimal_actions, 'age': np.asarray(subjdata['age']), 'params': params, 
            'regrets': np.array(subjdata['regrets']),'subjectID': subjdata['subjectID'],'hand': np.asarray(subjdata['hand']),
            'rewardorder': np.array(subjdata['rewardorder'] if 'rewardorder' in subjdata.keys() else None),
                'best_actions': best_actions}

                subjects.append(one_subject)

    # make dataframe
    data = pd.DataFrame(subjects)
    
    return data

def load_percondition_metrics(rule, drop=False, only_eval=False, load_primitives=False, primitive=0):

    data = load_behavior(rule)
    NUM_SUBJ = len(data)
    optimal_actions_dict, pooled_rewards_dict, pooled_regrets_dict, pooled_rewards_tasks_dict, pooled_regrets_tasks_dict = {}, {}, {}, {}, {}
    pooled_times_dict, pooled_times_tasks_dict = {}, {}
    n_subjs, mean_random_rewards, mean_random_regrets = {}, {}, {}
    pick_subtask = primitive if load_primitives else 2

    for exp in np.unique(np.stack(data.experiment.array)):
        
        pooled_rewards_trialwise, pooled_rewards_taskwise, pooled_regrets_trialwise, pooled_regrets_taskwise = [], [], [], []
        pooled_times_trialwise, pooled_times_taskwise = [], []
        random_rewards, random_regrets = [], []
        exp_data = data[data.experiment==exp]
        SUBTASK = pick_subtask if (exp =='compositional' or exp == 'loocompositional') else 0
        NUM_BLOCKS = 3 if (exp =='compositional' or exp == 'loocompositional' ) else 1
        NUM_TASKS = 20
        NUM_EVAL_TASK = 1 if (exp == 'noncompositional' or exp =='compositional') else 1
        
        # pool rewards
        pooled_true_rewards = np.vstack(exp_data.true_rewards.array)
        pooled_true_rewards = pooled_true_rewards[SUBTASK::NUM_BLOCKS]

        # rewards and regrets for random policy
        random_actions = torch.randint_like(torch.tensor(np.vstack(exp_data.actions.array)), low=0, high=5)
        random_actions = random_actions[SUBTASK::NUM_BLOCKS] 
        for jj in range(NUM_TRIALS):
            for ii in range(len(pooled_true_rewards)):
                random_rewards.append(pooled_true_rewards[ii, random_actions[ii, jj]]/pooled_true_rewards[ii].max())
                random_regrets.append(pooled_true_rewards[ii].max()-pooled_true_rewards[ii, random_actions[ii, jj]])
        mean_random_rewards[exp] = np.stack(random_rewards).mean()
        mean_random_regrets[exp] = np.stack(random_regrets).mean()
        
        if drop:
            drop_subjs = np.stack(np.asarray(exp_data.regrets.array)).mean(2)[:, :-NUM_EVAL_TASK].mean(1)>mean_random_regrets[exp] 
            exp_data.drop(index=data[(data['experiment']==exp)].index[drop_subjs], axis=0, inplace=True)
            data.drop(index=data[(data['experiment']==exp)].index[drop_subjs], axis=0, inplace=True)
        
        # normalized rewards, regrets pooling
        for idx in range(len(exp_data)):
            norm_rewards = exp_data.iloc[idx].rewards/(exp_data.iloc[idx].true_rewards.max(1)+0.3).reshape(NUM_BLOCKS*NUM_TASKS,1)
            norm_rewards_subtask = norm_rewards[SUBTASK::NUM_BLOCKS]
            regrets =  exp_data.iloc[idx].regrets
            regrets_subtask = regrets[SUBTASK::NUM_BLOCKS]
            times = exp_data.iloc[idx].times
            times_subtask = times[SUBTASK::NUM_BLOCKS]
            pooled_times_taskwise.append(np.expand_dims(times_subtask.mean(1), 0))
            pooled_rewards_taskwise.append(np.expand_dims(norm_rewards_subtask.mean(1), 0))
            pooled_regrets_taskwise.append(np.expand_dims(regrets_subtask.mean(1), 0))
            # eval or not
            norm_rewards_subtask = norm_rewards_subtask[-NUM_EVAL_TASK:] if only_eval else norm_rewards_subtask
            regrets_subtask = regrets_subtask[-NUM_EVAL_TASK:] if only_eval else regrets_subtask
            pooled_rewards_trialwise.append(np.expand_dims(norm_rewards_subtask, 0))
            pooled_regrets_trialwise.append(np.expand_dims(regrets_subtask, 0))
            pooled_times_trialwise.append(np.expand_dims(times_subtask, 0))
        
        # construction dicts
        optimal_actions_dict[exp] = np.stack(exp_data['optimal_actions'].array)[:,SUBTASK::NUM_BLOCKS][:, -NUM_EVAL_TASK:] if only_eval else  np.stack(exp_data['optimal_actions'].array)[:,SUBTASK::NUM_BLOCKS]
        pooled_rewards_dict[exp] = np.vstack(pooled_rewards_trialwise)
        pooled_regrets_dict[exp] = np.vstack(pooled_regrets_trialwise)
        pooled_times_dict[exp] = np.vstack(pooled_times_trialwise)
        pooled_rewards_tasks_dict[exp] =  np.vstack(pooled_rewards_taskwise)
        pooled_regrets_tasks_dict[exp] = np.vstack(pooled_regrets_taskwise)
        pooled_times_tasks_dict[exp] = np.vstack(pooled_times_taskwise)
        n_subjs[exp] = len(exp_data)
            
    return optimal_actions_dict, pooled_rewards_dict, pooled_regrets_dict, pooled_times_dict, pooled_rewards_tasks_dict, pooled_regrets_tasks_dict, pooled_times_tasks_dict, n_subjs