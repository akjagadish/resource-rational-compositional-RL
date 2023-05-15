import numpy as np
import pandas as pd
import torch


def return_model_comparison_metrics(models, rule='add', experiment='compositional', fit_tasks='all', fit_subtasks='composed', fit_trials='all', stickiness=True):
    nlls, bics, bfs, params, r2, logevidence = [], [], [], [], [], []
    MODEL_PATH =  '/notebooks/modelfits/softmax_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/softmax/{}/'.format(rule)
    NUM_SAMPLES = 100 if fit_subtasks == 'composed' else 300
    NUM_PARAMETERS = {'mean_tracker': 2, 'mean_tracker_compositional': 2, 'oracle': 1, 'GOD': 1, 
                  'nn_uvfa': 1, 'simple_grammar_constrained': 2, 
                  'LSTMQL_epsgreedy': 1, 'grammar': 2, 'random': 0,
                  'LSTMQL_epsgreedy_agent': 1, 'metarl_human_a2c': 1,
                  'corner': 0, 'constant_choice': 1}
    order_rl3, order_baselines = np.load(f'/notebooks/modelfits/rl3_to_participants/rl3_subject_order_{rule}.npy'), np.load(f'/notebooks/modelfits/rl3_to_participants/baselines_subject_order_{rule}.npy')
    # order_baselines=order_rl3
    # random policy
    NLL_random = np.array(-NUM_SAMPLES*np.log(1/6))
    BIC_random = 2*NLL_random

    for ix, model_name in enumerate(models):

        analysis_name = '{}_{}_{}_{}_{}'.format(model_name, experiment, fit_tasks, fit_subtasks, fit_trials)
        if model_name == 'random':
            nlls.append([NLL_random.mean() , NLL_random.std()])
            bics.append([BIC_random.mean() , BIC_random.std()])
            r2.append(np.zeros((2,)))
            params.append(np.zeros((6,)) if stickiness else np.zeros((4,)))
            logevidence.append(np.repeat(-BIC_random/2, NUM_SUBJECTS))

        elif model_name == 'rl3':
            # used the BIC values from Marcel
            rl3_bic = torch.load('/notebooks/modelfits/rl3_stickiness/{}/{}.pth'.format(rule, analysis_name))[order_rl3] if stickiness else torch.load('/notebooks/modelfits/rl3/{}/{}.pth'.format(rule, analysis_name))
            R2 = 1-rl3_bic/BIC_random
            logevidence.append(-np.array(rl3_bic.tolist())/2)
            bics.append([rl3_bic.mean() , rl3_bic.std()])
            params.append(np.zeros((6,)) if stickiness else np.zeros((4,)))
            r2.append([R2.mean(), R2.std()])
            nlls.append(np.zeros((2,)))

        elif model_name == 'rl2':
            rl2_bic = torch.load('/notebooks/modelfits/rl2_stickiness/{}_{}.pth'.format(rule, analysis_name))[order_rl3] if stickiness else np.NaN
            R2 = 1-rl2_bic/BIC_random
            logevidence.append(-np.array(rl2_bic.tolist())/2)
            bics.append([rl2_bic.mean(), rl2_bic.std()])
            params.append(np.zeros((6,)) if stickiness else np.zeros((4,)))
            r2.append([R2.mean(), R2.std()])
            nlls.append(np.zeros((2,)))

        elif model_name == 'metarl_human_a2c':
            a2c_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            a2c_path = a2c_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_a2c = pd.read_pickle(a2c_path)
            NLL = nll_a2c['nll_loss'] 
            BIC = 2*NLL + NUM_PARAMETERS[model_name]*np.log(NUM_SAMPLES)
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values[order_baselines]/2)   
            params.append([nll_a2c['eps'].mean(), nll_a2c['eps'].std(), 0, 0., nll_a2c['stickiness'].mean(), nll_a2c['stickiness'].std()] if stickiness else  [nll_a2c['eps'].mean(), nll_a2c['eps'].std(), 0, 0])
        
        elif model_name == 'corner':
            corner_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            corner_path = corner_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_corner = pd.read_pickle(corner_path)
            NLL = nll_corner['nll_loss'] 
            BIC = 2*NLL 
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  
        
        elif model_name == 'left_to_right':
            ltr_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            ltr_path = ltr_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_corner = pd.read_pickle(ltr_path)
            NLL = nll_corner['nll_loss'] 
            BIC = 2*NLL 
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  

        elif model_name == 'right_to_left':
            rtl_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            rtl_path = ltr_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_corner = pd.read_pickle(rtl_path)
            NLL = nll_corner['nll_loss'] 
            BIC = 2*NLL 
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  
    
        elif model_name == 'left_right_center':
            lrc_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            lrc_path = ltr_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_corner = pd.read_pickle(lrc_path)
            NLL = nll_corner['nll_loss'] 
            BIC = 2*NLL 
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  
        
        elif model_name == 'best_of_four':
            bof_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
            bof_path = ltr_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_corner = pd.read_pickle(bof_path)
            NLL = nll_corner['nll_loss'] 
            BIC = 2*NLL 
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  
        
        elif model_name == 'constant_choice': #model_name[:-1] == 'constant_choice_':
            pp = []
            for cc in range(6):
                analysis_name = '{}_{}_{}_{}_{}'.format(model_name + "_{}".format(cc), experiment, fit_tasks, fit_subtasks, fit_trials)
                cc_root = '/notebooks/modelfits/epsgreedy_stickiness/{}/'.format(rule) if stickiness else '/notebooks/modelfits/epsgreedy/{}/'.format(rule)
                cc_path = cc_root + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
                nll_cc = pd.read_pickle(cc_path)
                NLL = nll_cc['nll_loss'] 
                pp.append(NLL)
            NLL, _ = torch.tensor(pp).min(0)
            BIC = 2*NLL + NUM_PARAMETERS[model_name]*np.log(NUM_SAMPLES)
            R2 = 1-BIC/BIC_random
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC/2)   
            params.append([nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0., nll_corner['stickiness'].mean(), nll_corner['stickiness'].std()] if stickiness else  [nll_corner['eps'].mean(), nll_corner['eps'].std(), 0, 0])  

        else:
            nll_model_name = MODEL_PATH + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
            nll_per_model = pd.read_pickle(nll_model_name)
            NLL = nll_per_model['nll_loss'] 
            BIC = 2*NLL + NUM_PARAMETERS[model_name]*np.log(NUM_SAMPLES)
            R2 = 1-BIC/BIC_random
            NUM_SUBJECTS = len(nll_per_model['nll_loss'])
            param = [nll_per_model['beta'].mean(), nll_per_model['beta'].std(), nll_per_model['tau'].mean(), nll_per_model['tau'].std(),  
                    nll_per_model['stickiness'].mean(), nll_per_model['stickiness'].std()] if stickiness else [nll_per_model['beta'].mean(), 
                                                                                                                nll_per_model['beta'].std(), nll_per_model['tau'].mean(), nll_per_model['tau'].std()]
            params.append(param)
            nlls.append([NLL.mean(), NLL.std()])
            bics.append([BIC.mean(), BIC.std()])
            r2.append([R2.mean(), R2.std()])
            logevidence.append(-BIC.values[order_baselines]/2)

    nlls = np.stack(nlls)
    bics = np.stack(bics)
    r2 = np.stack(r2)
    logevidence = np.stack(logevidence)

    return nlls, bics, r2, logevidence, params


def compute_pertrial_nll_softmax(models, rule='add', experiment='compositional', fit_tasks='all', fit_subtasks='composed', fit_trials='all', stickiness=True):

    
    MODEL_PATH =  '/notebooks/modelfits/softmax_stickiness/{}/'.format(RULE) if stickiness else '/notebooks/modelfits/softmax/{}/'.format(RULE)
    NUM_ARMS = 6

    for model_ix, model_name in enumerate(models):

        # init
        negativeloglikelihoods = {}
        logsoftmax = nn.LogSoftmax(dim=1)
        num_participants = int(max(pred_data['subj_idx'])+1)

        # load model parameters
        analysis_name = '{}_{}_{}_{}_{}'.format(model_name, experiment, fit_tasks, fit_subtasks, fit_trials)
        nll_model_name = MODEL_PATH + '{}_model_fits_{}.pkl'.format(analysis_name, 'softmax')
        nll_per_model = pd.read_pickle(nll_model_name)
        pred_data = load_data(model_name, rule=RULE, experiment=experiment, fit_tasks=fit_tasks, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
        
        # loop over participants
        for subj_idx in np.arange(num_participants): 
            
            # get per-participant fitted parameters
            beta = nll_per_model['beta'][subj_idx]
            tau = nll_per_model['tau'][subj_idx]
            stickiness =  nll_per_model['stickiness'][subj_idx] if stickiness else 0.
            
            # load human choices
            predicted_rewards = torch.tensor(np.array([pred_data[pred_data['subj_idx']==subj_idx].loc[:, 'mu_%d' % ii].values for ii in range(NUM_ARMS)])).T
            predicted_uncertainities = torch.tensor(np.array([pred_data[pred_data['subj_idx']==subj_idx].loc[:, 'sigma_%d' % ii].values for ii in range(NUM_ARMS)])).T
            arms = torch.tensor(pred_data[pred_data['subj_idx']==subj_idx]['arm'].values).type(torch.LongTensor)
            prev_arms = construct_sticky_choice(pred_data[pred_data['subj_idx']==subj_idx])
            
            # compute softmax over means, uncertainities and stickiness
            logsoftmax_probs = logsoftmax(beta*predicted_rewards + tau*predicted_uncertainities + stickiness*prev_arms)
            logsoftmax_probs_action = torch.stack([logsoftmax_probs[ii, arms[ii]] for ii, xx in enumerate(logsoftmax_probs)]).reshape(20, 15)
            negativeloglikelihoods[model_name] = [-logsoftmax_probs_action.numpy()]
            # store in nll in df
            model_df = pd.DataFrame(negativeloglikelihoods, index=[subj_idx]) if subj_idx==0 else pd.concat((model_df, pd.DataFrame(negativeloglikelihoods, index=[subj_idx])))
        
        # store dfs for different models
        df = model_df if model_ix==0 else pd.concat([df, model_df], axis=1)

    return df

def return_mll_estimates(models, fit_subtasks, fit_trials, rule='add', experiment='compositional', fit_tasks='all'):

    L = []
    order_rl3, order_baselines = np.load(f'/notebooks/modelfits/rl3_to_participants/rl3_subject_order_{rule}.npy'), np.load(f'/notebooks/modelfits/rl3_to_participants/baselines_subject_order_{rule}.npy')
    seed=0
    # load mll estimates
    for ix, model_name in enumerate(models):
        analysis_name = '{}_{}_{}_{}_{}'.format(model_name, experiment, fit_tasks, fit_subtasks, fit_trials, seed)
        if model_name == 'random':
            pass  

        elif model_name == 'rl3':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'/notebooks/modelfits/rl3_to_participants/mlls_rl3_entropy_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3])
        
        elif model_name == 'rl3_old':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'/notebooks/modelfits/rl3_to_participants/RL3_old/mlls_rl3_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3])

        elif model_name == 'rl2':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'/notebooks/modelfits/rl2_to_participants/mlls_rl2_entropy_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3])

        else:
            mlls = np.load(f'/notebooks/modelfits/baselines_to_participants/{rule}/{analysis_name}.npy')
            L.append(mlls.squeeze()[order_baselines])
            
    return np.stack(L)


def return_mll_estimates_per_trial(models, fit_subtasks, fit_trials, rule='add', experiment='compositional', fit_tasks='all'):

    L = []
    order_rl3, order_baselines = np.load(f'../modelfits/rl3_to_participants/rl3_subject_order_{rule}.npy'), np.load(f'../modelfits/rl3_to_participants/baselines_subject_order_{rule}.npy')
    seed=0
    trial = 0 if fit_trials=='first' else 4 if fit_trials == 'last' else None
    # load mll estimates
    for ix, model_name in enumerate(models):
        
        if model_name == 'random':
            pass  
        elif model_name == 'rl3':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/pertrials_mlls_rl3_entropy_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3, trial])
        
        elif model_name == 'rl3_old':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'../../RL3NeurIPS/data/model_fits/pertrials_mlls_rl3_entropy_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3, trial])

        elif model_name == 'rl2':
            subtask =  'last' if fit_subtasks == 'composed' else 'full'
            mlls = torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/pertrials_mlls_rl2_entropy_{rule}_{subtask}.pth')
            L.append(mlls[order_rl3, trial])
        else:
            analysis_name = '{}_{}_{}_{}_{}'.format(model_name, experiment, fit_tasks, fit_subtasks, fit_trials, seed)
            mlls = np.load(f'../modelfits/baselines_to_participants/{rule}/{analysis_name}.npy')
            L.append(mlls.squeeze()[order_baselines])
            
    return np.stack(L)


