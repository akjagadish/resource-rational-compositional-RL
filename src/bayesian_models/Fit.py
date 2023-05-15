import numpy as np
import torch
from collections import defaultdict

NUM_TRIALS = 5
NUM_ARMS = 6
NUM_SUBTASKS = 3
eval_arms = torch.arange(NUM_ARMS)/(NUM_ARMS-1)

def fit(model, true_rewards, eps):
    r_vals = np.zeros((NUM_SUBTASKS, NUM_TRIALS, NUM_ARMS))
    r_sigma = np.zeros((NUM_SUBTASKS, NUM_TRIALS, NUM_ARMS))

    for i in range(NUM_SUBTASKS):
        Y = true_rewards[i]
        X_, y_ = None, None
        model.kernel = model.lin if i==0 else (model.per if i==1 else model.comp)
        counter = 0
        for jj in range(NUM_TRIALS):

            # model.transfer(feature_vector)  # check if transfer is available
            model.condition(X_, y_) # condition on t-1 training data points
            # print(model.kernel, jj, X_, y_, eval_arms.shape)
            ## collect reward and uncertainty estimates from model
            reward_estimates, uncertainty = model.predict_rewards(eval_arms)
            r_vals[i, counter] = reward_estimates
            r_sigma[i, counter] = uncertainty
            counter += 1 # increment
            act = model.eps_greedy_choice(torch.arange(NUM_ARMS), torch.tensor(reward_estimates), eps=eps)
            reward = torch.tensor([Y[act]])
            action = torch.tensor([eval_arms[act]])
            (X_, y_) = (action, reward)  if jj==0 else (torch.cat((X_, action)), torch.cat((y_, reward)))
            
            model.fit(X_,y_)

    print("Done!")
    
    return r_vals, X_
    
condition = 'compositional'
feature_dict_primitive = {"linpos": np.array([1, 0]),  "linneg": np.array([1, 0]), 
                          "pereven": np.array([0, 1]), "perodd": np.array([0, 1])}
feature_dict = defaultdict(lambda:np.array([1, 1]), feature_dict_primitive)
        
def fit_simplegrammar(model, true_rewards, description, eps, rule='add'):
    NUM_SUBTASKS = len(description)
    r_vals = np.zeros((NUM_SUBTASKS, NUM_TRIALS, NUM_ARMS))
    r_sigma = np.zeros((NUM_SUBTASKS, NUM_TRIALS, NUM_ARMS))
    regrets = np.zeros((NUM_SUBTASKS, NUM_TRIALS))

    for i in range(NUM_SUBTASKS):
        Y = true_rewards[i]
        X_, y_ = None, None
        counter = 0
        feature_vector = feature_dict[description[i]] # access vector representation
        model.new_task(feature_vector)
        task_is_compositional = feature_vector.sum() == 2  # there are two ones in the vector
        
        for jj in range(NUM_TRIALS):
            #print(jj)
            if (i==2 and jj==0):
                model.transfer(feature_vector, rule)  # check if transfer is available
            model.condition(X_, y_) # condition on t-1 training data points
            ## collect reward and uncertainty estimates from model
            idx = False if (i==2 and jj==0) else True
            reward_estimates, uncertainty = model.predict_rewards(eval_arms, idx)
            r_vals[i, counter] = reward_estimates
            r_sigma[i, counter] = uncertainty
            counter += 1 # increment

            eps = 0. if (i==2 and jj==0) else eps
            act =  model.eps_greedy_choice(torch.arange(NUM_ARMS), torch.tensor(reward_estimates), eps=eps) 
            reward = torch.tensor([Y[act]])
            action = torch.tensor([eval_arms[act]])
            regret = Y[torch.argmax(Y)] - Y[act]
            if (i==(NUM_SUBTASKS-1) and jj==0):
                greedy_act = np.argmax(reward_estimates)
                regret = Y[torch.argmax(Y)] - Y[greedy_act]
                re = reward_estimates

            regrets[i, jj] = regret
            (X_, y_) = (action, reward)  if jj==0 else (torch.cat((X_, action)), torch.cat((y_, reward)))
            
            model.fit(X_,y_)

            if jj == (NUM_TRIALS-1):
                # print('update the means and covs')
                model.update(X_,y_)  # update model
                if (condition != 'noncompositional') and task_is_compositional:
                        # reset episodic dictionary after each compositional task
                        if model.has_episodic_dict:
                            model.episodic_dict.reset()
        
    print("Done!")
    
    return regrets, X_, re #regret, greedy_act, re, r_vals, X_, y_