import numpy as np
import torch
from collections import defaultdict

class SimulationFit():
    def __init__(self, condition='compositional', num_trials=5, num_arms=6, rule='add', policy='epsgreedy', return_all=False):
        self.condition =  condition
        self.feature_dict_primitive = {"linpos": np.array([1, 0]),  "linneg": np.array([1, 0]), 
                                       "pereven": np.array([0, 1]), "perodd": np.array([0, 1])}
        self.feature_dict = defaultdict(lambda:np.array([1, 1]),  self.feature_dict_primitive)
        self.num_trials = num_trials
        self.num_arms  = num_arms
        self.num_subtasks = 3 if condition=='compositional' else 1
        self.rule = rule
        self.eval_arms = torch.arange(self.num_arms)/(self.num_arms-1)
        self.policy = policy
        self.return_all=return_all

    def pick_action(self, model, reward_estimates, uncertainity_estimates, params):
        if self.policy == 'epsgreedy':
            eps = params[0]
            act =  model.eps_greedy_choice(torch.arange(self.num_arms), torch.tensor(reward_estimates), eps=eps)
        elif self.policy  == 'softmax':
            beta, tau = params
            act = model.UCB_softmax_choice(torch.arange(self.num_arms), reward_estimates, uncertainity_estimates, beta=beta, tau=tau)
        elif self.policy == 'sticky_ucb':
            beta, tau, sticky, prev_arm = params
            act = model.sticky_UCB_softmax_choice(torch.arange(self.num_arms), reward_estimates, uncertainity_estimates, beta=beta, tau=tau, sticky=sticky, prev_arm=prev_arm)
        return act 

    def fit(self, model, true_rewards, description, params):
        r_vals = np.zeros((self.num_subtasks, self.num_trials, self.num_arms))
        r_sigma = np.zeros((self.num_subtasks, self.num_trials, self.num_arms))
        regrets = np.zeros((self.num_subtasks, self.num_trials))
        rewards = np.zeros((self.num_subtasks, self.num_trials))
        best_actions = np.zeros((self.num_subtasks, self.num_trials))
        actions = np.zeros((self.num_subtasks, self.num_trials))

        for subtask_idx in range(self.num_subtasks):
            Y = true_rewards[subtask_idx]
            X_, y_, action = None, None, None
            counter = 0
            feature_vector = self.feature_dict[description[subtask_idx]] # access vector representations
            model.new_task(feature_vector)
            # task_is_compositional = feature_vector.sum() == 2  # there are two ones in the vector
            prev_action = 0 if action is None else action
            for trial_idx in range(self.num_trials):
                
                model.transfer(feature_vector, self.rule)  # check if transfer is available for compositional subtask
                model.condition(X_, y_) # condition on t-1 training data points

                ## collect reward and uncertainty estimates from model 
                reward_estimates, uncertainty_estimates = model.predict_rewards(self.eval_arms)
                r_vals[subtask_idx, counter] = reward_estimates
                r_sigma[subtask_idx, counter] = uncertainty_estimates
                counter += 1 # increment

                # store regrets being greedy in compositional tassk
                if self.policy == 'epsgreedy':
                    parameters = params.copy()
                    # assert len(params) == 1, 'epsgreedy cannot get more than 1 parameter' 
                    # params = 0. if subtask_idx==(self.num_subtasks-1) else params
                elif self.policy == 'sticky_ucb':
                    parameters = params.copy()
                    parameters.append(prev_action)
                    
                else:
                    parameters = params.copy()
                
                act = self.pick_action(model, reward_estimates, uncertainty_estimates, parameters)
                reward = torch.tensor([Y[act]])
                prev_action = act
                action = torch.tensor([self.eval_arms[act]])
                regret = Y[torch.argmax(Y)] - Y[act]
                
                # if (i==(self.num_subtasks-1) and trial_idx==0):
                #     greedy_act = np.argmax(reward_estimates)
                #     regret = Y[torch.argmax(Y)] - Y[greedy_act]
                    # re = reward_estimates
                best_actions[subtask_idx, trial_idx] = int(torch.argmax(Y))
                actions[subtask_idx, trial_idx] = int(act)
                regrets[subtask_idx, trial_idx] = regret
                rewards[subtask_idx, trial_idx] = reward

                (X_, y_) = (action, reward)  if trial_idx==0 else (torch.cat((X_, action)), torch.cat((y_, reward)))
                
                model.fit(X_,y_)

                if trial_idx == (self.num_trials-1):
                    model.update(X_,y_)  # update model
            
        print("Done!")
        if self.return_all:
            return regrets, actions, best_actions, rewards
        return regrets, X_
