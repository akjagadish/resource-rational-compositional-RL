import numpy as np
import pandas as pd
import math
import torch
import gpytorch
import GP #from models.grammar.GP
from os import listdir
import json
from collections import defaultdict
from matplotlib import pyplot as plt
from ChoiceModel import RBFModelMemorySubTask


class ParticipantFit():
    def __init__(self, participant_id, model, params, path ="../../experiment/add_data/add_data/", savefolder = "grammar_preds", save_path = None, rule=None):
        self.participant_id = participant_id
        self.participant_name = participant_id[:-5]  # removes the ".json" part
        # self._data_folder = "../../experiment/add_data/add_data/"
        # self._data_folder = "../../experiment/add_data/add_data/"
        self._data_folder = path#"experiment/add_data/add_data/"
        self._data_path = self._data_folder + self.participant_id
        self.rule = rule
        with open(self._data_path) as f:
            self.data = json.load(f)

            # unpack
            self.actions = self.data["actions"]
            self.rewards = self.data["rewards"]
            self.contexts = self.data["condition"] # the task descriptions, a bit confusing nomenclature
            self.condition = self.data["experiment"] # the actual condition of the participant
            self.RTs = self.data["times"]



        self.model = model
        self.params = params
        self.has_been_fitted = False

        self.task_length = len(self.actions[0]) # number of trials per task
        self.num_tasks = len(self.actions)  # number of tasks
        self.num_trials = self.num_tasks * self.task_length  # number of tasks times number of trials per task
        self.num_actions = 6
        feature_dict_primitive = {"pos": np.array([1, 0]), "neg":np.array([1, 0]), "even": np.array([0, 1]),
                                  "odd":np.array([0, 1])}

        self.feature_dict = defaultdict(lambda:np.array([1, 1]), feature_dict_primitive)
        self.eval_arms = torch.arange(6)
        self.max_arm = 5
        self.min_arm = 0
        self.eval_arms = self.minmax(self.eval_arms)
        self.contexts_unpacked = self.contexts
        self.unpack_contexts()
        order = defaultdict(lambda: True, 
                               {"evenpos": False, "oddpos": False,
                                "evenneg": False, "oddneg": False})
        self.linfirst = order[np.array(self.contexts)[-1]]
        self.model.set_kernel(self.linfirst)
        self.save_path = f"/notebooks/models/grammar/reward_preds/{savefolder}/{self.participant_name}" if save_path is None else save_path + self.participant_name
        # for debugging
        self.ctx_predictions = {}
        self.ctx_X = {}
        self.ctx_Y = {}
        self.max_R = 20
        self.min_R = 0


    def unpack_contexts(self):
        ## now we'll unpack the contexts list.
        # make a new list which is the same length as the number of tasks
        new_context_list = []
        for c in self.contexts:
            for c_ in c:
                new_context_list.append(c_)

        self.contexts = new_context_list

    def create_full_dataset(self):
        self.data_x = torch.zeros(self.num_trials, 3) # number of trials x 1 action dimension and 2 context dimensions
        self.data_x[:, 0] = self.minmax(torch.tensor(self.actions)).ravel()
        self.data_y = torch.tensor(self.rewards).ravel()
        counter = 0
        for i in range(self.num_tasks):
            context = self.contexts[i]
            ctx_features = torch.tensor(self.feature_dict[context])

            for j in range(self.task_length):
                self.data_x[counter, 1:] = ctx_features
                counter += 1

    def minmax(self, X):
        X = (X - self.min_arm) / (self.max_arm - self.min_arm)
        return X

    def minmax_R(self, R):
        R = (R - self.min_R) / (self.max_R - self.min_R)
        return R


    def fit(self):
        #self.Q_vals = np.zeros((self.num_trials, self.num_actions))
        self.r_vals = np.zeros((self.num_trials, self.num_actions))
        self.r_sigma = np.zeros((self.num_trials, self.num_actions))
        # loop over action, reward pairs
        counter = 0
        for i, task in enumerate(zip(self.actions, self.rewards, self.contexts)):

            X, y, description = task  # unpack task tuple
            X = torch.tensor(X)
            X = self.minmax(X)
            y = torch.tensor(y)
            y = self.minmax_R(y)

            feature_vector = self.feature_dict[description] # access vector representation
            task_is_compositional = feature_vector.sum() == 2  # there are two ones in the vector

            self.model.new_task(feature_vector)
            for j in range(self.task_length):
                if j == 0:
                    X_, y_ = None, None
                else:
                    X_, y_ = X[:j], y[:j]  # data to condition on prior to choice
                X_prime, y_prime = X[:j+1], y[:j+1]  # data received after choice


                self.model.transfer(feature_vector, self.rule, self.linfirst)  # check if transfer is available
                self.model.condition(X_, y_) # condition on t-1 training data points

                ## collect reward and uncertainty estimates from model
                reward_estimates, uncertainty = self.model.predict_rewards(self.eval_arms)
                self.r_vals[counter] = reward_estimates
                self.r_sigma[counter] = uncertainty
                counter += 1 # increment

                self.model.fit(X_prime, y_prime)

                if j == (self.task_length - 1):  # last trial in task
                    #self.model.condition(X_prime, y_prime)  # condition the model before saving
                    self.model.update(X_prime, y_prime)  # update model

                    #print progress
                    print("Progress ", np.round(((i/len(self.actions))*100), 2) , "%", end="\r")

                    if (self.condition != 'noncompositional') and task_is_compositional:
                        # reset episodic dictionary after each compositional task
                        if self.model.has_episodic_dict:
                            self.model.episodic_dict.reset()

                        if type(self.model) == RBFModelMemorySubTask:
                            self.model.reset()
        print("Done!")


    def fit_all_tasks(self):
        #self.Q_vals = np.zeros((self.num_trials, self.num_actions))
        self.r_vals = np.zeros((self.num_trials, self.num_actions))
        self.r_sigma = np.zeros((self.num_trials, self.num_actions))
        # loop over action, reward pairs

        if not self.model.sees_context_features:
            self.data_x = self.data_x[:, 0]
            test_x = self.eval_arms.clone()

        for i in range(self.num_trials):

            X, y = self.data_x[:i+1], self.data_y[:i+1]  # data received after choice
            ## collect reward and uncertainty estimates from model
            if self.model.sees_context_features:
                feature_vec = self.data_x[i, 1:]
                test_x = torch.zeros(len(self.eval_arms), 3)
                test_x[:, 0] = self.eval_arms
                test_x[:, 1:] = feature_vec
            reward_estimates, uncertainty = self.model.predict_rewards(test_x) # GP predicts on whatever it was fitted on last trial
            self.r_vals[i] = reward_estimates
            self.r_sigma[i] = uncertainty

            self.model.fit(X, y)

            print("Progress ", np.round(((i/self.num_trials)*100), 2) , "%", end="\r")
        print("Done!")



    def to_csv(self):

        self.rewards_df = pd.DataFrame(self.r_vals)
        self.sigma_df = pd.DataFrame(self.r_sigma)
        reward_path = f"{self.save_path}_rewards.csv"
        sigma_path =f"{self.save_path}_uncertainty.csv"
        self.rewards_df.to_csv(reward_path, header=False, index=False)
        self.sigma_df.to_csv(sigma_path, header=False, index=False)
