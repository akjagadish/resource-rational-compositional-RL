#%%codecell
import numpy as np
import pandas as pd
import math
import torch
import gpytorch

#%%codecell
from GP import GP, CompositionalMean
from os import listdir
import json
from importlib import reload
from collections import defaultdict
from matplotlib import pyplot as plt
from ParticipantFit import ParticipantFit
from ChoiceModel import GrammarModel, ChoiceModel, SimpleGrammarModel, SimpleGrammarModelConstrained, SimpleGrammarModelConstrainedChangePoint
from KernelGrammar import KernelGrammar
from EpisodicDictionary import EpisodicDictionary
from Kernels import Kernels
#%%codecell

rule = 'changepoint'

# data_folder = "../../experiment/add_data/add_data/"
data_folder = "/notebooks/experiment/{}_data/{}_data/".format(rule, rule)
save_path = "/notebooks/modelfits/reward_preds/{}/simple_grammar_constrained_preds/".format(rule)
# participant_data = listdir(data_folder)
participant_data = np.array([pos_json for pos_json in listdir(data_folder) if pos_json.endswith('.json')])
np.random.shuffle(participant_data)
existing_model_files = listdir(save_path)

for i, participant_id in enumerate(participant_data):
    #print("Participant number: ", i)
    participant_identifier = participant_id[:-5] # removes json suffix
    filename = participant_identifier + "_rewards.csv"
    print(participant_id)
    if filename in existing_model_files:
        print("already fitted -- move on to next participant")

        continue
    else:
        depth = 1
        grammar = KernelGrammar(basis_kernels = Kernels.get_basic_kernels(), complexity_penalty = 0.7, ignore_warnings=True, depth=1)
        episodic_dict = EpisodicDictionary(num_features=2)
        value_function = ChoiceModel.UCB

        num_iters = 200
        if rule == 'add':
            model = SimpleGrammarModelConstrained(grammar, episodic_dict, value_function, choice_function=None, training_iters=num_iters)
        else:
            model = SimpleGrammarModelConstrainedChangePoint(grammar, episodic_dict, value_function, choice_function=None, training_iters=num_iters)
        
        #%%codecell


        #%%codecell
        params = {}  # this is useless, we can clean up this code later.
        participant_fitter = ParticipantFit(participant_id, model, params, path=data_folder, save_path = save_path, rule=rule)
        participant_fitter.fit()

        participant_fitter.to_csv()  # then we save the csv files
