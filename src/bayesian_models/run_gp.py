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
from ChoiceModel import GrammarModel, ChoiceModel, MeanTracker, RBFModelMemory
from KernelGrammar import KernelGrammar
from EpisodicDictionary import EpisodicDictionary
from Kernels import Kernels
#%%codecell

rule = 'changepoint'
data_folder = "/u/ajagadish/resource-rational-compositional-RL/data/raw_data/{}_data/".format(rule)
save_path = "/u/ajagadish/resource-rational-compositional-RL/src/model_fits/" #"/notebooks/modelfits/reward_preds/{}/simple_grammar_constrained_preds/".format(rule)
participant_data = np.array([pos_json for pos_json in listdir(data_folder) if pos_json.endswith('.json')])
np.random.shuffle(participant_data)
existing_model_files = listdir(save_path)

for i, participant_id in enumerate(participant_data):
    print("Participant number: ", i)
    participant_identifier = participant_id[:-5] # removes json suffix
    filename = participant_identifier + "_rewards.csv"
    if filename in existing_model_files:
        print("already fitted -- move on to next participant")
        continue
    else:

        num_iters = 50
        sees_context_features = False  # set this to False to get the RBF no context model
        model = RBFModelMemory(sees_context_features, num_iters)

        params = {}
        participant_fitter = ParticipantFit(participant_id, model, params,  path=data_folder, savefolder = "rbf_nocontext_preds")

        participant_fitter.create_full_dataset()
        participant_fitter.fit_all_tasks() # here we use the fit_all_tasks method instead

        participant_fitter.to_csv()  # then we save the csv files
