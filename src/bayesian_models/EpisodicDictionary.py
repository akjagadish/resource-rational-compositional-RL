import numpy as np
from scipy import spatial
import torch
import numpy as np
from matplotlib import pyplot as plt
import gpytorch
import random
from Kernels import Kernels #from models.grammar
import math


# Neural dictionary class for storing past observations alongside contexts, posteriors over kernels. Includes methods
# for appending new memories, computing similarity between contexts, and informing priors based on past experiences.


class EpisodicDictionary():
    def __init__(self, num_features):

        self.context_dict = {}
        self.num_features = num_features


    def reset(self):
        self.context_dict = {}

    def new_entry(self, key, features, GP):
        # Key is a string identifying the context.
        # The key should be 'TaskNumber-FunctionNumber'
        self.context_dict[key] = {}
        self.context_dict[key]["features"] = features  # context features
        self.context_dict[key]["GP"] = GP # there are no GP models at this stage
        self.context_dict[key]["kernel"] = None # nor are there kernels
        self.current_context_key = key

    def append(self, key, X, Y, GP, kernel):

        self.context_dict[key]["X"] = X
        self.context_dict[key]["Y"] = Y
        self.context_dict[key]["GP"] = GP
        self.context_dict[key]["kernel"] = kernel


    def get_context_features(self):
        num_contexts = len(self.context_dict)

        features = np.zeros((num_contexts, self.num_features))
        for i, ctx in enumerate(self.context_dict):
            features[i] = self.context_dict[ctx]["features"]

        return features

    def get_context_similarities(self, current_context):
        self.num_contexts = len(self.context_dict)

        np.zeros(self.num_contexts)

        for i, ctx in enumerate(self.context_dict):
            features = self.context_dict[ctx]["features"]
            sim = self.cosine_similarity(current_context, features)
            similarities.append(sim)

        return similarities


    def cosine_similarity(self, vec_1, vec_2):
        sim = 1 - spatial.distance.cosine(vec_1, vec_2)
        return sim



    def function_transfer(self, current_context):


        gps = []
        similarities = []
        for i, ctx in enumerate(self.context_dict):
            if self.current_context_key == ctx:
                continue
            gp = self.context_dict[ctx]["GP"]
            features = self.context_dict[ctx]["features"]
            sim = self.cosine_similarity(current_context, features)
            similarities.append(sim)
            gps.append(gp)

        similarities = np.array(similarities)
        if similarities.sum() < 0.0001:
            return None
        return gps, similarities


    def structure_transfer(self, current_context, kernel_kernel):
        ## TODO:
        return NotImplementedError
