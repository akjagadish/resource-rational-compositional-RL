import torch
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from .MultiArmedBandit import MultiArmedBandit
from .utils import set_random_seed

class BlockMABs():
    def __init__(self, bandit, seed, num_rounds=1, curriculum=True, cues=None, rule=['add'], normalize=True, linear_first=True, evaluate=False):
        """    
        Generate data for compositional curriculum.

        Args:
            bandit ([task]): base task typically, MultiArmedBandit task
            num_blocks ([type]): 
            num_rounds (int, optional): [description]. Defaults to 10.
            curriculum (bool, optional): [description]. Defaults to True.
            cues ([type], optional): [description]. Defaults to None.
            rule (str, optional): [description]. Defaults to 'add'.
            normalize (bool, optional): [description]. Defaults to True.
        """
          
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.curriculum = curriculum
        self.cues = cues if cues else bandit.cues
        self.cues = self.cues if curriculum else {'linperiodic':[]}
        self.num_cues = len(self.cues)
        self.num_rounds = num_rounds
        self.normalize = normalize
        self.eval_control = False
        self.evaluate = evaluate
        self.return_composition_only = False
        self.rule = rule
        self.ctx_dim = bandit.ctx_dim
        self.linear_first = linear_first
        # set the manual seed 
        set_random_seed(seed)
        self.bandit.evaluate = True
        
    def sample(self, start_rnd=0, end_rnd=None, block=['linear'], apply_rule=None, LOO=False):

        end_rnd = end_rnd if end_rnd else start_rnd+self.num_rounds 
        apply_rule = np.array(apply_rule).reshape(-1) if apply_rule else self.rule
        self.return_composition_only = True if LOO else self.return_composition_only

        X, Y, S, K, P = [], [], [], [], []
        for _ in np.arange(start_rnd, end_rnd):

            for kernel_indx, kernel in enumerate(block):
                used_kernels = []
                params = {}
                if self.curriculum or self.eval_control or LOO:
                    
                    assert len(block)>1, "when composition is to be performed, block size should be greater than 1"
                    if kernel == 'linear':
                        use_kernel = np.random.choice(['linpos', 'linneg'])
                    elif kernel == 'periodic':
                        use_kernel = np.random.choice(['perodd', 'pereven'])
                    else:
                        use_kernel = kernel

                    if kernel == 'linperiodic':
                        x, y = self.composed_x, self.composed_y
                    else:
                        assert use_kernel in ['linpos', 'linneg', 'pereven', 'perodd'], 'its not one of the base structures'
                        x, y, p = self.bandit._sample_one_subtask(cue=use_kernel)    
                        params = {use_kernel: p}

                    if kernel_indx<(len(block)-1):
                        rule = np.random.choice(apply_rule) if len(apply_rule)>1 else apply_rule[0]
                        _ = self.do_composition(x, y, kernel_indx, rule=rule)
                    
                    # build a list of kernels used
                    used_kernels.append(use_kernel)
                else:

                    assert len(block)==1, "when no composition is to be performed, block should be of size 1"
                    
                    if kernel == 'linperiodic':
                        rule = np.random.choice(apply_rule) if len(apply_rule)>1 else apply_rule[0]
                        # linear
                        kernel_lin = np.random.choice(['linpos', 'linneg'])
                        xlin, ylin, plin = self.bandit._sample_one_subtask(cue=kernel_lin)
                        _ = self.do_composition(xlin, ylin, 0, rule=rule)
                        used_kernels.append(kernel_lin)
                        params = {kernel_lin: plin}
                        # periodic
                        kernel_per = np.random.choice(['perodd', 'pereven'])
                        xper, yper, pper = self.bandit._sample_one_subtask(cue=kernel_per)
                        params = {kernel_per: pper}
                        # linear * rule * periodic
                        _ = self.do_composition(xper, yper, 1, rule=rule)
                        x, y = self.composed_x, self.composed_y
                        used_kernels.append(kernel_per)                                    
                

                y = self.norm(y) if self.normalize else y

                X.append(x)
                Y.append(y)
                S.append(torch.tensor(kernel_indx).reshape(-1))
                P.append(params)
            K.append(used_kernels)

        if self.return_composition_only:
            X, Y, S = [], [], []
            X.append(self.composed_x)
            Y.append(self.norm(self.composed_y))
            S.append(torch.tensor(0).reshape(-1))

        Y = torch.stack(Y)
        X = torch.stack(X)
        S = torch.stack(S)
        K = np.vstack(K)
        #P = np.vstack(P)

        if self.evaluate:
            return X, Y, S, P
        else:
            return X, Y, S #, K

    def evaluate_control(self):
        self.eval_control = True
        self.return_composition_only = True

    def evaluate_composition(self):
        self.eval_control = False
        self.return_composition_only = False
                
    def do_composition(self, x, y, kernel_idx, rule='add'):

        if kernel_idx == 0:
            self.composed_x = x
            self.composed_y = y
        else:
            if rule == 'add' or rule == 'sum':
                self.composed_y =  self.composed_y + y
                self.composed_y = self.composed_y #/2
                self.composed_x = self.composed_x + x  # 
                if len(self.rule)>=2:
                    self.composed_x = torch.tensor([0., 1.]) # arbitary set
               
            elif rule == 'chngpnt' or rule == 'changepoint':
                if self.linear_first:
                    self.composed_y = torch.cat((self.composed_y[:int(self.num_arms/2)].type(torch.FloatTensor), y[int(self.num_arms/2):].type(torch.FloatTensor)))
                else:
                    self.composed_y = torch.cat((y[:int(self.num_arms/2)].type(torch.FloatTensor), self.composed_y[int(self.num_arms/2):].type(torch.FloatTensor)))
                self.composed_x = self.composed_x + x  # 
                if len(self.rule)>=2:
                    self.composed_x = torch.tensor([1., 0.])

        return self.composed_x, self.composed_y

    @staticmethod
    def norm(y):    
        y = y/2. 
        return y

    @staticmethod
    def prepare_data(X, Y, S, block, n_trials):
        X = np.repeat(X, n_trials, axis=0)
        Y = np.repeat(Y, n_trials, axis=0)
        S = np.repeat(S, n_trials, axis=0)
        cues = np.repeat(block, n_trials)
        return X, Y, S, cues

def compositionalcurriculum_loader(seed, num_arms=6, n_rounds_per_kernel=1, curriculum=False, ctx_dim=None,   
                                   normalize_rewards=True, noise_per_arm=None, noise_var=0.1, cues=None, 
                                   linear_first=True, rule=['add']):

    bandit = MultiArmedBandit(seed=seed, cues=cues, num_arms=num_arms, ctx_dim=ctx_dim,
                              normalize=False, noise_per_arm=noise_per_arm, noise_var=noise_var)
    task = BlockMABs(bandit=bandit, seed=seed, curriculum=curriculum, num_rounds=n_rounds_per_kernel, rule=rule, 
                     normalize=normalize_rewards, linear_first=linear_first)
            
    return task