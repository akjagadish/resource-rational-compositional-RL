import torch
import numpy as np
import pdb
from sklearn.metrics.pairwise import linear_kernel
from .utils import set_random_seed
 
class MultiArmedBandit():
    def __init__(self, seed=0, cues=None, num_arms=6, ctx_dim=None,
                normalize=True, noise_per_arm=False, noise_var=0.1, 
                cue_per_epoch=False, evaluate=False, to_list=False):
        """ Genereate  data for MultiArmedBandit Task.
        
        Args:
            seed (int): seed for the task. Defaults to 0.
            cues ([dict.]): functions and harded binary cues for each. Defaults to None.
            num_arms (int): number of arms for the structured bandit.
            ctx_dim (int, optional): dimension of context. Defaults to 2.
            normalize (bool, optional): normalize samples. Defaults to True.
            noise_per_arm (bool, optional): add noise per arm. Defaults to False.
            cue_per_epoch (bool, optional): shuffle cues. Defaults to False.
        """
        default_cues = {'linear': [], 'periodic': [], 'linperiodic': []}
        self.cues =  default_cues if cues is None else cues
        self.num_cues  = len(self.cues)
        self.ctx_dim = ctx_dim if ctx_dim else len(list(self.cues.values())[0])
        self.num_arms = num_arms
        self.start_arm = 0
        self.end_arm = self.num_arms-1
        self.noise_per_arm = noise_per_arm
        self.noise_var = noise_var
        self.cue_per_epoch = cue_per_epoch
        self.x_dim = self.ctx_dim + 1
        self.y_dim = self.num_arms
        self.normalize=normalize
        self.evaluate = evaluate
        self.to_list = to_list
        set_random_seed(seed)  # set the manual seed 
        
    def sample(self, cue, n_runs, noise_var=None):
       
        # sample reward structures
        noise_var = noise_var if noise_var else self.noise_var
        X, Y = [], []
        for _ in range(n_runs):
            x, y = self._sample_one_subtask(cue=cue, noise_var=noise_var)
            X.append(x)
            Y.append(y)

        Y = torch.stack(Y) 
        X = torch.stack(X)

        return X, Y
            
    def _sample_one_subtask(self, cue='linear',freq=0.25, noise_var=None): 

        # get context
        if cue in self.cues:
            ctx = self.cues[cue]
        elif cue == 'linpos' or cue == 'linneg':
            ctx = self.cues['linear']
        elif cue == 'perodd' or cue == 'pereven':
            ctx = self.cues['periodic']
        else:
             raise ValueError("cue doesn't belong to base structures")
           
        # arms to generate rewards 
        x =  torch.linspace(self.start_arm, self.end_arm, self.num_arms)#/self.num_arms*3

        # noise
        noise_var = noise_var if noise_var else self.noise_var
        if self.noise_per_arm:
            noise = torch.randn((self.num_arms,)) * np.sqrt(noise_var)
        else:
            noise = torch.randn(1) * np.sqrt(noise_var)

        # generate rewards
        if cue == 'periodic':
            phase = np.random.sample([0., 1.])
            #freq=torch.rand(1)
            #phase=torch.randn(1)
            y, params = self._visualize_periodic_structure(x, freq=freq, phase=phase) 
            y = y + noise
        
        elif cue == 'pereven':
            y, params = self._get_periodic_structure(x, freq=freq, phase=1.) 
            y = y + noise
        
        elif cue == 'perodd':
            y, params = self._get_periodic_structure(x, freq=freq, phase=0.) 
            y = y + noise
        
        elif cue == 'linear' :

            slope = np.random.randint([])
            y, params = self._get_linear_structure(slope='pos') 
            y = y + noise

        
        elif cue == 'linpos':

            y, params = self._get_linear_structure(slope='pos') 
            y = y + noise

        elif cue == 'linneg':

            y, params = self._get_linear_structure(slope='neg') 
            y = y + noise
            
        context = torch.as_tensor(ctx).type(torch.FloatTensor)
        y = y/2. if self.normalize else y 
        reward = y.type(torch.FloatTensor)
        params.append(reward.max()-reward.min())
        params = torch.tensor(params).tolist() if self.to_list else params

        if self.evaluate:
            return context, reward, params
        else:
            return context, reward
    
    def _get_periodic_structure(self, x, freq, phase, amp=None):
        amp = amp if amp else torch.empty(1).uniform_(0., 4) 
        y =  amp*torch.abs(torch.sin((x-phase) * (2*np.pi*freq)))
        b = -amp/2 # 1.75
        y = y + b
        params = [amp, freq, phase] #{'amp': amp, 'freq': freq, 'phase': phase}
        return y, params
    
    def _visualize_periodic_structure(self, x, freq, phase, amp=None):
        amp = amp if amp else torch.empty(1).uniform_(0., 4) 
        y =  amp*torch.sin((x-phase) * (2*np.pi*freq))
        b = -amp/2 # 1.75
        y = y + b
        params = [amp, freq, phase] #{'amp': amp, 'freq': freq, 'phase': phase}
        return y, params

    def _get_linear_structure(self, slope='pos'):
        x = torch.linspace(-1., 1., self.num_arms)
        w = torch.empty(1).uniform_(0., 2.) if slope == 'pos' else torch.empty(1).uniform_(-2., -0.) 
        b = torch.empty(1).uniform_(-1., 1.)
        params = [w, b] #{'w': w, 'b':b}
        y = w * x + b
        return y, params


def multiarmedbandit_loader(seed, start_arm=0, end_arm=7, n_subtasks=1, ctx_dim=None, normalize_rewards=True, noise_per_arm=None, 
                            noise_var=0.1, cues=None):
    
    task = MultiArmedBandit(seed=seed, cues=cues, start_arm=start_arm, end_arm=end_arm, ctx_dim=ctx_dim,
                           normalize=normalize_rewards, noise_per_arm=noise_per_arm, noise_var=noise_var)
            
    return task
