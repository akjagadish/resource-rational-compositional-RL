from .MultiArmedBandit import MultiArmedBandit
from .BlockMABs import BlockMABs
import torch
import numpy as np


# torch.manual_seed(seed_val)
# np.random.seed(seed_val)
# torch.cuda.manual_seed(seed_val)

def sample_data(n_rounds, block, rule, seed_val=0, composition_block=True, noise_per_arm=True, noise_var=0.1, normalize_rewards=True):
    
    start_arm = 0
    end_arm = 5
    n_arms = (end_arm - start_arm) + 1
    n_rounds_per_kernel = 1 # per function

    CUES =  {'linear': [0, 0],  'periodic': [0, 0], 'linperiodic': []}
    ctx_dim = len(CUES['linear'])
    n_blocks = 1 
    bandit = MultiArmedBandit(seed=seed_val, cues=CUES, start_arm=start_arm, end_arm=end_arm,
                            normalize=False, noise_per_arm=noise_per_arm, noise_var=noise_var,
                            to_list=True)
    task = BlockMABs(bandit, seed=seed_val, composition_block=composition_block, num_rounds=n_rounds_per_kernel, rule=rule, 
                    normalize=normalize_rewards, evaluate=True)

    X, Y, S, P = [], [], [], []
    for _ in range(n_rounds):
        x, y, s, p = task.sample(end_rnd=1, block=block, apply_rule=rule)
    #     x, y, s, _ = task.prepare_data(x, y, s, block, n_trials)
        X.append(x)
        Y.append(y)
        S.append(s)
        P.append(p)

    X = torch.stack(X).squeeze(1)
    Y = torch.stack(Y).squeeze(1)
    S = torch.stack(S).squeeze(1)
    
    return X, Y, S

def sample_rewards(num_batch, block, rule, seed_val=0, curriculum=True, noise_per_arm=True, noise_var=0.1, normalize_rewards=True):
    
    num_arms = 6
    cues =  {'linear': [0, 0],  'periodic': [0, 0], 'linperiodic': []}
    ctx_dim = len(cues['linear'])
    bandit = MultiArmedBandit(seed=seed_val, cues=cues, num_arms=num_arms, ctx_dim=ctx_dim,
                              normalize=False, noise_per_arm=noise_per_arm, noise_var=noise_var)
    task = BlockMABs(bandit=bandit, seed=seed_val, curriculum=curriculum, num_rounds=1, rule=rule, 
                     normalize=normalize_rewards, evaluate=True)


    X, Y, S, P = [], [], [], []
    for _ in range(num_batch):
        x, y, s, p = task.sample(end_rnd=1, block=block, apply_rule=rule)
        # x, y, s, _ = task.prepare_data(x, y, s, block, n_trials)
        X.append(x)
        Y.append(y)
        S.append(s)
        P.append(p)

    X = torch.stack(X).squeeze(1)
    Y = torch.stack(Y).squeeze(1)
    S = torch.stack(S).squeeze(1)
    
    return X, Y, S