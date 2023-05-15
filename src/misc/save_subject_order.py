import torch
import numpy as np
from utils import load_data

rule = 'add'
ref_model = 'mean_tracker'
trials = np.arange(15)
path = '/notebooks/modelfits/rl3_to_participants'

if rule == 'changepoint':
    rl3_data, load_rewards, _ = torch.load(f'{path}/rl3_{rule}_curriculum.pth')
    regrets, ref_rewards, _ = torch.load(f'{path}/baselines_regrets_{rule}_curriculum.pth') 
else:
    rl3_data, load_rewards = torch.load(f'{path}/rl3_{rule}_curriculum.pth')
    regrets, ref_rewards = torch.load(f'{path}/baselines_regrets_{rule}_curriculum.pth') 
    
# sort rl3
rewards = load_rewards[:, :, trials].mean(1).sum(1) 
args = np.argsort(rewards)
# sort ref rewards and use those indices to sort baselines
args2 = np.argsort(ref_rewards[:, :, trials].mean(1).sum(1))

np.save(f'{path}/rl3_subject_order_{rule}', args)
np.save(f'{path}/baselines_subject_order_{rule}',args2)