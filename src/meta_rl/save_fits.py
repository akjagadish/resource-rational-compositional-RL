import torch
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--full', action='store_true', default=False, help='compute for all subtasks')
parser.add_argument('--changepoint', action='store_true', default=False, help='compute for changepoint')
parser.add_argument('--entropy', action='store_true', default=False, help='compute for entropy')
parser.add_argument('--unconstrained', action='store_true', default=False, help='no KL term')

args = parser.parse_args()

num_parameters = 3

if args.changepoint:
    num_participants = 109
    if args.full:
        num_trials = 20 * 15
        full_path = '_full=True_changepoint=True.pth'
        save_path = 'data/RL3Fits/model_fits/bic_rl3_entropy_changepoint_full.pth'
        save_path_mlls = 'data/RL3Fits/model_fits/mlls_rl3_entropy_changepoint_full.pth'
        save_path_pertrial_mlls = 'data/RL3Fits/model_fits/pertrials_mlls_rl3_entropy_changepoint_full.pth'
        num_trials_per_task = 15
    else:
        num_trials = 20 * 5
        full_path = '_full=False_changepoint=True.pth'
        save_path = 'data/RL3Fits/model_fits/bic_rl3_entropy_changepoint_last.pth'
        save_path_mlls = 'data/RL3Fits/model_fits/mlls_rl3_entropy_changepoint_last.pth'
        save_path_pertrial_mlls = 'data/RL3Fits/model_fits/pertrials_mlls_rl3_entropy_changepoint_last.pth'
        num_trials_per_task = 5
else:
    num_participants = 92
    if args.full:
        num_trials = 20 * 15
        full_path = '_full=True_changepoint=False.pth'
        save_path = 'data/RL3Fits/model_fits/bic_rl3_entropy_additive_full.pth'
        save_path_mlls = 'data/RL3Fits/model_fits/mlls_rl3_entropy_additive_full.pth'
        save_path_pertrial_mlls = 'data/RL3Fits/model_fits/pertrials_mlls_rl3_entropy_add_full.pth'
        num_trials_per_task = 15
    else:
        num_trials = 20 * 5
        full_path = '_full=False_changepoint=False.pth'
        save_path = 'data/RL3Fits/model_fits/bic_rl3_entropy_additive_last.pth'
        save_path_mlls = 'data/RL3Fits/model_fits/mlls_rl3_entropy_additive_last.pth'
        save_path_pertrial_mlls = 'data/RL3Fits/model_fits/pertrials_mlls_rl3_entropy_add_last.pth'
        num_trials_per_task = 5

likelihoods = torch.stack([torch.load('data/RL3Fits/grid_parameters/jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path) for i in range(num_participants)])
likelihoods = likelihoods.reshape(num_participants, -1)
a, b = likelihoods.max(-1)
bic_rl3 = (a.numpy() - (0.5 * num_parameters * math.log(num_trials))) * -2

print(bic_rl3)
print(bic_rl3.mean())
torch.save(bic_rl3, save_path)

if args.unconstrained:
    likelihoods = likelihoods[:, -1] 
    save_path = save_path.replace('bic_rl3_entropy', 'bic_rl2_entropy')
    save_path_mlls = save_path_mlls.replace('mlls_rl3_entropy', 'mlls_rl2_entropy')
    save_path_pertrial_mlls = save_path_pertrial_mlls.replace('mlls_rl3_entropy', 'mlls_rl2_entropy')
    
    # compute mlls
    N_dl = 1
    N_greedy = 51
    N_stick = 51
    MLLs = torch.stack([torch.logsumexp(torch.load('data/RL3Fits/grid_parameters/jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)[-1].squeeze(), dim=[0, 1]) - torch.log(torch.tensor(N_dl*N_greedy*N_stick)) for i in range(num_participants)])
    torch.save(MLLs, save_path_mlls)

    pertrial_MLLs = torch.stack([torch.stack([torch.logsumexp(torch.load('data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)[trial, -1].squeeze(), dim=[0, 1]) - torch.log(torch.tensor(N_dl*N_greedy*N_stick)) for trial in range(num_trials_per_task)]) for i in range(num_participants)])
    torch.save(pertrial_MLLs, save_path_pertrial_mlls)
else:
    print(likelihoods.shape)
    likelihoods_dls = likelihoods.reshape(num_participants, 1000, -1)

    # save dls
    dls = np.stack([np.where(likelihoods_dls[subj].max()==likelihoods_dls[subj])[0][0] for subj in range(num_participants)])
    torch.save(dls, 'data/RL3Fits/model_fits/dls_rl3' + full_path)

    # compute mlls
    N_dl = 1000
    N_greedy = 51
    N_stick = 51
    MLLs = torch.stack([torch.logsumexp(torch.load('data/RL3Fits/grid_parameters/jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path), dim=[0, 1, 2]) - torch.log(torch.tensor(N_dl*N_greedy*N_stick)) for i in range(num_participants)])
    torch.save(MLLs, save_path_mlls)

    pertrial_MLLs = torch.stack([torch.stack([torch.logsumexp(torch.load('data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)[trial], dim=[0, 1, 2]) - torch.log(torch.tensor(N_dl*N_greedy*N_stick)) for trial in range(num_trials_per_task)]) for i in range(num_participants)])
    torch.save(pertrial_MLLs, save_path_pertrial_mlls)