from tqdm import tqdm
import torch
import argparse
import gym
import envs.bandits
from torch.distributions import Bernoulli, Categorical
import numpy as np
import math
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
parser.add_argument('--full', action='store_true', default=False, help='compute for all subtasks')
parser.add_argument('--changepoint', action='store_true', default=False, help='compute for changepoint')
parser.add_argument('--prior', default='svdo', help='type of prior')
parser.add_argument('--subject', type=int, default=0, help='subject id')
parser.add_argument('--unconstrained', action='store_true', default=False, help='no KL term')
parser.add_argument('--entropy-loss', action='store_true', default=True, help='entropy loss')
args = parser.parse_args()
if args.recompute:
    betas = ['unconstrained'] if args.unconstrained else np.linspace(10, 10000, 1000)#[:10]
    model_samples = 10

    with torch.no_grad():
        if args.changepoint:
            participant_actions, participant_rewards, linear_first = torch.load('data/changepoint_curriculum.pth')
            linear_first = linear_first[args.subject]
        else:
            participant_actions, participant_rewards = torch.load('data/additive_curriculum.pth')
        participant_actions = participant_actions[args.subject]
        participant_rewards = participant_rewards[args.subject]

        num_tasks = participant_actions.shape[0] # 20
        num_trials = participant_actions.shape[1] # 15

        likelihood_human_sampling = torch.zeros(len(betas), num_tasks, num_trials, 1, 6)
        for num_beta, beta in enumerate(tqdm(betas)):
            if args.changepoint:
                if linear_first:
                    file_name = 'trained_models/env=jagadish2022curriculum-v0_prior=' + args.prior + '_entropy' + str(args.entropy_loss) + '_constraint=' + str(beta) + 'algo=a2c_run=0.pt'
                else:
                    file_name = 'trained_models/env=jagadish2022curriculum-v1_prior=' + args.prior + '_entropy' + str(args.entropy_loss) + '_constraint=' + str(beta) + 'algo=a2c_run=0.pt'
            else:
                file_name = 'trained_models/env=jagadish2022curriculum-v0_prior=' + args.prior + '_entropy' + str(args.entropy_loss) + '_constraint=' + str(beta) + 'algo=a2c_run=0.pt'
            _, _, agent = torch.load(file_name, map_location='cpu')

            for task in range(num_tasks):
                obs = torch.zeros(1, 12)
                obs[0, 2] = 1 # cue for subtask ID
                hx = [agent.initial_states(1) for k in range(model_samples)]
                zeta = [agent.get_zeta(1) for k in range(model_samples)]

                subtask_index = 0
                subtask_step = 0

                for trial in range(num_trials):
                    policy_sampling = torch.zeros(1, 6)
                    for sample in range(model_samples):
                        policy, _, hx[sample], _ = agent.act(obs, hx[sample], zeta[sample])
                        if trial == 0:
                            policy_sampling +=  (1/6.0) * torch.ones(1, 6)
                        else:
                            policy_sampling += policy.probs
                    likelihood_human_sampling[num_beta, task, trial] = policy_sampling / model_samples

                    # update subtask step and index
                    subtask_step += 1
                    if subtask_step >=5:
                        subtask_step = 0
                        subtask_index += 1

                    # update observation
                    if subtask_index == 1:
                        obs[0, 2] = 0 # cue for subtask index
                        obs[0, 3] = 1 # cue for subtask index
                    if subtask_index == 2:
                        if args.changepoint:
                            obs[0, 1] = 1 # cue for changepoint rule
                        else:
                            obs[0, 0] = 1 # cue for addtive rule
                        obs[0, 2] = 1 # cue for subtask index
                    obs[0, 4:10] = F.one_hot((torch.ones([]) * participant_actions[task, trial]).long(), num_classes=6)
                    obs[0, 10] = (participant_rewards[task, trial]/8.0) - 0.6
                    obs[0, 11] = trial + 1
        # add unconstrained
        torch.save([likelihood_human_sampling, participant_actions], 'data/RL3Fits/jagadish_likelihoods_subject=' + str(args.subject) + '_prior=' + args.prior  + '_changepoint=' + str(args.changepoint) + '_entropy' + str(args.entropy_loss) + '.pth')
else:
    betas =  ['unconstrained'] if args.unconstrained else np.linspace(1, 1000, 1000)
    epsilons1 = np.linspace(0, 1, 51)
    epsilons2 = np.linspace(0, 1, 51)
    probs, _ = torch.load('data/RL3Fits/jagadish_likelihoods_subject=' + str(args.subject) +  '_prior=' + args.prior + '_changepoint=' + str(args.changepoint) + '_entropy' + str(args.entropy_loss) + '.pth')

    if args.changepoint:
        participant_actions, participant_rewards, _ = torch.load('data/changepoint_curriculum.pth')
    else:
        participant_actions, participant_rewards = torch.load('data/additive_curriculum.pth')

    participant_actions = participant_actions[args.subject]
    participant_rewards = participant_rewards[args.subject]

    num_tasks = participant_actions.shape[0] # 20
    num_trials = participant_actions.shape[1] # 15
    time_steps = range(15) if args.full else range(10, 15)
    likelihood_human_sampling = torch.zeros(len(betas), len(epsilons1), len(epsilons1))
    per_trial_likelihood_human_sampling = torch.zeros(len(time_steps), len(betas), len(epsilons1), len(epsilons1))
    for num_beta, beta in enumerate(tqdm(betas)):
        for num_epsilon1, epsilon1 in enumerate(epsilons1):
            for num_epsilon2, epsilon2 in enumerate(epsilons2):
                if epsilon1 + epsilon2 > 1:
                    likelihood_human_sampling[num_beta, num_epsilon1, num_epsilon2] = -math.inf
                    per_trial_likelihood_human_sampling[trial_id, num_beta, num_epsilon1, num_epsilon2] = -math.inf
                else:
                    for task in range(num_tasks):
                        for trial_id, trial in enumerate(time_steps):
                            last_action = F.one_hot(participant_actions[task, trial-1], num_classes=6).unsqueeze(0).float() if trial > 0 else (1/6.0) * torch.ones(1, 6)
                            policy_sampling = (1 - epsilon1 - epsilon2) * probs[num_beta, task, trial] + epsilon1 * (1/6.0) * torch.ones(1, 6) + epsilon2 * last_action
                            action = float(participant_actions[task, trial])
                            likelihood_human_sampling[num_beta, num_epsilon1, num_epsilon2] += Categorical(policy_sampling).log_prob(action * torch.ones(1)).squeeze()
                            per_trial_likelihood_human_sampling[trial_id, num_beta, num_epsilon1, num_epsilon2] += Categorical(policy_sampling).log_prob(action * torch.ones(1)).squeeze()

    torch.save(likelihood_human_sampling, 'data/RL3Fits/grid_parameters/jagadish_subject=' + str(args.subject) + '_prior=' + args.prior + '_entropy' + str(args.entropy_loss) + '_full=' + str(args.full) + '_changepoint=' + str(args.changepoint) + '.pth')
    torch.save(per_trial_likelihood_human_sampling, 'data/RL3Fits/grid_parameters/pertrial_jagadish_subject=' + str(args.subject) + '_prior=' + args.prior + '_entropy' + str(args.entropy_loss) + '_full=' + str(args.full) + '_changepoint=' + str(args.changepoint) + '.pth')

# had to convert zeros to math.inf for full False fits
# full_path='_full=False_changepoint=False.pth'
# for i in range(92):
#     per_trial = torch.load('../../RL3NeurIPS/data/RL3Fits/grid_parameters/pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)
#     per_trial = torch.tensor(np.where(per_trial==0, -math.inf, per_trial))
#     torch.save(per_trial, '../../RL3NeurIPS/data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)

# full_path='_full=True_changepoint=False.pth'
# for i in range(92):
#     per_trial = torch.load('../../RL3NeurIPS/data/RL3Fits/grid_parameters/pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)
#     per_trial = torch.tensor(np.where(per_trial==0, -math.inf, per_trial))
#     torch.save(per_trial, '../../RL3NeurIPS/data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)

# import numpy as np
# import math
# full_path='_full=True_changepoint=True.pth'
# for i in range(92):
#     per_trial = torch.load('../../RL3NeurIPS/data/RL3Fits/grid_parameters/pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)
#     per_trial = torch.tensor(np.where(per_trial==0, -math.inf, per_trial))
#     torch.save(per_trial, '../../RL3NeurIPS/data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)
# full_path='_full=False_changepoint=True.pth'
# for i in range(109):
#     per_trial = torch.load('../../RL3NeurIPS/data/RL3Fits/grid_parameters/pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)
#     per_trial = torch.tensor(np.where(per_trial==0, -math.inf, per_trial))
#     torch.save(per_trial, '../../RL3NeurIPS/data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropyTrue' + full_path)