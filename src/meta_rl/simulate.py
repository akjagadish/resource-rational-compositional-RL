from tqdm import tqdm
import torch
import argparse
import gym
import envs.bandits
import numpy as np
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import seaborn as sns

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--num-episodes', type=int, default=100, help='number of episodes')
parser.add_argument('--changepoint', action='store_true', default=False, help='compute for changepoint')
parser.add_argument('--prior', default='gaussian', help='type of prior')
parser.add_argument('--unconstrained', action='store_true', default=False, help='no KL term')
parser.add_argument('--policy', default='MLE', help='policy parameter')
parser.add_argument('--entropy', action='store_true', default=False, help='entropy term')
parser.add_argument('--full', action='store_true', default=False, help='full model')
parser.add_argument('--env-name', default='jagadish2022curriculum-v0', help='name of the environment')
parser.add_argument('--per-trial',type=int, default=None, help='which trial')
    
args = parser.parse_args()
policy = args.policy
env_name= args.env_name
env = gym.make(env_name)
env.batch_size = 1
env.device = 'cpu'
model_samples = 10

epsilons1 = np.linspace(0, 1, 51)
epsilons2 = np.linspace(0, 1, 51)

if args.changepoint:
    num_participants = 109
else:
    num_participants = 92

task_type = torch.zeros(num_participants, args.num_episodes)
actions = torch.zeros(num_participants, args.num_episodes, 15)
regrets = torch.zeros(num_participants, args.num_episodes, 15)
rewards = torch.zeros(num_participants, args.num_episodes, 15)
subtask_rewards = torch.zeros(num_participants, args.num_episodes, 6, 15)
contexts = [] #torch.zeros(num_models, args.num_episodes, 15)
true_best_action = torch.zeros(num_participants, args.num_episodes, 15)

for i in tqdm(range(num_participants)):

    if args.per_trial is not None:
        likelihoods = torch.load('/eris/scratch/ajagadish/RL3NeurIPS/data/RL3Fits/grid_parameters/noninf_pertrial_jagadish_subject=' + str(i) +  '_prior=svdo' + '_entropy' + str(args.entropy) + '_full=' + str(args.full) + '_changepoint=' + str(args.changepoint) + '.pth')[args.per_trial]
    else:
        likelihoods = torch.load('data/RL3Fits/grid_parameters/jagadish_subject=' + str(i) + '_prior=' + args.prior + '_entropy' + str(args.entropy) + '_full=' + str(args.full) + '_changepoint=' + str(args.changepoint) + '.pth').numpy() 
    
    if args.unconstrained:
        eps1, eps2 = np.where(likelihoods[-1]==likelihoods[-1].max())
        dl=[999]
    else:
        dl, eps1, eps2 = np.where(likelihoods==likelihoods.max())
        #print(dl)
    eps1 = epsilons1[eps1][0] if args.policy == 'MLE' else None #0.24
    eps2 = epsilons2[eps2][0] if args.policy == 'MLE' else None #0.26
    print(eps1, eps2)
    dl = float((dl[0] + 1)*10)
    print(dl)
    print(eps1)
    print(eps2)

    file_name = 'trained_models/env='+ env_name + '_prior=' + args.prior + '_entropy' + str(args.entropy) + '_constraint=' + str(dl) + 'algo=a2c_run=0.pt'
    _, _, agent = torch.load(file_name, map_location='cpu')
    context_dl = []
    for t in range(args.num_episodes):
        # reset env
        done = False
        if args.changepoint:
            obs = env.reset(rule='changepoint')
        else:
            obs = env.reset(rule='add') 

        if env.w >= 0 and env.phase == 0: # posodd
            task_type[i, t] = 0
        if env.w >= 0 and env.phase == 1: # poseven
            task_type[i, t] = 1
        if env.w < 0 and env.phase == 0: # negodd
            task_type[i, t] = 2
        if env.w < 0 and env.phase == 1: # negeven
            task_type[i, t] = 3

        # reset model
        hx = [agent.initial_states(env.batch_size) for k in range(model_samples)]
        zeta = [agent.get_zeta(env.batch_size) for k in range(model_samples)]
        while not done:
            policy_sampling = torch.zeros(env.batch_size, 6)
            for sample in range(model_samples):
                policy, _, hx[sample], _ = agent.act(obs, hx[sample], zeta[sample])
                policy_sampling += policy.probs

            policy_sampling = policy_sampling / model_samples

            last_action = F.one_hot(action, num_classes=6).float() if env.t > 0 else (1/6.0) * torch.ones(1, 6)
            policy_sampling = (1 - eps1 - eps2) * policy_sampling + eps1 * (1/6.0) * torch.ones(1, 6) + eps2 * last_action

            action = Categorical(policy_sampling).sample()

            obs, reward, done, info = env.step(action)
            actions[i, t, env.t-1] = action.item()
            sub_task_idx = 0 if env.t<=5 else 1 if (env.t>5 and env.t<=10) else 2 
            subtask_rewards[i, t, :, env.t-1] = env.mean_reward[0, sub_task_idx]
            rewards[i, t, env.t-1] = env.mean_reward[0, sub_task_idx, action.item()]
            regrets[i, t, env.t-1] = env.mean_reward[0, sub_task_idx].max() - reward
            true_best_action[i, t, env.t-1] = torch.argmax(env.mean_reward[0, sub_task_idx, :])
    # storee contexts
    contexts.append(context_dl)

# fig, axs = plt.subplots(1, 4)
# policies = torch.zeros(4, 6)
# for k, i in enumerate([2, 3, 0, 1]):
#     for participant_index in range(num_participants):
#         current_task_type = task_type[participant_index, :] == i
#         policies[k] += F.one_hot(actions[participant_index, :, 10].long()).float()[current_task_type].mean(0)

#     policies[k] = policies[k] / num_participants
#     axs[k].bar(np.arange(6), policies[k])
# sns.despine()
# plt.savefig('figures/action_probs.pdf')
# plt.show()

# torch.save(policies, 'data/temp/action_probs_changepoint=' + str(args.changepoint) + '.pth')
torch.save([regrets, task_type], 'data/RL3Fits/fitted_simulations/regrets_recovery_changepoint=' + str(args.changepoint) + '_full=' + str(args.full) + '_entropy' + str(args.entropy) + env_name + '_pertrial' + str(args.per_trial) +'.pth')
torch.save([actions, task_type], 'data/RL3Fits/fitted_simulations/pickedactions_recovery_changepoint=' + str(args.changepoint) + '_full=' + str(args.full) + '_entropy' + str(args.entropy) + env_name + '_pertrial' + str(args.per_trial) +'.pth')
torch.save([actions, rewards, contexts, regrets, true_best_action], 'data/RL3Fits/fitted_simulations/all_stats_changepoint=' + str(args.changepoint) + '_full=' + str(args.full) + '_entropy' + str(args.entropy) + env_name + '_pertrial' + str(args.per_trial) +'.pth')
torch.save([subtask_rewards], 'data/RL3Fits/fitted_simulations/subtask_rewards_changepoint=' + str(args.changepoint) + '_full=' + str(args.full) + '_entropy' + str(args.entropy) + env_name + '_pertrial' + str(args.per_trial) +'.pth')

