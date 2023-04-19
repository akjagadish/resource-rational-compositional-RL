import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch

class CompositionalBandit(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps_per_subtask, num_actions, linear_first, curriculum, batch_size=1): # 1 for simulations
        self.num_actions = num_actions
        self.max_steps_per_subtask = max_steps_per_subtask
        self.batch_size = batch_size

        self.curriculum = curriculum
        self.linear_first = linear_first

        self.action_space = spaces.Discrete(self.num_actions )
        self.observation_space = spaces.Box(np.ones(self.num_actions + 6), np.ones(self.num_actions + 6))

    def reset(self, rule=None):
        # randomize settings
        if rule is None:
            rule = np.random.choice(['add', 'changepoint'])

        # keep track of time-steps
        self.t = 0
        self.subtask_step = 0
        self.subtask_index = 0

        self.rule = rule

        if self.curriculum:
            self.num_subtasks = 3
        else:
            self.num_subtasks = 1

        self.max_steps = self.max_steps_per_subtask * self.num_subtasks

        # generate reward functions
        mean_rewards_linear = self.sample_mean_reward_subtask(linear=True)
        mean_rewards_periodic = self.sample_mean_reward_subtask(linear=False)

        if rule == 'add':
            mean_reward_compositional = mean_rewards_linear + mean_rewards_periodic
        elif rule == 'changepoint':
            if self.linear_first:
                mean_reward_compositional = torch.cat((mean_rewards_linear[:, :int(self.num_actions/2)], mean_rewards_periodic[:, int(self.num_actions/2):]), dim=1)
            else:
                mean_reward_compositional = torch.cat((mean_rewards_periodic[:, :int(self.num_actions/2)], mean_rewards_linear[:, int(self.num_actions/2):]), dim=1)

        self.cue_rule = torch.zeros(self.batch_size, self.max_steps_per_subtask, self.num_subtasks, 2)
        if rule == 'add':
            self.cue_rule[:, :, -1, 0] = 1
        else:
            self.cue_rule[:, :, -1, 1] = 1

        self.cue_subtask = torch.zeros(self.batch_size, self.max_steps_per_subtask, self.num_subtasks, 2)
        if self.curriculum:
            self.cue_subtask[:, :, 0, 0] = 1
            self.cue_subtask[:, :, 1, 1] = 1
            self.cue_subtask[:, :, 2, :] = 1
            #if self.linear_first:
            self.mean_reward = torch.stack((mean_rewards_linear, mean_rewards_periodic, mean_reward_compositional), dim=1)
            #else:
            #    self.mean_reward = torch.stack((mean_rewards_periodic, mean_rewards_linear, mean_reward_compositional), dim=1)
        else:
            self.cue_subtask[:, :, 0, :] = 1
            self.mean_reward = mean_reward_compositional.unsqueeze(1)

        # add noise to mean reward (mean rewards is tensor of size batch_size X num_subtasks X num_arms)
        self.mean_reward = Normal(self.mean_reward, math.sqrt(0.1)).sample()
        self.rewards = Normal(self.mean_reward, math.sqrt(0.1)).sample((self.max_steps_per_subtask, )).transpose(0, 1)

        last_action = torch.zeros(self.batch_size, 6)
        last_reward = torch.zeros(self.batch_size)

        return self.get_observation(last_reward, last_action, self.t, self.cue_rule[:, 0, 0], self.cue_subtask[:, 0, 0])

    def get_observation(self, last_reward, last_action, time_step, cue_rule, cue_subtask):
        return torch.cat([
            cue_rule,
            cue_subtask,
            last_action,
            last_reward.unsqueeze(-1),
            torch.ones(self.batch_size, 1) * time_step],
            dim=1)

    def sample_mean_reward_subtask(self, linear=True):
        if linear:
            x = torch.linspace(-1., 1., self.num_actions).unsqueeze(0)
            self.w = torch.empty(self.batch_size, 1).uniform_(-2., 2.)
            b = torch.empty(self.batch_size, 1).uniform_(-1., 1.)
            y = self.w @ x + b
        else:
            x = torch.linspace(0, self.num_actions-1, self.num_actions)
            self.phase = torch.randint(0, 2, (self.batch_size, 1))
            freq = 0.25
            amp = torch.empty(self.batch_size, 1).uniform_(0., 4)
            y = amp*torch.abs(torch.sin((x-self.phase) * (2*np.pi*freq)))
            b = -amp/2
            y = y + b

        # dimensions: self.batch_size X self.num_actions
        return 0.5 * y

    def step(self, action):
        regrets = self.mean_reward[:, self.subtask_index].max(dim=1)[0] - self.mean_reward[:, self.subtask_index].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.rewards[:, self.subtask_step, self.subtask_index].gather(1, action.unsqueeze(1)).squeeze(1)

        self.t += 1
        self.subtask_step += 1
        if self.subtask_step >=5:
            self.subtask_step = 0
            self.subtask_index += 1
        done = True if (self.t >= self.max_steps) else False

        if not done:
            observation = self.get_observation(reward, F.one_hot(action.long(), num_classes=self.num_actions), self.t, self.cue_rule[:, self.subtask_step, self.subtask_index], self.cue_subtask[:, self.subtask_step, self.subtask_index])
        else:
            observation = torch.zeros(self.batch_size, self.num_actions + 6) # this is not used
        return observation, reward, done, {'regrets': regrets.mean()}

class CompositionalBanditAlternative(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps_per_subtask, num_actions, linear_first, curriculum, batch_size=32): # 1 for simulations
        self.num_actions = num_actions
        self.max_steps_per_subtask = max_steps_per_subtask
        self.batch_size = batch_size

        self.curriculum = curriculum
        self.linear_first = linear_first

        self.action_space = spaces.Discrete(self.num_actions )
        self.observation_space = spaces.Box(np.ones(self.num_actions + 6), np.ones(self.num_actions + 6))

    def reset(self, rule=None):
        # randomize settings
        if rule is None:
            rule = np.random.choice(['add', 'changepoint'])

        # keep track of time-steps
        self.t = 0
        self.subtask_step = 0
        self.subtask_index = 0

        self.rule = rule

        if self.curriculum:
            self.num_subtasks = 3
        else:
            self.num_subtasks = 1

        self.max_steps = self.max_steps_per_subtask * self.num_subtasks

        # generate reward functions
        mean_rewards_linear = self.sample_mean_reward_subtask(linear=True)
        mean_rewards_periodic = self.sample_mean_reward_subtask(linear=False)

        if rule == 'add':
            mean_reward_compositional = mean_rewards_linear + mean_rewards_periodic
        elif rule == 'changepoint':
            if self.linear_first:
                mean_reward_compositional = torch.cat((mean_rewards_linear[:, :int(self.num_actions/2)], mean_rewards_periodic[:, int(self.num_actions/2):]), dim=1)
            else:
                mean_reward_compositional = torch.cat((mean_rewards_periodic[:, :int(self.num_actions/2)], mean_rewards_linear[:, int(self.num_actions/2):]), dim=1)

        self.cue_rule = torch.zeros(self.batch_size, self.max_steps_per_subtask, self.num_subtasks, 2)
        if rule == 'add':
            self.cue_rule[:, :, -1, 0] = 1
        else:
            self.cue_rule[:, :, -1, 1] = 1

        self.cue_subtask = torch.zeros(self.batch_size, self.max_steps_per_subtask, self.num_subtasks, 2)
        if self.curriculum:
            self.cue_subtask[:, :, 0, 0] = 1
            self.cue_subtask[:, :, 1, 1] = 1
            self.cue_subtask[:, :, 2, :] = 1
            #if self.linear_first:
            self.mean_reward = torch.stack((mean_rewards_linear, mean_rewards_periodic, mean_reward_compositional), dim=1)
            #else:
            #    self.mean_reward = torch.stack((mean_rewards_periodic, mean_rewards_linear, mean_reward_compositional), dim=1)
        else:
            self.cue_subtask[:, :, 0, :] = 1
            self.mean_reward = mean_reward_compositional.unsqueeze(1)

        # add noise to mean reward (mean rewards is tensor of size batch_size X num_subtasks X num_arms)
        self.mean_reward = Normal(self.mean_reward, math.sqrt(0.2)).sample()
        self.rewards = Normal(self.mean_reward, math.sqrt(0.1)).sample((self.max_steps_per_subtask, )).transpose(0, 1)

        last_action = torch.zeros(self.batch_size, 6)
        last_reward = torch.zeros(self.batch_size)

        return self.get_observation(last_reward, last_action, self.t, self.cue_rule[:, 0, 0], self.cue_subtask[:, 0, 0])

    def get_observation(self, last_reward, last_action, time_step, cue_rule, cue_subtask):
        return torch.cat([
            cue_rule,
            cue_subtask,
            last_action,
            last_reward.unsqueeze(-1),
            torch.ones(self.batch_size, 1) * time_step],
            dim=1)

    def sample_mean_reward_subtask(self, linear=True):
        if linear:
            x = torch.linspace(0, self.num_actions-1, self.num_actions).unsqueeze(0)
            self.w = torch.empty(self.batch_size, 1).uniform_(-2.5, 2.5)
            b = torch.empty(self.batch_size, 1).uniform_(2.5, 7.5)
            y = self.w @ (2.0*x/5.0 -1) + b
        else:
            x = torch.linspace(0, self.num_actions-1, self.num_actions)
            self.phase = torch.randint(0, 2, (self.batch_size, 1))
            amp = torch.empty(self.batch_size, 1).uniform_(0., 7.5)
            b = (0-amp/1.4)* torch.rand(self.batch_size, 1) + amp/1.4
            y = amp * torch.abs(torch.sin((x-self.phase) * (0.5 * np.pi)))
            y = y + b

        # dimensions: self.batch_size X self.num_actions
        return y

    def step(self, action):
        regrets = self.mean_reward[:, self.subtask_index].max(dim=1)[0] - self.mean_reward[:, self.subtask_index].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.rewards[:, self.subtask_step, self.subtask_index].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = reward / self.reward_scaling
        self.t += 1
        self.subtask_step += 1
        if self.subtask_step >=5:
            self.subtask_step = 0
            self.subtask_index += 1
        done = True if (self.t >= self.max_steps) else False

        if not done:
            observation = self.get_observation(reward, F.one_hot(action.long(), num_classes=self.num_actions), self.t, self.cue_rule[:, self.subtask_step, self.subtask_index], self.cue_subtask[:, self.subtask_step, self.subtask_index])
        else:
            observation = torch.zeros(self.batch_size, self.num_actions + 6) # this is not used
        return observation, reward, done, {'regrets': regrets.mean()}
