import numpy as np
import gymnasium as gym
from gym import spaces
import torch
import random

class OptimismBiasTaskPalminteri(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, num_actions=2, max_steps_per_context=2, num_contexts=4, reward_scaling=1, batch_size=1): # 1 for simulations
        
        self.num_actions = num_actions
        self.num_contexts = num_contexts
        self.batch_size = batch_size
        self.max_steps_per_context =  max_steps_per_context 
        self.reward_scaling = reward_scaling
        self.action_space = spaces.Discrete(self.num_actions)
        self.probs =  {'low':0.25, 'high':0.75} #{'low':0., 'high':1.} 
        self.observation_space = spaces.Box(np.ones(5), np.ones(5)) #  5 = action+reward+context+trial

    def reset(self):

        # keep track of time-steps
        self.t = 0
        self.max_steps = self.max_steps_per_context * self.num_contexts
        
        # sample the letters for each context
        self.arms = []
        self.alphabet = 'ABCDEFGHJKLMNOPQRSTVW'
        for i in range(1, 1+self.num_contexts):
            arm1, arm2, self.alphabet = self.sample_alphabet(self.alphabet)
            self.arms.append({ arm1:0, arm2:1})

        # generate reward functions
        mean_rewards_context_ll, rewards_context_ll = self.sample_contextual_rewards(context=['low', 'low'])
        mean_rewards_context_lh, rewards_context_lh = self.sample_contextual_rewards(context=['low', 'high'])
        mean_rewards_context_hl, rewards_context_hl = self.sample_contextual_rewards(context=['high', 'low']) #['high', 'low'])
        mean_rewards_context_hh, rewards_context_hh = self.sample_contextual_rewards(context=['high', 'high']) #['high', 'high'])

       # integer machines
        self.cue_context = torch.zeros(self.batch_size, self.num_contexts, self.max_steps_per_context, 1)
        self.cue_context[:, 0] = 1
        self.cue_context[:, 1] = 2
        self.cue_context[:, 2] = 3
        self.cue_context[:, 3] = 4

        # stack all the rewards together
        # shape: batch_size X max_steps X num_actions/context_size
        self.mean_rewards = torch.stack((mean_rewards_context_ll, mean_rewards_context_lh, mean_rewards_context_hl, mean_rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
        self.rewards = torch.stack((rewards_context_ll, rewards_context_lh, rewards_context_hl, rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
        self.contexts = self.cue_context.reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, 1)
        
        # shuffle the orders
        shuffle_idx = torch.randperm(self.max_steps_per_context*self.num_contexts)
        self.mean_rewards =  self.mean_rewards[:, shuffle_idx]
        self.rewards =  self.rewards[:, shuffle_idx] 
        self.contexts = self.contexts[:, shuffle_idx]
        last_reward = torch.zeros(self.batch_size)
        last_action = 0 #torch.zeros(self.batch_size)

        return self.get_observation(last_reward, last_action, self.t, self.contexts[:, 0])

    def get_observation(self, last_reward, last_action, time_step, cue_context):
        return torch.cat([
            torch.tensor([last_reward]).unsqueeze(-1), #last_reward.unsqueeze(-1),
            torch.tensor([last_action]).unsqueeze(-1),
            torch.ones(self.batch_size, 1) * time_step,
            cue_context],
            dim=1)

    def sample_contextual_rewards(self, context):
        assert len(context)==self.num_actions, "lengths of context and actions do not match"
        ones = torch.ones((self.batch_size, self.max_steps_per_context, self.num_actions))
        mean_rewards_context = torch.zeros((self.batch_size, self.max_steps_per_context, self.num_actions))
        for idx, option in enumerate(context):
            ones[..., idx] = ones[..., idx]*self.probs[option]
            mean_rewards_context[..., idx] = self.probs[option]
        rewards_context = torch.bernoulli(ones)
        return mean_rewards_context, rewards_context

    def step(self, action):
       
        regrets = self.mean_rewards[:, self.t].max(1).values[0] - self.mean_rewards[:, self.t][:, action][0]
        reward = self.rewards[:, self.t][:, action][0]
        reward = reward / self.reward_scaling
        self.t += 1
        done = True if (self.t >= self.max_steps-1) else False
        
        observation = self.get_observation(reward, action, self.t, self.contexts[:, self.t])
        return observation, reward, done, {'regrets': regrets.mean()}

    def sample_alphabet(self, alphabet): 
        arm1 = random.choice(alphabet)
        alphabet = alphabet.replace(arm1, '')
        arm2 = random.choice(alphabet)
        alphabet = alphabet.replace(arm2, '')
        return arm1, arm2, alphabet


#! hashed out for now
# class TwoArmedBanditTask(gym.Env):
#     metadata = {'render.modes': ['human']}
#     def __init__(self, num_actions=2, max_steps_per_context=2, num_contexts=4, reward_scaling=1, batch_size=1): # 1 for simulations
        
#         self.num_actions = num_actions
#         self.num_contexts = num_contexts
#         self.batch_size = batch_size
#         self.max_steps_per_context =  max_steps_per_context 
#         self.reward_scaling = reward_scaling
#         self.action_space = spaces.Discrete(self.num_actions)
#         self.probs =  {'low':0., 'high':1.} #{'low':0.25, 'high':0.75}
#         self.observation_space = spaces.Box(np.ones(5), np.ones(5)) #  5 = action+reward+context+trial
 
#     def reset(self):

#         # keep track of time-steps
#         self.t = 0
#         self.max_steps = self.max_steps_per_context * self.num_contexts

#         # generate reward functions
#         mean_rewards_context_ll, rewards_context_ll = self.sample_contextual_rewards(context=['low', 'high']) #['low', 'low'])
#         mean_rewards_context_lh, rewards_context_lh = self.sample_contextual_rewards(context=['low', 'high'])
#         mean_rewards_context_hl, rewards_context_hl = self.sample_contextual_rewards(context=['low', 'high']) #['high', 'low'])
#         mean_rewards_context_hh, rewards_context_hh = self.sample_contextual_rewards(context=['low', 'high']) #['high', 'high'])

#        # integer machines
#         self.cue_context = torch.zeros(self.batch_size, self.num_contexts, self.max_steps_per_context, 1)
#         self.cue_context[:, 0] = 1
#         self.cue_context[:, 1] = 2
#         self.cue_context[:, 2] = 3
#         self.cue_context[:, 3] = 4

#         # one hot machines
#         # self.cue_context = torch.zeros(self.batch_size, self.num_contexts, self.max_steps_per_context, 2)
#         # self.cue_context[:, 0, :, :] = 0
#         # self.cue_context[:, 1, :, 1] = 1
#         # self.cue_context[:, 2, :, 0] = 1
#         # self.cue_context[:, 3, :, :] = 1

#         # stack all the rewards together
#         # shape: batch_size X max_steps X num_actions/context_size
#         self.mean_rewards = torch.stack((mean_rewards_context_ll, mean_rewards_context_lh, mean_rewards_context_hl, mean_rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
#         self.rewards = torch.stack((rewards_context_ll, rewards_context_lh, rewards_context_hl, rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
#         self.contexts = self.cue_context.reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, 1)
        
#         # shuffle the orders
#         shuffle_idx = torch.randperm(self.max_steps_per_context*self.num_contexts)
#         self.mean_rewards =  self.mean_rewards[:, shuffle_idx]
#         self.rewards =  self.rewards[:, shuffle_idx] 
#         self.contexts = self.contexts[:, shuffle_idx]
#         last_reward = torch.zeros(self.batch_size)
#         last_action = 0 #torch.zeros(self.batch_size)

#         return self.get_observation(last_reward, last_action, self.t, self.contexts[:, 0])

#     def get_observation(self, last_reward, last_action, time_step, cue_context):
#         return torch.cat([
#             torch.tensor([last_reward]).unsqueeze(-1), #last_reward.unsqueeze(-1),
#             torch.tensor([last_action]).unsqueeze(-1),
#             torch.ones(self.batch_size, 1) * time_step,
#             cue_context],
#             dim=1)

#     def sample_contextual_rewards(self, context):
#         assert len(context)==self.num_actions, "lengths of context and actions do not match"
#         ones = torch.ones((self.batch_size, self.max_steps_per_context, self.num_actions))
#         mean_rewards_context = torch.zeros((self.batch_size, self.max_steps_per_context, self.num_actions))
#         for idx, option in enumerate(context):
#             ones[..., idx] = ones[..., idx]*self.probs[option]
#             mean_rewards_context[..., idx] = self.probs[option]
#         rewards_context = torch.bernoulli(ones)
#         return mean_rewards_context, rewards_context

#     def step(self, action):
       
#         regrets = self.mean_rewards[:, self.t].max(1).values[0] - self.mean_rewards[:, self.t][:, action][0]
#         reward = self.rewards[:, self.t][:, action][0]
#         reward = reward / self.reward_scaling
#         self.t += 1
#         done = True if (self.t >= self.max_steps-1) else False
        
#         observation = self.get_observation(reward, action, self.t, self.contexts[:, self.t])
#         return observation, reward, done, {'regrets': regrets.mean()}

