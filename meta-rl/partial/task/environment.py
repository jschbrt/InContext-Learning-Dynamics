"""
This file contains the contextual bandit task used for the partial task.
"""
import numpy as np
import torch

class ContextualBanditTask():
    def __init__(self, batch_size, mode, success=0.5, fail=0.0, debugging=False,with_random=False) -> None:
        self.number_cues = 4
        self.trials_per_cue = 24 if not debugging else 4
        self.batch_size = batch_size
        self.mode = mode
        self.with_random = with_random
        self.success = success
        self.fail = fail
        self.mode_to_prob_cues = {
            'test': [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
            'high_prob': [[0.9, 0.8], [0.8, 0.9], [0.6, 0.9], [0.7, 0.8]],
            'low_prob': [[0.1, 0.2], [0.2, 0.1], [0.4, 0.1], [0.3, 0.2]],
            'test_2': [[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
        }

        self.setup()

    def setup(self):
        self.setup_prob_cues()
        self.generate_cue_order()
        self.calculate_expected_rewards()
        self.max_steps = len(self.cue_order[0])
        self.step = 0
        self.finished = False

    def calculate_expected_rewards(self):
        self.expected_rewards = self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail

    def generate_cue_order(self):
        self.cue_order = np.tile(np.arange(self.number_cues), (self.batch_size, self.trials_per_cue))
        # shuffle for each batch differently
        np.apply_along_axis(np.random.shuffle, 1, self.cue_order)

    def setup_prob_cues(self):
        if self.mode == 'train':
            self.prob_list_cues = np.random.uniform(size=(self.batch_size,8))
            self.prob_list_cues = np.reshape(self.prob_list_cues, (self.batch_size,4,2))
            self.prob_list_cues = self.prob_list_cues.round(2)
        elif self.mode in self.mode_to_prob_cues:
            self.prob_list_cues = np.repeat(np.array([self.mode_to_prob_cues[self.mode]]), self.batch_size, axis=0)

    def cue(self):
        return torch.from_numpy(self.cue_order[:, self.step])

    def sample(self, actions):
        cue = self.cue_order[:, self.step]

        if self.with_random:
            self.rand_actions = np.random.randint(0,2,self.batch_size)
            rewards, counter_rewards, rand_rewards = self.rewards(actions, cue)
            regrets, rand_regrets, optimal_actions, rand_optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = (torch.from_numpy(rand_rewards), torch.from_numpy(rand_regrets), torch.from_numpy(rand_optimal_actions), torch.from_numpy(self.rand_actions))

        else:
            rewards, counter_rewards = self.rewards(actions, cue)
            regrets, optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = 'none'

        self.step +=1
        if self.step == self.max_steps:
            self.finished = True

        return torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions), random_data, self.finished

    def rewards(self, actions, cue):
        cues = self.prob_list_cues[np.arange(self.batch_size), cue, :] # select the correct cue that is used for this experiment
        probs = cues[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        self.counter_actions = [int(not(a)) for a in actions]
        counter_probs = cues[np.arange(len(actions)), self.counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail

        if self.with_random:
            """Get some random rewards"""
            rand_probs = cues[np.arange(len(self.rand_actions)), self.rand_actions] # select the prob of the chosen action for a success
            rand_successes = np.random.binomial(1,rand_probs)
            rand_rewards = rand_successes * self.success + np.logical_not(rand_successes) * self.fail
            return rewards, counter_rewards, rand_rewards
        else:
            return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions, cue):
        expected_rewards = self.expected_rewards[np.arange(self.batch_size), cue] # select the correct cue that is used for this experiment
        regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(expected_rewards, axis=1)
        if self.mode == 'test':
            optimal_actions = np.where((cue == 0) | (cue == 3), actions, optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
        
        if self.with_random:
            rand_regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(self.rand_actions)), self.rand_actions]
            rand_optimal_actions = np.argmax(expected_rewards, axis=1)
            if self.mode == 'test':
                rand_optimal_actions = np.where((cue == 0) | (cue == 3), self.rand_actions, rand_optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
            return regrets, rand_regrets, optimal_actions, rand_optimal_actions
        else:
            return regrets, optimal_actions

#o = ContextualBanditTask(1000, mode='train', with_random=False)
#o.sample(np.repeat(1,1000))
