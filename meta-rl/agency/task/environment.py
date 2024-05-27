"""
Contextual Bandit Class
"""

import numpy as np
import torch

class AgencyTask():
    """ Env for a 2-armed bandit task with forced choice """

    def __init__(self, batch_size=int, success=1., fail=-1., test_reward_probs:str='original', testing=False) -> None:
        
        # Assign input parameters as properties
        self.batch_size = batch_size
        self.success = success
        self.fail = fail
        self.number_of_cues = 2
        self.test_reward_probs = test_reward_probs

        # Setup the correct acount of blocks/trials for the task
        if testing:
            self.number_free = 40 // 4
            self.number_forced = 80 // 4
        else:
            self.number_free = 40
            self.number_forced = 80
        self.number_of_blocks = 4
        self.total_trials = (2 * self.number_free) + (2 * self.number_forced)
        self.setup_blocks()
        self.start_block(0)

    def setup_blocks(self):
        """ Set up a dict containing positions of forced actions and trial order of forced actions in blocks. """
        reward_block_type = np.repeat([[0,0,1,1]], repeats=self.batch_size, axis=0) # high/low reward blocks 
        forced_type = np.array([0,1,0,1]) # free/forced choice blocks
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, # 0 high, 1 low
                       'forced_type':forced_type, # 0 free, 1 forced
                       'forced_actions': forced_actions, # 0: Symbol 1, 1: Symbol 2
                       'forced_trial_order': trial_order} # 0: free, 1: forced
        
        # Set up episode variables
        self.total_trials_idx = 0
        self.forced_block = 0

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
        forced_actions = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks // 2, self.number_forced//4))
        np.apply_along_axis(np.random.shuffle, 2, forced_actions)
        # trial order of free-choice (0) and forced-choice trials (1) 
        trial_order = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks // 2, self.number_forced//2))
        np.apply_along_axis(np.random.shuffle, 2, trial_order)
        
        return forced_actions, trial_order

    def start_block(self, block, train=False):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.reward_block_type = self.blocks['reward_block_type'][:, block]
        self.forced_type = self.blocks['forced_type'][block]

        if self.forced_type == 1:
            self.forced_step_in_block = np.zeros(self.batch_size, dtype=int)
            self.forced_trial_order = self.blocks['forced_trial_order'][:, self.forced_block, :]
            self.forced_actions = self.blocks['forced_actions'][:, self.forced_block, :]
            self.forced_block += 1
            self.total_trials_block = self.number_forced
        else:
            self.total_trials_block = self.number_free

        # Set up reward probabilities
        if train: 
            self.rew_probs = np.random.uniform(size=(self.batch_size,2)).round(2)
        else:
            self.set_test_reward_probs()
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def set_test_reward_probs(self):

        if self.test_reward_probs == 'original':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.6]),
                1: np.array([0.4, 0.1])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])
        elif self.test_reward_probs == 'high':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.8]),
                1: np.array([0.8, 0.9])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])

    def cue(self, step):
        """ Returns batch wether trial is forced choice (1) or free choice (0) trial. """
        
        one_hot_cues = np.zeros((self.batch_size, 2))

        if self.forced_type == 1:
            cue = self.forced_trial_order[:, step]
            # Get the forced actions
            forced_trials_index = np.where(cue == 1)[0]

            if forced_trials_index.size > 0:
                forced_actions = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
                one_hot_forced_actions = np.zeros((len(forced_actions), 2))
                one_hot_forced_actions[np.arange(len(forced_actions)), forced_actions] = 1

                one_hot_cues[forced_trials_index, :] = one_hot_forced_actions
        
        return torch.from_numpy(one_hot_cues)

    def sample(self, actions, step):
        """ 
        Samples rewards and regrets. 
        
        Parameters:
            - actions: batch of actions in current bandit
            - step in current bandit

        """

        # Overwrite action in forced choice trials.
        actions = np.asarray(actions) if isinstance(actions, list) else actions

        if self.forced_type == 1:
            cue = self.forced_trial_order[:, step]
            forced_trials_index = np.where(cue == 1)[0]

            if forced_trials_index.size > 0:
                actions[forced_trials_index] = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
                self.forced_step_in_block[forced_trials_index] += 1

        rewards, counter_rewards = self.rewards(actions)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions)

        # set indices correctly
        self.total_trials_idx +=1
        self.step_in_block +=1
        return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

    def rewards(self, actions):
        probs = self.rew_probs[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        counter_actions = [int(not(a)) for a in actions]
        counter_probs = self.rew_probs[np.arange(len(actions)), counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions):
        regrets =  self.expected_rewards.max(axis=1) - self.expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(self.expected_rewards, axis=1)
        return regrets, optimal_actions
