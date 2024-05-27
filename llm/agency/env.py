"""
Contextual Bandit Class for the 2-armed bandit task with free and forced choices.
"""
import numpy as np
import random

class AgencyTask():
    """ Env for a 2-armed bandit task with forced choice """
    def __init__(self, success=1., fail=-1., test_reward_probs:str='original', testing=False) -> None:
        
        # Assign input parameters as properties
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
        reward_block_type = np.asarray([0,0,1,1]) 
        forced_type = np.asarray([0,1,0,1])
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, # 0 high, 1 low
                       'forced_type':forced_type, # 0 free, 1 forced
                       'forced_actions': forced_actions, # 0: Symbol 1, 1: Symbol 2
                       'forced_trial_order': trial_order} # 0: free, 1: forced
        
        # Set up episode variables
        self.total_trials_idx = 0
        self.forced_block = 0

        # sample the letters for each context
        self.block_arms = []
        self.alphabet = 'ABCDEFGHJKLMNOPQRSTVW'

        for i in range(self.number_of_blocks):
            arm1, arm2, self.alphabet = self.sample_alphabet(self.alphabet)
            self.block_arms.append({ arm1:0, arm2:1})

    def sample_alphabet(self, alphabet): 
        arm1 = random.choice(alphabet)
        alphabet = alphabet.replace(arm1, '')
        arm2 = random.choice(alphabet)
        alphabet = alphabet.replace(arm2, '')
        return arm1, arm2, alphabet

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
        forced_actions = np.tile(np.arange(2), (self.number_of_blocks // 2, self.number_forced//4))
        np.apply_along_axis(np.random.shuffle, 1, forced_actions)
        # trial order of free-choice (0) and forced-choice trials (1) 
        trial_order = np.tile(np.arange(2), (self.number_of_blocks // 2, self.number_forced//2))
        np.apply_along_axis(np.random.shuffle, 1, trial_order)
        
        return forced_actions, trial_order

    def start_block(self, block, train=False):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.reward_block_type = self.blocks['reward_block_type'][block]
        self.forced_type = self.blocks['forced_type'][block]

        self.arms = self.block_arms[block]

        if self.forced_type == 1:
            self.forced_trials_idx = 0
            self.forced_trial_order = self.blocks['forced_trial_order'][self.forced_block, :]
            self.forced_actions = self.blocks['forced_actions'][self.forced_block, :]
            self.forced_block += 1
            self.total_trials_block = self.number_forced
        else:
            self.total_trials_block = self.number_free

        # Set up reward probabilities
        self.reward_block_prob_dict = {
                0: np.array([0.9, 0.6]),
                1: np.array([0.4, 0.1])
            }
        self.rew_probs = self.reward_block_prob_dict[self.reward_block_type]
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def cue(self, step):
        if self.forced_type == 1:
            is_forced = self.forced_trial_order[step]
        else:
            is_forced = 0
        return is_forced

    def sample(self, action, step):

        # Overwrite action in forced choice trials.     
        if self.forced_type == 1:
            is_forced = self.forced_trial_order[step]
            if is_forced:
                action = self.forced_actions[self.forced_trials_idx]
                self.forced_trials_idx += 1

        rewards, counter_rewards = self.rewards(action)
        regrets, optimal_actions = self.regrets_and_optimal_actions(action)

        # set indices correctly
        self.total_trials_idx +=1
        self.step_in_block +=1
        return action, rewards, counter_rewards, regrets, optimal_actions

    def rewards(self, action):
        probs = self.rew_probs[action] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        counter_action = int(not(action))
        counter_probs = self.rew_probs[counter_action] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions):
        regrets =  self.expected_rewards.max() - self.expected_rewards[actions]
        optimal_actions = np.argmax(self.expected_rewards)
        return regrets, optimal_actions

"""
ct = AgencyTask()
ct.setup_blocks()
ct.start_block(3)
ct.cue(0)
ct.sample(0,0)
ct.sample(0,1)
ct.sample(0,2)"""