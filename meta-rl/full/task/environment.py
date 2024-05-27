"""
Environment for the complete task.
"""
import numpy as np
import torch

# - 1: high, partial
# - 2: high, full
# - 3: low, partial
# - 4: low, full
# per block: 20 free, 20 forced

class CompleteTask():

    def __init__(self, batch_size, train, success=1., fail=-1., debugging=False) -> None:
        
        # Assign input parameters as properties
        self.success = success
        self.fail = fail
        self.number_of_cues = 2
        self.batch_size = batch_size
        self.train = train

        # Setup the correct acount of blocks/trials for the task
        if debugging:
            self.number_free = 20 // 2
            self.number_forced = 20 // 2
        else:
            self.number_free = 20
            self.number_forced = 20
        self.total_trials_block = self.number_forced + self.number_free

        self.number_of_blocks = 4
        self.total_trials = 4 * self.total_trials_block
        self.setup_blocks()
        self.start_block(0)

    def setup_blocks(self):
        """ Set up a dict containing positions of forced actions and trial order of forced actions in blocks. """
        reward_block_type = np.asarray([0,0,1,1]) # high/low reward blocks 
        feedback_type = np.asarray([0,1,0,1])
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, # 0 high, 1 low
                       'feedback_type':feedback_type, # 0 partial, 1 full
                       'forced_actions': forced_actions, # 0: Symbol 1, 1: Symbol 2
                       'forced_trial_order': trial_order} # 0: free, 1: forced
        
        # Set up episode variables
        self.total_trials_idx = 0

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
        forced_actions = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, self.number_forced // 2)) # we should have 20 forced actions per block and trial_order = 40 as free and forced
        np.apply_along_axis(np.random.shuffle, 2, forced_actions)
        # trial order of free-choice (0) and forced-choice trials (1) 
        trial_order = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, self.number_forced))
        np.apply_along_axis(np.random.shuffle, 2, trial_order)
        
        return forced_actions, trial_order

    def start_block(self, block_idx):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.forced_step_idx = np.zeros(self.batch_size, dtype=int)
        self.reward_block_type = self.blocks['reward_block_type'][block_idx]
        self.feedback_type = self.blocks['feedback_type'][block_idx]

        self.forced_trial_order = self.blocks['forced_trial_order'][:, block_idx, :]
        self.forced_actions = self.blocks['forced_actions'][:, block_idx, :]

        # Set up reward probabilities
        if self.train: 
            self.rew_probs = np.random.uniform(size=(self.batch_size,2)).round(2)
        else:
            self.reward_block_prob_dict = {
                0: np.array([0.6, 0.9]),
                1: np.array([0.4, 0.1])
            }
            self.rew_probs = np.repeat([self.reward_block_prob_dict[self.reward_block_type]], repeats=self.batch_size, axis=0)
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def cue(self, step):

        current_trial = self.forced_trial_order[:, step]
        self.is_forced_idx = np.where(current_trial == 1)[0]

        one_hot_cue = np.zeros((self.batch_size, self.number_of_cues))

        if self.is_forced_idx.size > 0:
            self.step_forced_actions = self.forced_actions[self.is_forced_idx, self.forced_step_idx[self.is_forced_idx]]
            one_hot_cue[self.is_forced_idx, self.step_forced_actions] = 1
            
        return torch.from_numpy(one_hot_cue)
    
    def sample(self, actions, step):

        # Overwrite action in forced choice trials.
        actions = np.asarray(actions) if isinstance(actions, list) else actions

        if self.is_forced_idx.size > 0:
            actions[self.is_forced_idx] = self.step_forced_actions
            self.forced_step_idx[self.is_forced_idx] += 1

        rewards, counter_rewards = self.rewards(actions)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions)

        # set indices correctly
        self.total_trials_idx +=1
        self.step_in_block +=1

        if self.feedback_type == 1:
            return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)
        else: # return 0 for counter_rewards if feedback is partial
            zeros = np.zeros_like(counter_rewards)
            return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(zeros), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

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

# ct = CompleteTask(5)
# ct.setup_blocks()
# ct.start_block(0)
# ct.cue(0)
# ct.sample([0]*5,0)
# ct.sample([0]*5,1)
# ct.sample([1]*5,2)