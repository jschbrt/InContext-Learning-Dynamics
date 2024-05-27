"""
Contains a simple Bandit Class for the Partial Task with 2, 3 or 4 options.
"""
import numpy as np
import random

class PartialTask():
    def __init__(self, success=0.5, fail=0.0, num_options=2, debugging=False) -> None:

        self.success = success
        self.fail = fail
        self.num_options = num_options

        if num_options == 2:
            self.number_cues = 4
            self.trials_per_cue = 24 if not debugging else 4
            self.prob_list_cues = np.array([[0.25, 0.25], 
                                            [0.25, 0.75], 
                                            [0.75, 0.25], 
                                            [0.75, 0.75]])

        elif num_options == 3:
            self.number_cues = 3
            self.trials_per_cue = 32 if not debugging else 4
            self.prob_list_cues = np.array([[0.25, 0.25, 0.75], 
                                            [0.75, 0.25, 0.25], 
                                            [0.75, 0.25, 0.75]])

        elif num_options == 4:
            self.number_cues = 2
            self.trials_per_cue = 48 if not debugging else 4
            self.prob_list_cues = np.array([[0.25, 0.25, 0.25, 0.75], 
                                            [0.75, 0.25, 0.75, 0.75]])

        self.total_trials = self.trials_per_cue * self.number_cues
        self.setup()

    def setup(self):
        self.generate_cue_order()
        self.calculate_expected_rewards()
        self.max_steps = len(self.cue_order)
        self.step = 0
        self.finished = False

        self.setup_cue_arms()

    def setup_cue_arms(self):
        # sample the letters for each context
        self.cue_arms = []
        self.alphabet = 'ABCDEFGHJKLMNOPQRSTVW'

        for i in range(self.number_cues):
            arms, self.alphabet = self.sample_alphabet(self.alphabet, self.num_options)
            self.cue_arms.append(arms)

    def sample_alphabet(self, alphabet, num_options): 
        arms = dict()
        for i in range(num_options):
            arm = random.choice(alphabet)
            alphabet = alphabet.replace(arm, '')
            arms[arm] = i
        return arms, alphabet
        
    def calculate_expected_rewards(self):
        self.expected_rewards = self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail

    def generate_cue_order(self):
        self.cue_order = np.tile(np.arange(self.number_cues), (self.trials_per_cue))
        # shuffle for each batch differently
        np.apply_along_axis(np.random.shuffle, 0, self.cue_order)

    def cue(self):
        return self.cue_order[self.step]
    
    def arms(self):
        return self.cue_arms[self.cue()]

    def sample(self, actions):
        cue = self.cue()

        rewards = self.rewards(actions, cue)
        regrets = self.regrets(actions, cue)
        return rewards, regrets

    def step_forward(self):
        self.step +=1
        if self.step == self.max_steps:
            self.finished = True

    def rewards(self, actions, cue):
        cues = self.prob_list_cues[cue, :] # select the correct cue that is used for this experiment
        probs = cues[actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        return rewards

    def regrets(self, actions, cue):
        expected_rewards = self.expected_rewards[cue] # select the correct cue that is used for this experiment
        regrets =  expected_rewards.max() - expected_rewards[actions]
        return regrets

#o = PartialTask()
#o.sample(0)
