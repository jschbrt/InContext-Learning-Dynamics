from typing import Any
import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from itertools import groupby

class AgentMemoryOptim:
    """Memory of the agent. Stores the data of the agent during the simulation."""
    def __init__(self) -> None:
        pass

    def init_tensors(self, episode_length, batch_size):

        self.batch_size = batch_size
        self.episode_length = episode_length

        self.prompt =  []
        self.block_idx = self.create_tensor(init_value=-1.)
        self.forced_type = self.create_tensor()
        self.reward_block_type = self.create_tensor()
        self.trials_idx = self.create_tensor()
        self.rewards = self.create_tensor()
        self.counter_rewards = self.create_tensor()
        self.actions = self.create_tensor()
        self.optimal_actions = self.create_tensor()
        self.cues = self.create_tensor()
        self.regrets = self.create_tensor()

    def create_tensor(self, init_value=0.):
        return torch.full((self.batch_size, self.episode_length), init_value)

    def insert_cues(self, 
                    idx, 
                    trials_idx, 
                    block_idx, 
                    cues):

        self.cues[:,idx].copy_(torch.tensor([cues]))
    
        block_idx = torch.tensor([block_idx]*self.batch_size)
        self.block_idx[:, idx].copy_(block_idx)

        trials_idx = torch.tensor([trials_idx]*self.batch_size) 
        self.trials_idx[:, idx].copy_(trials_idx)

    def insert_data(self, 
                    idx, 
                    forced_type, 
                    reward_block_type, 
                    rewards, 
                    forgone_rewards, 
                    regrets, 
                    actions, 
                    optimal_actions, 
                    prompt):

        forced_type = torch.tensor([forced_type]*self.batch_size)
        reward_block_type = torch.tensor(reward_block_type)

        self.forced_type[:, idx].copy_(forced_type)
        self.reward_block_type[:, idx].copy_(reward_block_type)
        self.rewards[:, idx].copy_(rewards)
        self.counter_rewards[:, idx].copy_(forgone_rewards)
        self.regrets[:, idx].copy_(regrets)
        self.actions[:, idx].copy_(actions)
        self.optimal_actions[:, idx].copy_(optimal_actions)
        self.prompt.append(prompt)

    def get_simulated_data(self):
        saved_data = (self.block_idx, 
                      self.forced_type, 
                      self.reward_block_type, 
                      self.trials_idx, 
                      self.rewards, 
                      self.counter_rewards, 
                      self.regrets, 
                      self.optimal_actions, 
                      self.actions)
        return self.cues, self.prompt, saved_data

class LogProgress:
    """Log the progress of the training."""

    def __init__(self, 
                 folder_path, 
                 name,
                 episodes) -> None:
        """
        Internal Memory of the Advantage Actor Critic (A2C).
      
        Args:
            :episode_length: The length of the rollout of the LSTM.
            :is_cuda: Bool indicating whether or not to use GPU processing.
            :writer: torch.SummaryWriter for logging
        """
        self.folder_path = folder_path
        self.eps = episodes
        self.name = name

    def init_df(self, trial_per_eps):

        self.trials_per_eps = trial_per_eps
        self.init_simulation_df()

    def init_simulation_df(self):

        simulation_df_columns = [
            'run', # multiple participants per episode
            'block_idx',
            'block_forced_type',
            'block_reward_type',
            'trials_idx',
            'cues',
            'actions',
            'rewards',
            'forgone_rewards',
            'regrets',
            'opt_actions',
            'prompt',
        ]
        df = pd.DataFrame(columns=simulation_df_columns)
        df = df.astype({'run':int,
                        'block_idx':int,
                        'block_forced_type':int,
                        'block_reward_type':int,
                        'trials_idx':int,
                        'cues':int,
                        'actions':int,
                        'rewards':float,
                        'forgone_rewards':float,
                        'regrets':float,
                        'opt_actions':int,
                        'prompt':str,
                        })
        self.simulation_df = df

        if not os.path.exists(self.folder_path+self.name+'.csv'):
            self.simulation_df.to_csv(self.folder_path+self.name+'.csv', mode='w', index=False, header=True)
        else:
            self.simulation_df = pd.read_csv(self.folder_path+self.name+'.csv')

    def save_test(self, eps_idx, simulated_data):
            
            cues, prompt, saved_data = simulated_data
            block_idx, forced_type, reward_block_type, trials_idx, rewards, forgone_rewards, regrets, opt_actions, actions = [d.detach() for d in saved_data]
    
            # Create dataframe
            df = pd.DataFrame({
                'run': np.repeat(eps_idx, self.trials_per_eps),
                'block_idx': block_idx.flatten(),
                'block_forced_type': forced_type.flatten(),
                'block_reward_type': reward_block_type.flatten(), 
                'trials_idx': trials_idx.flatten(),
                'cues': cues.flatten(),
                'actions': actions.flatten(),
                'rewards': rewards.flatten(),
                'forgone_rewards': forgone_rewards.flatten(),
                'regrets': regrets.flatten(),
                'opt_actions': opt_actions.flatten(),
                'prompt': prompt
            })

            df.to_csv(self.folder_path+'sim.csv', mode='a', index=False, header=False)