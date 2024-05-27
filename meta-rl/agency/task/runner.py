"""
    Runner class for training and testing the agent on the AgencyTask.
"""
import sys
import torch
import torch.nn as nn
import os

sys.path.append('../../meta-rl_utils')
from agent import A2COptim
from environment import AgencyTask
from utils import AgentMemoryOptim, LogProgress

class Runner():
    def __init__(self, 
                 agent:A2COptim, 
                 bandit: AgencyTask,
                 memory: AgentMemoryOptim,
                 logger: LogProgress
                 ) -> None:
        
        self.agent = agent
        self.memory = memory
        self.agent_model = agent.agent_model
        self.batch_size = agent.batch_size
        self.bandit = bandit
        self.logger = logger

    def _rollout_trajectory(self, block_idx, train=True):
        for trials_idx in range(self.bandit.total_trials_block):
            idx = self.bandit.total_trials_idx
            self.memory.insert_cues(idx, trials_idx, block_idx, self.bandit.cue(trials_idx))
            x = self.memory.get_padded_data(idx, trials_idx, block_idx)
            # generate an action and return policy, value_fn (baseline)
            if train:
                actions, log_policy_a, entropy, value_fn, policy = self.agent.step(x, trials_idx)
            else:
                with torch.no_grad():
                    actions, log_policy_a, entropy, value_fn, policy = self.agent.step(x, trials_idx)
            # interact with environment
            actions_replaced, rewards, forgone_rewards, regrets, optimal_actions = self.bandit.sample(actions.cpu().numpy(), trials_idx)
            self.memory.insert_data(idx, trials_idx, self.bandit.forced_type, self.bandit.reward_block_type, rewards, forgone_rewards, regrets, actions_replaced, optimal_actions, log_policy_a, value_fn, entropy, policy)

    def training(self, 
                 train_eps, 
                 entropy_final_value, 
                 max_grad_norm):

        # Set up dataframe for saving/plotting loss/reward data
        self.logger.init_df(self.bandit.total_trials) 
        for train_idx in range(train_eps):
            # Calculate decaying entropy coefficient
            entropy_coef = max(1-(train_idx/(0.5*train_eps)), entropy_final_value)

            # Set up bandit for forced choice task
            self.bandit.setup_blocks()
            self.memory.init_tensors(self.bandit.total_trials, self.batch_size) # initializes / resets tensors

            # save losses
            losses_l = []
            # Simulate agent for each block
            for block_idx in range(self.bandit.number_of_blocks):

                # ensure all gradients are 0
                self.agent.optimizer.zero_grad()

                self.bandit.start_block(block_idx, train=True)
                self.memory.init_tensors_block(self.bandit.total_trials_block)
                self._rollout_trajectory(block_idx)

                """Calculate loss, propagate back, clip gradients and then do optimization on weights"""
                losses = self.memory.a2c_loss(entropy_coef, block_idx)
                loss = losses[0]
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.net.parameters(), max_grad_norm)
                self.agent.optimizer.step()
                #losses_l.append([l.detach().cpu().numpy() for l in losses])
            
            losses_l = zip(losses_l)
            mean_losses = [sum(lst) / len(lst) for lst in losses_l]
            simulated_data = self.memory.get_simulated_data()

            self.logger.plot_save(train_idx, simulated_data, agent=self.agent, losses=losses)

        """Save the best hyperparameters for test."""
        self.logger.save_model(train_idx, self.agent, final=True)

    def test(self, test_eps, folder_path):
        """
            Attributes:
            test_case (str): 'independent': independent bandit
                             'test_2': only two two armed bandits
                             'test': full test case with four two armed bandits
        """

        # Load the model
        if not os.path.exists(folder_path+'model_state.pt'):
            raise ValueError(f'No model found for {folder_path}')
        if self.memory.is_cuda:
            checkpoint = torch.load(os.path.join(folder_path,"model_state.pt"))
        else:
            checkpoint = torch.load(os.path.join(folder_path,"model_state.pt"), map_location=torch.device('cpu'))
        self.agent.net.load_state_dict(checkpoint['model'])

        # Initialize dataframe for saving/plotting loss/reward data
        self.logger.init_df(self.bandit.total_trials)

        for test_idx in range(test_eps):

            # Set up bandit for forced choice task
            self.bandit.setup_blocks()
            self.memory.init_tensors(self.bandit.total_trials, self.batch_size) # initializes / resets tensors

            # Simulate agent for each block
            for block_idx in range(self.bandit.number_of_blocks):
                
                self.bandit.start_block(block_idx, train=False)
                self.memory.init_tensors_block(self.bandit.total_trials_block)
                self._rollout_trajectory(block_idx, train=False)

            simulated_data = self.memory.get_simulated_data()
            self.logger.save_test(test_idx, simulated_data)

        self.logger.save_test_df()