"""
Utility functions for the full task.
"""
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
import os
import numpy as np

class AgentMemoryOptim:
    def __init__(self, 
                 with_time, 
                 is_cuda,
                 return_coef,
                 value_coef,
                 return_fn,
                 agency_test) -> None:
        """
        Internal Memory of the Advantage Actor Critic (A2C).
        """
        self.with_time = with_time
        self.is_cuda = is_cuda
        self.return_coef = return_coef
        self.value_coef = value_coef

        func_map = {
                    'discounted_return': self.compute_discounted_returns,
                    'discounted_return_exclude_observations': self.compute_discounted_returns_exclude_observations,
                    }
        self.returns_func = func_map[return_fn]

        agency_loss_fn = {
                    'mask_policy_value_loss': self.mask_policy_value_loss,
                    'no_mask': self.no_mask,
                    }

        self.agency_loss_fn = agency_loss_fn[agency_test]

    def init_tensors(self, episode_length, batch_size):
        """
        Tensors for saving temporary data are initalized after every 
        episode.

        Args:
            episode_length: the length of one rollout.
        """

        self.batch_size = batch_size
        self.episode_length = episode_length

        self.block_idx = self.create_tensor(init_value=-1.)
        self.feedback_type = self.create_tensor()
        self.reward_block_type = self.create_tensor()
        self.trials_idx = self.create_tensor()
        self.rewards = self.create_tensor(cuda=True)
        self.counter_rewards = self.create_tensor(cuda=True)
        self.actions = self.create_tensor(cuda=True)
        self.optimal_actions = self.create_tensor()
        self.cues = self.create_tensor(expand_dim=2, cuda=True) # we mark free/forced
        self.log_policies_a = self.create_tensor()
        self.value_fn = self.create_tensor(cuda=True)
        self.regrets = self.create_tensor()
        self.entropy = self.create_tensor()
        self.policy = self.create_tensor(expand_dim=2)
        self.time_matrix = self.create_tensor(expand_dim=1, cuda=True)

        # for one hot encoding
        self.one_hot_actions = self.create_tensor(expand_dim=2, cuda=True)
        self.one_hot_cues = self.create_tensor(expand_dim=2, cuda=True)
        self.one_hot_prev_cues = self.create_tensor(expand_dim=2, cuda=True)
        self.one_rewards = self.create_tensor(expand_dim=1, cuda=True)
        self.one_counter_rewards = self.create_tensor(expand_dim=1, cuda=True)

    def create_tensor(self, expand_dim=None, cuda=False, init_value=0.):
        """
        Creates zero tensors to free up space given a specific size.
        """
        if expand_dim:
            if cuda and self.is_cuda:
                return torch.full((self.batch_size, self.episode_length, expand_dim),init_value).cuda()
            else:
                return torch.full((self.batch_size, self.episode_length, expand_dim),init_value)
        else:
            if cuda and self.is_cuda:
                return torch.full((self.batch_size, self.episode_length), init_value).cuda()
            else:
                return torch.full((self.batch_size, self.episode_length), init_value)

    def init_tensors_block(self, block_length):

        self.block_length = block_length

        # for gradient propagation
        self.block_log_policies_a = self.create_block_tensor(cuda=True)
        self.block_value_fn = self.create_block_tensor(cuda=True)
        self.block_entropy = self.create_block_tensor(cuda=True)

    def create_block_tensor(self, cuda=False, init_value=0.):
        if cuda and self.is_cuda:
            return torch.full((self.batch_size, self.block_length), init_value).cuda()
        else:
            return torch.full((self.batch_size, self.block_length), init_value)


    def insert_cues(self, idx, trials_idx, block_idx, cues, feedback_type):
        self.cues[:,idx].copy_(cues)

        feedback_type = torch.tensor([feedback_type]*self.batch_size)
        self.feedback_type[:, idx].copy_(feedback_type)
    
        block_idx = torch.tensor([block_idx]*self.batch_size)
        self.block_idx[:, idx].copy_(block_idx)

        trials_idx = torch.tensor([trials_idx]*self.batch_size) 
        self.trials_idx[:, idx].copy_(trials_idx)

    def insert_data(self, idx, trials_idx, reward_block_type, rewards, counter_rewards, regrets, actions, optimal_actions, log_policy_as, baselines, entropy, policy):
        """
        Inserts data after each training episode.
        TODO: make more comprehensive
        """
        reward_block_type = torch.tensor(reward_block_type)

        self.reward_block_type[:, idx].copy_(reward_block_type)
        self.rewards[:, idx].copy_(rewards)
        self.counter_rewards[:, idx].copy_(counter_rewards)
        self.regrets[:, idx].copy_(regrets)
        self.actions[:, idx].copy_(actions)
        self.optimal_actions[:, idx].copy_(optimal_actions)
        self.log_policies_a[:, idx].copy_(log_policy_as.detach())
        self.value_fn[:, idx].copy_(baselines.detach())
        self.entropy[:, idx].copy_(entropy.detach())
        self.policy[:, idx, :].copy_(policy.detach())

        self.block_log_policies_a[:, trials_idx].copy_(log_policy_as)
        self.block_value_fn[:, trials_idx].copy_(baselines)
        self.block_entropy[:, trials_idx].copy_(entropy)

    def get_padded_data(self, idx, trials_idx, block_idx):
        # just return zero in the first run
        if trials_idx == 0:
            cues = self.cues[:, idx].unsqueeze(-1).clone()
            if self.is_cuda:
                rewards, counter_rewards, actions = (torch.zeros((self.batch_size,1)).cuda() for i in range(3))
                prev_cues = torch.zeros((self.batch_size,2)).cuda()
            else:
                rewards, counter_rewards, actions = (torch.zeros((self.batch_size,1)) for i in range(3))
                prev_cues = torch.zeros((self.batch_size,2))
            #times = torch.zeros(self.batch_size)

        else:
            #times = torch.tensor([idx]*self.batch_size)
            cues = self.cues[:, idx].unsqueeze(-1).clone()

            # idx-1 because we want to input the previous values
            rewards = self.rewards[:, idx-1].unsqueeze(-1).clone()
            counter_rewards = self.counter_rewards[:, idx-1].unsqueeze(-1).clone()
            actions = self.actions[:, idx-1].clone()
            prev_cues = self.cues[:, idx-1].unsqueeze(-1).clone()
        
        actions = actions.squeeze().long() # has to be converted to long to be used with one_hot
        actions = F.one_hot(actions, num_classes=2)
        cues = cues.squeeze().long()
        prev_cues = prev_cues.squeeze().long()

        # map actions
        self.one_hot_actions[:,idx,:] = actions
        self.one_rewards[:,idx,:] = rewards
        self.one_counter_rewards[:,idx,:] = counter_rewards
        self.one_hot_prev_cues[:,idx,:] = prev_cues
        self.one_hot_cues[:,idx, :] = cues

        mask1d = (self.block_idx == block_idx).unsqueeze(-1)
        mask = mask1d.expand(self.batch_size, self.episode_length, 2)

        one_hot_actions = self.one_hot_actions[mask].view(self.batch_size, -1, 2).clone()
        one_rewards = self.one_rewards[mask1d].view(self.batch_size, -1, 1).clone()
        one_counter_rewards = self.one_counter_rewards[mask1d].view(self.batch_size, -1, 1).clone()
        one_hot_prev_cues = self.one_hot_prev_cues[mask].view(self.batch_size, -1, 2).clone()
        one_hot_cues = self.one_hot_cues[mask].view(self.batch_size, -1, 2).clone()

        data = torch.cat((one_hot_cues, one_hot_prev_cues, one_hot_actions, one_rewards, one_counter_rewards), dim=2)

        if self.with_time:
            norm_time = idx / self.episode_length
            self.time_matrix[:,idx, :] = norm_time
            time_matrix = self.time_matrix[mask1d].view(self.batch_size, -1, 1).clone()
            data = torch.cat((data, time_matrix), dim=2)

        return data

    def get_simulated_data(self):
        
        # convert to cpu
        rewards = self.rewards.cpu()
        counter_rewards = self.counter_rewards.cpu()
        actions = self.actions.cpu()
        cues = self.cues.cpu()    
        saved_data = (self.block_idx, self.feedback_type, self.reward_block_type, self.trials_idx, rewards, counter_rewards, self.regrets, self.optimal_actions, actions, self.entropy, self.value_fn, self.policy)
        return cues, saved_data

    def compute_discounted_returns(self, rewards):
        """
        Computes the discounted returns at the end of 
        the training episode 
        """
        returns = torch.zeros_like(rewards)
        Return = torch.tensor([0.] * self.batch_size)
        for idx in reversed(range(rewards.shape[1])):
            Return = rewards[:, idx] +  self.return_coef * Return
            returns[:, idx] = Return
        return returns

    def compute_discounted_returns_exclude_observations(self, rewards, mask):
        """
        Computes the discounted returns at the end of 
        the training episode 
        """
        returns = torch.zeros_like(rewards)
        Return = torch.tensor([0.] * self.batch_size)
        for batch_idx in range(self.batch_size):
            for idx in reversed(range(rewards.shape[1])):
                if mask[batch_idx, idx] == False:
                    Return[batch_idx] = rewards[batch_idx, idx] +  self.return_coef * Return[batch_idx]
                returns[batch_idx, idx] = Return[batch_idx]
        return returns

    def a2c_loss(self, entropy_coef, block_idx):
        """
        Calculates the loss for the A2C
        """

        return self.agency_loss_fn(entropy_coef, block_idx)

    # case 1
    # we mask rewards and value_fn
    # we mask policy and value loss
    def mask_policy_value_loss(self, entropy_coef, block_idx):

        block_mask = self.block_idx == block_idx

        rewards = self.rewards[block_mask].cpu().view(self.batch_size, -1)
        value_fn = self.block_value_fn.cpu()

        cues1d = self.cues.sum(axis=2)
        mask = (cues1d == 1)[block_mask].view(self.batch_size, -1)
        mask = mask.cpu()

        log_policies_a = self.block_log_policies_a.cpu()
        entropy = self.block_entropy.cpu()


        # Mask the observations
        rewards = rewards.masked_fill(mask, 0)
        value_fn = value_fn.masked_fill(mask, 0)
        
        # either td_error or discounted_return 
        returns = self.returns_func(rewards)

        # calculate the advantage
        advantage = (returns - value_fn)

        # calculate the policy loss
        policy_loss = -(log_policies_a * advantage)
        policy_loss = policy_loss.masked_fill(mask, 0)
        policy_loss = policy_loss.sum()

        # calculate the value loss
        value_loss = advantage.pow(2)
        value_loss = value_loss.masked_fill(mask, 0)
        value_loss = value_loss.sum()

        entropy = entropy.masked_fill(mask,0) 
        entropy_loss = entropy.sum()

        loss = self.value_coef * value_loss.sum() + policy_loss.sum() - entropy_coef * entropy_loss 

        sum_rewards = rewards.detach().sum()
        return (loss, policy_loss, value_loss, entropy_loss, sum_rewards)

    # case 4 we do not mask anything
    def no_mask(self, entropy_coef, block_idx):

        block_mask = self.block_idx == block_idx

        rewards = self.rewards[block_mask].cpu().view(self.batch_size, -1)
        value_fn = self.block_value_fn.cpu()

        cues1d = self.cues.sum(axis=2)
        mask = (cues1d == 1)[block_mask].view(self.batch_size, -1)
        mask = mask.cpu()

        log_policies_a = self.block_log_policies_a.cpu()
        entropy = self.block_entropy.cpu()
            
        # either td_error or discounted_return
        returns = self.returns_func(rewards)

        # calculate the advantage
        advantage = (returns - value_fn)

        # calculate the policy loss
        policy_loss = -(log_policies_a * advantage)
        policy_loss = policy_loss.sum()

        # calculate the value loss
        value_loss = advantage.pow(2)
        value_loss = value_loss.sum()

        entropy_loss = entropy.sum()

        loss = self.value_coef * value_loss.sum() + policy_loss.sum() - entropy_coef * entropy_loss 

        sum_rewards = rewards.detach().sum()
        return (loss, policy_loss, value_loss, entropy_loss, sum_rewards)

class LogProgress:

    def __init__(self, 
                 folder_path, 
                 batch_size, 
                 episodes, 
                 train, 
                 plot_freq=-1, 
                 plot_window_size=2) -> None:
        """
        Logging class for the training and testing of the agent on the full feedback task.
        """
        self.folder_path = folder_path
        self.writer = SummaryWriter(log_dir=folder_path+'tb/')

        self.batch_size = batch_size
        self.eps = episodes
        self.train = train
        # init plot index
        self.temp_plot_idx = None
        self.is_within_plot_window = False

        self.plot_window_idx = 0
        self.plot_window_size = plot_window_size
        self.plot_freq = plot_freq

    def init_df(self, trial_per_eps):

        self.trials_per_eps = trial_per_eps

        if self.train:
            self.init_train_df()
            self.init_loss_df()

        else:
            self.init_simulation_df()

    def init_loss_df(self):
        loss_df_columns = ['plot_idx',
                                'train_eps_idx',
                                'loss',
                                'policy_loss',
                                'value_loss',
                                'entropy_sum'
                                ]
        df = pd.DataFrame(columns=loss_df_columns)
        df = df.astype({'plot_idx':int,
                        'train_eps_idx':int, 
                        'loss':float,
                        'policy_loss':float,
                        'value_loss':float,
                        'entropy_sum':float})
        self.loss_df_empty = df
        self.loss_df = self.loss_df_empty.copy()
        self.loss_df.drop(['train_eps_idx'], axis=1).to_csv(self.folder_path+'loss_df.csv', mode='w', index=False)

    def init_train_df(self):

        self.train_df_columns = [
            'plot_idx',
            'train_eps_idx',
            'block_idx',
            'block_feedback_type',
            'rewards',
            'forgone_rewards',
            'regrets',
            'perc_opt_actions',
        ]
        df = pd.DataFrame(columns=self.train_df_columns)
        df = df.astype({'plot_idx':int,
                        'train_eps_idx':int, 
                        'block_idx':int,
                        'block_feedback_type':int,
                        'rewards':float,
                        'forgone_rewards':float,
                        'regrets':float,
                        'perc_opt_actions':float})
        self.train_df_empty = df
        self.train_df = self.train_df_empty.copy()

        # Save train
        self.train_df.drop(['train_eps_idx'],axis=1).to_csv(self.folder_path+'train_df.csv', mode='w', index=False)

    def init_simulation_df(self):

        simulation_df_columns = [
            'test_eps_idx', # multiple participants per episode
            'batch_idx', # simulation of one participant
            'block_idx',
            'block_feedback_type',
            'block_reward_type',
            'trials_idx',
            'cues',
            'actions',
            'rewards',
            'forgone_rewards',
            'regrets',
            'opt_actions',
            'entropy',
            'value_fn',
            'policy_0',
            'policy_1',
        ]
        df = pd.DataFrame(columns=simulation_df_columns)
        df = df.astype({'test_eps_idx':int,
                        'batch_idx':int,
                        'block_idx':int,
                        'block_feedback_type':int,
                        'block_reward_type':int,
                        'trials_idx':int,
                        'cues':int,
                        'actions':int,
                        'rewards':float,
                        'forgone_rewards':float,
                        'regrets':float,
                        'opt_actions':int,
                        'entropy':float,
                        'value_fn':float,
                        'policy_0':float,
                        'policy_1':float,
                        })
        self.simulation_df = df

    def plot_save(self, eps_idx, simulated_data, agent=None, losses=None):
        
        if eps_idx == 0 or (eps_idx+1) % self.plot_freq == 0 or (eps_idx+1) == self.eps:
            if eps_idx == 0: 
                self.temp_plot_idx = 0
            else: 
                self.temp_plot_idx = (eps_idx+1)
            
            self.is_within_plot_window = True
            self.plot_window_idx = 0
        
        if self.is_within_plot_window:
            # calculate the mean per cue 
            self.train_mean_free_cue(eps_idx, simulated_data, losses)
            self.plot_window_idx +=1

            if self.plot_window_idx == self.plot_window_size or ((eps_idx+1) == self.eps):
                dfs = self.save_train()
                self.plot_train(dfs)
                self.save_model(eps_idx, agent)
                self.is_within_plot_window = False

    def save_test(self, eps_idx, simulated_data):
            
            cues, saved_data = simulated_data
            block_idx, feedback_type, reward_block_type, trials_idx, rewards, forgone_rewards, regrets, opt_actions, actions, entropy, value_fn, policy = [d.detach() for d in saved_data]
    
            # Create a one dimensional cue vector
            one_dim_cues = torch.zeros_like(cues[:,:,0]) # shape should be batch*eps_length
            idx = torch.where(cues[:,:,0] == 1)
            one_dim_cues[idx] = 1
            idx = torch.where(cues[:,:,1] == 1)
            one_dim_cues[idx] = 2 
            
            # Create dataframe
            df = pd.DataFrame({
                'test_eps_idx': np.repeat(eps_idx, self.trials_per_eps * self.batch_size),
                'batch_idx': np.repeat(np.arange(self.batch_size), self.trials_per_eps),
                'block_idx': block_idx.flatten(),
                'block_feedback_type': feedback_type.flatten(),
                'block_reward_type': reward_block_type.flatten(), 
                'trials_idx': trials_idx.flatten(),
                'cues': one_dim_cues.flatten(),
                'actions': actions.flatten(),
                'opt_actions': opt_actions.flatten(),
                'regrets': regrets.flatten(),
                'rewards': rewards.flatten(),
                'forgone_rewards': forgone_rewards.flatten(),
                'entropy': entropy.flatten(),
                'value_fn': value_fn.flatten(),
                'policy_0': policy[:,:,0].flatten(),
                'policy_1': policy[:,:,1].flatten()
            })
    
            self.simulation_df = pd.concat([self.simulation_df, df])

    def train_mean_free_cue(self, eps_idx, simulated_data, losses):

            cues, saved_data = simulated_data
            block_idx, feedback_type, reward_block_type, trials_idx, rewards, forgone_rewards, regrets, opt_actions, actions, entropy, value_fn, policy = [d.detach() for d in saved_data]

            # select only free trials
            summed_cue = cues.sum(axis=2)
            idx = torch.where(summed_cue == 0)

            free_block_idx = block_idx[idx].view(self.batch_size, -1)
            free_feedback_type = feedback_type[idx].view(self.batch_size, -1)

            free_cue_rewards = rewards[idx].view(self.batch_size, -1)
            free_cue_forgone_rewards = forgone_rewards[idx].view(self.batch_size, -1)
            free_cue_regrets = regrets[idx].view(self.batch_size, -1)
            free_cue_opt_actions = opt_actions[idx].view(self.batch_size, -1)
            free_cue_actions = actions[idx].view(self.batch_size, -1)

            # Average over batch
            free_cue_rewards = free_cue_rewards.mean(dim=0)
            free_cue_forgone_rewards = free_cue_forgone_rewards.mean(dim=0)
            free_cue_regrets = free_cue_regrets.mean(dim=0)
            free_cue_perc_opt_actions = torch.count_nonzero(free_cue_actions == free_cue_opt_actions, dim=0) / self.batch_size  # count nonzero over batch and then normalize
            
            # just select first dim as all are same
            free_block_idx = free_block_idx[0,:]
            free_feedback_type = free_feedback_type[0,:]

            # Create dataframe
            free_trials_per_eps = free_cue_rewards.shape[0] # 160 free trials
            free_trials_per_block = free_trials_per_eps // 4

            df = pd.DataFrame({
                'plot_idx': np.tile(self.temp_plot_idx, free_trials_per_eps), # for each cue
                'train_eps_idx': np.tile(eps_idx, free_trials_per_eps),
                'block_idx' : free_block_idx,
                'block_feedback_type': free_feedback_type, # should be 0 always
                'trials_idx': np.tile(np.arange(free_trials_per_block),4),
                'rewards': free_cue_rewards,
                'forgone_rewards': free_cue_forgone_rewards,
                'regrets': free_cue_regrets,
                'perc_opt_actions': free_cue_perc_opt_actions
            })
            self.train_df = pd.concat([self.train_df, df])

            # Loss DF
            loss, policy_loss, value_loss, entropy_loss, sum_rewards = losses
            df = pd.DataFrame([{
                'plot_idx': self.temp_plot_idx, # for each cue
                'train_eps_idx': eps_idx,
                'loss': loss.detach().numpy().item(),
                'policy_loss': policy_loss.detach().numpy().item(),
                'value_loss': value_loss.detach().numpy().item(),
                'entropy_sum': entropy_loss.detach().numpy().item()}])
            self.loss_df = pd.concat([self.loss_df, df])

    def save_train(self):
        # Save Train dataframe
        train_df = self.train_df.groupby(['plot_idx', 'trials_idx', 'block_idx', 'block_feedback_type']).mean(numeric_only=True).reset_index()
        train_df = train_df.drop(['train_eps_idx'], axis=1)
        train_df.to_csv(self.folder_path+'train_df.csv', mode='a', index=False, header=False)
        self.train_df = self.train_df_empty.copy()

        # Save Loss dataframe
        loss_df = self.loss_df.groupby(['plot_idx']).mean(numeric_only=True).reset_index()
        self.loss_df.to_csv(self.folder_path+'loss_df.csv', mode='a', index=False, header=False)
        self.loss_df = self.loss_df_empty.copy()
        return (train_df, loss_df)

    def save_test_df(self):
        # Save Test dataframe
        self.simulation_df.to_csv(self.folder_path+'simulation_df.csv', mode='w', index=False, header=True)

    def plot_train(self, dfs):
        # Log Perfomance
        train_df, loss_df = dfs
        tr_plt = train_df.groupby(['plot_idx', 'block_feedback_type']).mean(numeric_only=True)
        
        self.writer.add_scalar('tr.p_opt_a.full', tr_plt.perc_opt_actions.loc[self.temp_plot_idx, 1], self.temp_plot_idx)
        self.writer.add_scalar('tr.regrets.full', tr_plt.regrets.loc[self.temp_plot_idx, 1], self.temp_plot_idx)
        self.writer.add_scalar('tr.rewards.full', tr_plt.rewards.loc[self.temp_plot_idx, 1], self.temp_plot_idx)

        self.writer.add_scalar('tr.p_opt_a.partial', tr_plt.perc_opt_actions.loc[self.temp_plot_idx, 0], self.temp_plot_idx)
        self.writer.add_scalar('tr.regrets.partial', tr_plt.regrets.loc[self.temp_plot_idx, 0], self.temp_plot_idx)
        self.writer.add_scalar('tr.rewards.partial', tr_plt.rewards.loc[self.temp_plot_idx, 0], self.temp_plot_idx)

        # Log Losses
        loss_df = loss_df[loss_df.plot_idx == self.temp_plot_idx]
        self.writer.add_scalar("tr.x.loss.tot", loss_df.loss.item(), self.temp_plot_idx)
        self.writer.add_scalar("tr.x.loss.pol", loss_df.policy_loss.item(), self.temp_plot_idx)
        self.writer.add_scalar("tr.x.loss.val", loss_df.value_loss.item(), self.temp_plot_idx)
        self.writer.add_scalar("tr.x.loss.ent", loss_df.entropy_sum.item(), self.temp_plot_idx)

        self.writer.close()

    def save_model(self, eps_idx, agent, final=False):
        """Save model stages"""
        if not final: 
            pass
        #    os.makedirs(self.folder_path +'model_stages/', exist_ok=True)
        #    state = {'epoch': eps_idx, 
        #             'model': agent.net.state_dict(),
        #             'optimizer': agent.optimizer.state_dict(),
        #             'writer': self.writer}
        #    torch.save(state, f'{self.folder_path}model_stages/model_state_{eps_idx+1}.pt')
        else:
            state = {'epoch': eps_idx, 
                     'model': agent.net.state_dict(),
                     'optimizer': agent.optimizer.state_dict(), 
                     'writer': self.writer}
            if os.path.exists(self.folder_path + "model_state.pt"):
                torch.save(state, self.folder_path+"model_state_RESCUE.pt")
                assert ValueError('model exists!')
            else:
                torch.save(state, self.folder_path+"model_state.pt")