"""
Runner class for training and testing the A2C agent on the contextual bandit task.
"""
import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn
import os

sys.path.append('../../cognitive_models')
from agent import A2COptim
from utils import AgentMemoryOptim
from environment import ContextualBanditTask


class Runner():
    def __init__(self, 
                 agent:A2COptim,
                 bandit: ContextualBanditTask,
                 memory: AgentMemoryOptim,
                 writer,
                 folder_path, 
                 with_time, 
                 ) -> None:
        
        self.agent = agent
        self.memory = memory
        self.agent_model = agent.agent_model
        self.batch_size = agent.batch_size
        self.bandit = bandit
        self.writer = writer
        self.folder_path = folder_path
        self.with_time = with_time
        self.episode_length = bandit.max_steps

    def init_df(self, train:bool):

        df_columns = []

        if train:
            df_columns.extend([
                'plot_idx',
                'train_idx',
            ])

        else:
            df_columns.extend([
                'test_part_idx',
            ])

        df_columns.extend([
            'cues',
            'trials_idx',
            'rewards',
            'forgone_rewards',
            'regrets',
            'perc_opt_actions',
            'entropy',
            'value_fn',
            'policy_0',
            'policy_1',
        ])

        df_columns.extend([
            'uo_idx',
            'uo_regrets',
            'uo_rewards',
            'uo_forgone_rewards',
            'uo_perc_opt_actions',
        ])

        if train:
            # Save loss
            self.loss_df_columns = ['train_idx',
                            'loss',
                            'policy_loss',
                            'value_loss',
                            'entropy_sum'
                            ]
            self.loss_df = pd.DataFrame(columns=self.loss_df_columns)
            self.loss_df.to_csv(self.folder_path+'loss_df.csv', mode='w', index=False)
            # Save train
            self.train_df_columns = df_columns
            self.train_df = pd.DataFrame(columns=df_columns)
            self.train_df.drop('train_idx',axis=1).to_csv(self.folder_path+'train_df.csv', mode='w', index=False)
        
        else:
            # For saving test data
            self.test_df = pd.DataFrame(columns=df_columns)

            sim = ['test_part_idx',
                    'batch_idx',
                    'trials_idx',
                    'cues',
                    'actions',
                    'rewards',
                    'forgone_rewards',
                    'opt_actions',
                    'regrets',
                    'entropy',
                    'policy0',
                    'policy1',
                    'value_fn',
            ]

            self.simulation_df = pd.DataFrame(columns=sim) 

    def _rollout_trajectory(self, train=True):
        finished = False
        idx = 0
        if self.agent.is_cuda:
            time_matrix = torch.full((self.batch_size, self.episode_length, 1), 0.).cuda()
        else:
            time_matrix = torch.full((self.batch_size, self.episode_length, 1), 0.)
            
        while finished == False:
            self.memory.insert_cues(idx, self.bandit.cue())
            x = self.memory.get_padded_data(idx)
            if self.with_time:
                # normalized time
                norm_time = idx / self.episode_length
                time_matrix[:,idx, :] = norm_time
                x = torch.cat((x, time_matrix), dim=2)

            if train:
                actions, log_policy_a, entropy, value_fn, policy = self.agent.step(x, idx)
            else:
                with torch.no_grad():
                    actions, log_policy_a, entropy, value_fn, policy = self.agent.step(x, idx)
            # interact with environment
            rewards, forgone_rewards, regrets, optimal_actions, random_data, finished = self.bandit.sample(actions.cpu().numpy())
            self.memory.insert_data(idx, rewards, forgone_rewards, regrets, actions, optimal_actions, log_policy_a, value_fn, entropy, policy, random_data)
            idx+=1
        # get simulations
        simulated_data = self.memory.get_simulated_data()
        return simulated_data

    def training(self, 
                 train_eps, 
                 entropy_final_value, 
                 max_grad_norm):
        self.init_df(train=True) # For saving loss and training simulation data  
        self.temp_plot_idx = None
        for train_idx in range(train_eps):
            # Calculate decaying entropy coefficient
            entropy_coef = max(1-(train_idx/(0.5*train_eps)), entropy_final_value)

            # initializes / resets tensors
            self.memory.init_tensors(self.episode_length, self.batch_size) 
            
            # setup bandit
            self.bandit.setup()

            # Simulate one Training Episode
            simulated_data = self._rollout_trajectory()
            cues, trained_data, rand_data = simulated_data
            
            # Plot data"
            self.plot_save_training(train_eps, train_idx, cues, trained_data, rand_data, plot_freq=100)
            
            # ensure all gradients are 0
            self.agent.optimizer.zero_grad()
            
            # Calculate loss, propagate back, clip gradients and then do optimization on weights
            loss, policy_loss, value_loss, entropy_sum = self.memory.a2c_loss(train_idx, entropy_coef) 
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.net.parameters(), max_grad_norm)
            self.agent.optimizer.step()
            """ Save loss data """ 
            if (train_idx+1) % 50_000 == 0 or train_idx==0 or (train_idx+1)==train_eps:
                df = pd.DataFrame([{'train_idx':train_idx,
                    'loss': loss.detach().numpy(),
                    'policy_loss': policy_loss.detach().numpy(),
                    'value_loss': value_loss.detach().numpy(),
                    'entropy_sum': entropy_sum.detach().numpy()}])
                self.loss_df = pd.concat([self.loss_df, df])
            if (train_idx+1) % 100_000 == 0:
                self.loss_df.to_csv(self.folder_path+'loss_df.csv', mode='a', index=False, header=False)
                self.loss_df = pd.DataFrame(columns=self.loss_df_columns)
            """Save model stages"""
            if (train_idx+1) % 50_000 == 0 or train_idx == 0 or (train_idx+1)==train_eps:
                os.makedirs(self.folder_path +'model_stages/', exist_ok=True)
                state = {'epoch': train_idx + 1, 'model': self.agent.net.state_dict(),
                         'optimizer': self.agent.optimizer.state_dict(), 'writer': self.writer}
                torch.save(state, f'{self.folder_path}model_stages/model_state_{train_idx+1}.pt')

        """Save the best hyperparameters for test."""
        state = {'epoch': train_idx + 1, 'model': self.agent.net.state_dict(),
             'optimizer': self.agent.optimizer.state_dict(), 'writer': self.writer, }
        if os.path.exists(self.folder_path + "model_state.pt"):
            assert ValueError('model exists!')
        else:
            torch.save(state, self.folder_path+"model_state.pt")

    def plot_save_training(self, train_eps, train_idx, cues, trained_data, rand_data, plot_freq):
        plot_window_start = 0
        plot_window_end = plot_freq - 1
        is_end_of_training = train_idx+1 == train_eps
        is_within_plot_window = (train_idx+1) % plot_freq >= plot_window_start and (train_idx+1) % plot_freq <= plot_window_end

        if is_within_plot_window:
            if train_idx==0 or (train_idx+1) % plot_freq == 0 or is_end_of_training:
                self.temp_plot_idx = train_idx+1

            # calculate the mean per cue 
            self.per_cue_mean(train_idx, cues, trained_data, rand_data, train=True)

        # Save training data
        if (train_idx) % plot_freq == plot_window_end or is_end_of_training:  
            df = self.save_training_data()
            self.plot_training_data(df)

    def save_training_data(self):
        df = self.train_df.astype({'plot_idx':int,'train_idx':int,'cues':int,'trials_idx':int, 'uo_idx':int})
        df_o = df.groupby(['plot_idx', 'cues','trials_idx']).mean().drop(['train_idx'], axis=1).reset_index()
        df_o = df_o.drop(df.filter(regex='uo').columns,axis=1)

        df_uo = df.groupby(['plot_idx', 'uo_idx']).mean().reset_index()
        df_uo = df_uo.filter(regex='uo')
        df = pd.concat((df_o, df_uo), axis=1)
        df.to_csv(self.folder_path+'train_df.csv', mode='a', index=False, header=False)
        self.train_df = pd.DataFrame(columns=self.train_df_columns)
        return df

    def plot_training_data(self, df):
        df = df.drop(df.filter(regex='uo').columns,axis=1)
        df = df.groupby(['plot_idx', 'cues','trials_idx']).mean()
        df = df.groupby(level='plot_idx').mean()
        self.writer.add_scalar('train.perc_optimal_actions', df.perc_opt_actions.loc[self.temp_plot_idx], self.temp_plot_idx)
        self.writer.add_scalar('train.regret', df.regrets.loc[self.temp_plot_idx], self.temp_plot_idx)
        self.writer.close()

    def per_cue_mean(self, idx, cues, trained_data, rand_data, train:bool):

        m_regrets_c, m_rewards_c, m_forgone_rewards_c, perc_opt_actions_c, m_entropy_c, m_value_fn_c, m_policy_0_c, m_policy_1_c = ([] for i in range(8))
        rewards, forgone_rewards, regrets, opt_actions, actions, entropy, value_fn, policy = [d.detach() for d in trained_data]
        
        if not train:
            df = pd.DataFrame({
                    'batch_idx': np.repeat(np.arange(self.batch_size), 96),
                    'trials_idx': np.tile(np.arange(96), self.batch_size),
                    'cues': cues.flatten(),
                    'actions': actions.flatten(),
                    'rewards': rewards.flatten(),
                    'forgone_rewards': forgone_rewards.flatten(),
                    'opt_actions': opt_actions.flatten(),
                    'entropy': entropy.flatten(),
                    'policy0': policy[:,:,0].flatten(),
                    'policy1': policy[:,:,1].flatten(),
                    'value_fn': value_fn.flatten(),
                    'regrets': regrets.flatten() 
                })
            
            if self.test_case.startswith('exp'):
                df2 = pd.DataFrame({
                    'test_part_idx': np.repeat(self.p_idx,self.batch_size*96),
                })
            else:
                df2 = pd.DataFrame({
                    'test_part_idx': np.repeat(idx,self.batch_size*96),
                })
            df = pd.concat([df2, df], axis=1)

            self.simulation_df = pd.concat([self.simulation_df, df])

        trials_idx = np.tile(np.arange(24), 4).reshape(4,24)
        cues_idx = np.repeat(range(4),24)
        
        for cue in range(4):    
            cue_idx = torch.where(cues == cue)
            regrets_c = regrets[cue_idx] 
            rewards_c = rewards[cue_idx]
            forgone_rewards_c = forgone_rewards[cue_idx]
            actions_c = actions[cue_idx]
            opt_actions_c = opt_actions[cue_idx]
            entropy_c = entropy[cue_idx]
            value_fn_c = value_fn[cue_idx]
            policy_0_c = policy[cue_idx[0],cue_idx[1],0]
            policy_1_c = policy[cue_idx[0],cue_idx[1],1]

            # calculate the average over batch dimension
            batch_size = cues.shape[0]

            m_regrets_c.append(regrets_c.view(batch_size,-1).mean(dim=0)) # over episode, 24 times show, averaged over batch, episodes
            m_rewards_c.append(rewards_c.view(batch_size,-1).mean(dim=0)) 
            m_forgone_rewards_c.append(forgone_rewards_c.view(batch_size,-1).mean(dim=0))
            perc_opt_actions_c.append(torch.count_nonzero(actions_c.view(batch_size,-1) == opt_actions_c.view(batch_size, -1), dim=0) / batch_size)
            m_entropy_c.append(entropy_c.view(batch_size,-1).mean(dim=0))
            m_value_fn_c.append(value_fn_c.view(batch_size,-1).mean(dim=0))
            m_policy_0_c.append(policy_0_c.view(batch_size,-1).mean(dim=0))
            m_policy_1_c.append(policy_1_c.view(batch_size,-1).mean(dim=0))

        df = pd.DataFrame({
        'cues': cues_idx,
        'trials_idx': trials_idx.flatten(),
        'rewards': torch.stack(m_rewards_c).flatten(),
        'forgone_rewards': torch.stack(m_forgone_rewards_c).flatten(),
        'regrets': torch.stack(m_regrets_c).flatten(),
        'perc_opt_actions': torch.stack(perc_opt_actions_c).flatten(),
        'entropy': torch.stack(m_entropy_c).flatten(),
        'value_fn': torch.stack(m_value_fn_c).flatten(),
        'policy_0': torch.stack(m_policy_0_c).flatten(),
        'policy_1': torch.stack(m_policy_1_c).flatten(),
        'uo_idx': np.arange(0,96, dtype=int),
        'uo_regrets': regrets.mean(dim=0),
        'uo_rewards': rewards.mean(dim=0),
        'uo_forgone_rewards': forgone_rewards.mean(dim=0),
        'uo_perc_opt_actions': torch.count_nonzero(actions == opt_actions, dim=0) / batch_size,
        })
        if train:
            df1 = pd.DataFrame({
                'plot_idx': np.repeat(self.temp_plot_idx, 96),
                'train_idx': np.repeat(idx, 96),
            })
            df = pd.concat([df,df1], axis=1)
        else:
            if self.test_case.startswith('exp'):
                df1 = pd.DataFrame({
                            'test_part_idx': np.repeat(self.p_idx, 96),
                })
            else:
                df1 = pd.DataFrame({
                            'test_part_idx': np.repeat(idx, 96),
                })
            df = pd.concat([df,df1], axis=1)

        if train:
            self.train_df = pd.concat([self.train_df, df])
        else:
            self.test_df = pd.concat([self.test_df, df])

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
            checkpoint = torch.load(folder_path+"model_state.pt")
        else:
            checkpoint = torch.load(folder_path+"model_state.pt", map_location=torch.device('cpu'))
        self.agent.net.load_state_dict(checkpoint['model'])

        # Init dataframe
        self.init_df(train=False)
        self.test_case = 'normal'

        for test_idx in range(test_eps):

            self.memory.init_tensors(self.episode_length, self.batch_size)
            # setup bandit
            self.bandit.setup()

            simulated_data = self._rollout_trajectory(train=False)
            cues, trained_data, rand_data = simulated_data

            # save test
            self.per_cue_mean(test_idx, cues, trained_data, rand_data, train=False)
        self.test_df.to_csv(self.folder_path+'test_df.csv', mode='x', index=False)
        self.simulation_df.to_csv(self.folder_path+'simulation_df.csv', mode='x', index=False)
