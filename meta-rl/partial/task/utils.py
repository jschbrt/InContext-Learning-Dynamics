"""
Utility functions for the partial task.
"""
import torch
import torch.nn.functional as F

class AgentMemoryOptim:
    def __init__(self, 
                 is_cuda, 
                 writer,
                 return_coef,
                 value_coef,
                 return_fn) -> None:
        """
        Internal Memory of the Advantage Actor Critic (A2C).
        """
        self.is_cuda = is_cuda
        self.writer = writer
        self.return_coef = return_coef
        self.value_coef = value_coef
        self.return_fn = return_fn

    def init_tensors(self, episode_length, batch_size, with_random=False):
        """
        Tensors for saving temporary data are initalized after every 
        episode.

        Args:
            episode_length: the length of one rollout.
        """

        self.batch_size = batch_size
        self.episode_length = episode_length
        self.with_random = with_random

        self.rewards = self.create_tensor(cuda=True)
        self.counter_rewards = self.create_tensor(cuda=True)
        self.actions = self.create_tensor(cuda=True)
        self.optimal_actions = self.create_tensor()
        self.cues = self.create_tensor(cuda=True)
        self.log_policies_a = self.create_tensor()
        self.value_fn = self.create_tensor()
        self.discounted_returns = self.create_tensor()
        self.regrets = self.create_tensor()
        self.entropy = self.create_tensor()
        self.policy = self.create_tensor(expand_dim=2)
        # for one hot encoding
        self.one_hot_actions = self.create_tensor(expand_dim=2, cuda=True)
        self.one_hot_cues = self.create_tensor(expand_dim=4, cuda=True)
        self.one_hot_rewards = self.create_tensor(expand_dim=1, cuda=True)
        self.one_hot_counter_rewards = self.create_tensor(expand_dim=1, cuda=True)

        if self.with_random:
            self.rand_rewards = self.create_tensor()
            self.rand_regrets = self.create_tensor()
            self.rand_optimal_actions = self.create_tensor()
            self.rand_actions = self.create_tensor()

    def create_tensor(self, expand_dim=None, cuda=False):
        """
        Creates zero tensors to free up space given a specific size.
        """
        if expand_dim:
            if cuda and self.is_cuda:
                return torch.full((self.batch_size, self.episode_length, expand_dim),0.).cuda()
            else:
                return torch.full((self.batch_size, self.episode_length, expand_dim),0.)
        else:
            if cuda and self.is_cuda:
                return torch.full((self.batch_size, self.episode_length), 0.).cuda()
            else:
                return torch.full((self.batch_size, self.episode_length), 0.)

    def insert_cues(self, idx, cues):
        self.cues[:,idx].copy_(cues)

    def insert_data(self, idx, rewards, counter_rewards, regrets, actions, optimal_actions, log_policy_as, baselines, entropy, policy, random_data):
        """
        Inserts data after each training episode.
        TODO: make more comprehensive
        """
        self.rewards[:, idx].copy_(rewards)
        self.counter_rewards[:, idx].copy_(counter_rewards)
        self.regrets[:, idx].copy_(regrets)
        self.actions[:, idx].copy_(actions)
        self.optimal_actions[:, idx].copy_(optimal_actions)
        self.log_policies_a[:, idx].copy_(log_policy_as)
        self.value_fn[:, idx].copy_(baselines)
        self.entropy[:, idx].copy_(entropy)
        self.policy[:, idx, :].copy_(policy)

        if self.with_random:
            # insert random data
            rand_rewards, rand_regrets, rand_optimal_actions, rand_actions = random_data
            self.rand_rewards[:, idx].copy_(rand_rewards)
            self.rand_regrets[:, idx].copy_(rand_regrets)
            self.rand_optimal_actions[:, idx].copy_(rand_optimal_actions) 
            self.rand_actions[:, idx].copy_(rand_actions)

    def get_padded_data(self, idx, with_counter_reward=False):
        """
        TODO: optimize so that one_hot encoding is only applied for one batch and added to a tensor with other 
        retrieves data for the A2C forward pass
        """
        # just return zero in the first run
        if idx == 0:
            cues = self.cues[:, idx].unsqueeze(-1).clone()
            if self.is_cuda:
                rewards, counter_rewards, actions = (torch.zeros((self.batch_size,1)).cuda() for i in range(3))
            else:
                rewards, counter_rewards, actions = (torch.zeros((self.batch_size,1)) for i in range(3))
            #times = torch.zeros(self.batch_size)

        else:
            #times = torch.tensor([idx]*self.batch_size)
            cues = self.cues[:, idx].unsqueeze(-1).clone()

            # idx-1 because we want to input the previous values
            rewards = self.rewards[:, idx-1].unsqueeze(-1).clone()
            counter_rewards = self.counter_rewards[:, idx-1].unsqueeze(-1).clone()
            actions = self.actions[:, idx-1].clone()
        
        actions = actions.squeeze().long() # has to be converted to long to be used with one_hot
        actions = F.one_hot(actions, num_classes=2)
        cues = cues.squeeze().long()
        cues = F.one_hot(cues, num_classes=4)

        if idx == 0: 
            self.one_hot_actions[:,idx,:] = actions
            self.one_hot_rewards[:,idx,:] = rewards
            self.one_hot_counter_rewards[:,idx,:] = counter_rewards
        else: 
            self.one_hot_actions[:,idx-1,:] = actions
            self.one_hot_rewards[:,idx-1,:] = rewards
            self.one_hot_counter_rewards[:,idx-1,:] = counter_rewards

        self.one_hot_cues[:,idx,:] = cues

        #TODO only for with_counter_reward == 0, batch_size!=1 
        if with_counter_reward == 0:
            #if self.batch_size == 1:
            #    data = torch.cat((cues, actions, rewards)).view(1,-1)
            #else:
            data = torch.cat((self.one_hot_cues, self.one_hot_actions, self.one_hot_rewards), dim=2)
        elif with_counter_reward == 1: 
            if self.batch_size == 1:
                data = torch.cat((cues, actions, counter_rewards)).view(1,-1)
            else:
                data = torch.cat((cues, actions, counter_rewards.view(-1,1)), dim=1)
        elif with_counter_reward == 2: 
            if self.batch_size == 1:
                data = torch.cat((cues, actions, rewards, counter_rewards)).view(1,-1)
            else:
                data = torch.cat((cues, actions, rewards, counter_rewards), dim=2)
        #data = torch.cat((times.view(-1,1), cues, actions, rewards.view(-1,1)), dim=1) # version with time
        #data = torch.cat((cues, actions, rewards)).view(1,-1) # for batch == 1
        return data

    def get_simulated_data(self):

        self.rewards = self.rewards.cpu()
        self.counter_rewards = self.counter_rewards.cpu()
        self.actions = self.actions.cpu()
        self.cues = self.cues.cpu()

        if self.with_random:
            rand_data = (self.rand_rewards, self.rand_regrets, self.rand_optimal_actions, self.rand_actions)
            trained_data = (self.rewards, self.counter_rewards, self.regrets, self.optimal_actions, self.actions, self.entropy, self.value_fn, self.policy)
            return self.cues, trained_data, rand_data
        else:
            trained_data = (self.rewards, self.counter_rewards, self.regrets, self.optimal_actions, self.actions, self.entropy, self.value_fn, self.policy)
            rand_data = False
            return self.cues, trained_data, rand_data

    def compute_discounted_returns(self, return_coef):
        """
        Computes the discounted returns at the end of 
        the training episode 
        """
        Return = torch.tensor([0.] * self.batch_size)
        for idx in reversed(range(self.episode_length)):
            Return = self.rewards[:, idx] +  return_coef * Return
            self.discounted_returns[:, idx] = Return

    def compute_td_error(self, return_coef):
        for idx in range(self.episode_length):
            if idx == self.episode_length-1:
                self.discounted_returns[:, idx] = self.rewards[:, idx]
            else:
                self.discounted_returns[:, idx] = self.rewards[:, idx] + return_coef * self.value_fn[:, idx+1]

    def a2c_loss(self, train_idx, entropy_coef):
        """
        Calculates the loss for the A2C
        """
        if self.return_fn == 'discounted_return':
            self.compute_discounted_returns(self.return_coef)
        elif self.return_fn == 'td_error':
            self.compute_td_error(self.return_coef) # different way of calculating discounted values
        else:
            assert(ValueError('Use either: "discounted_return" or "td_error"'))
        advantage = (self.discounted_returns - self.value_fn)

        policy_loss = -(self.log_policies_a * advantage).sum()
        value_loss = advantage.pow(2).sum()
        #Huber loss didn't work
        #value_loss = F.huber_loss(self.value_fn, self.discounted_returns, reduction='sum')
        loss = self.value_coef * value_loss + policy_loss - entropy_coef * self.entropy.sum() 
        if train_idx % 1000 == 0 or train_idx+1==500000:
            self.log_progress(train_idx, loss, policy_loss, value_loss, self.entropy.sum())
        return loss, policy_loss, value_loss, self.entropy.sum()

    def log_progress(self, train_idx, loss, actor_loss, critic_loss, episode_entropy):
        """
        Logs the training progress for one rollout.
        """
        if self.writer is not None:
            # log Losses
            self.writer.add_scalar("train.loss.total", loss.item(), train_idx)
            self.writer.add_scalar("train.loss.policy", actor_loss.item(), train_idx) # mean
            self.writer.add_scalar("train.loss.value function", critic_loss.item(), train_idx)
            self.writer.add_scalar("train.loss.entropy", episode_entropy, train_idx) 
            # Log Perfomance
            sum_rewards = self.rewards.detach().sum()
            self.writer.add_scalar("train.perf.rewards", sum_rewards, train_idx)
            self.writer.close()
            #mean_value = self.value_fn.detach().mean()
            #self.writer.add_scalar("train.perf.value", mean_value, train_idx)


