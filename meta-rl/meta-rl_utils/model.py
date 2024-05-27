import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    """
    Transformer architecture version of the A2C model.
    The Transformer is used to approximate both the policy and the value function.
    """
    def __init__(self, batch_size, input_size, hidden_size, is_cuda, num_layers, n_heads) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if is_cuda else "cpu")

        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.transformer_encoder_layer = TransformerEncoderLayer(
             d_model=hidden_size, nhead=n_heads, batch_first=True, dim_feedforward=128).to(self.device)
        
        self.transformer_encoder = TransformerEncoder(
             self.transformer_encoder_layer, num_layers=num_layers).to(self.device)
  
        # number of actions is 2 in our case
        self.actor = nn.Linear(in_features=hidden_size, out_features=2).to(self.device)
        self.critic = nn.Linear(in_features=hidden_size, out_features=1).to(self.device)

    def forward(self, x, idx):

        x = x[:, :idx+1, :]
        x = self.input_layer(x)

        output = self.transformer_encoder(x)
        # take only the prediction of the values related to the encoding of current cue (idx) 
        # for which an action should be chosen
        output = output[:, idx, :]

        # get the softmax over dim=1 (dim=0 is batch dim) -> returns 2 probabilities
        policy = F.softmax(self.actor(output), dim=1)
        # get the value function (for state a) -> returns a single value
        value = self.critic(output)

        return (policy, value.squeeze())
class LSTM(nn.Module):

    def __init__(self, batch_size, hidden_size, input_size, is_cuda) -> None:
        """
        LSTM and Linear functions used in the A2C Class.

        The LSTM approximates both the policy and the value function.
        The policy and the value function are produced by transforming
        the LSTM output with two seperate linear functions.

        Initialized as in paper [@wang2017]
        """
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTMCell(input_size=input_size,hidden_size=hidden_size)        
        self.init_hidden(is_cuda)

        #self.input_layer = nn.Linear(in_features=4, out_features=hidden_size)
        #self.middle_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.actor = nn.Linear(in_features=hidden_size,out_features=2) # number of actions is 2 in our case
        self.critic = nn.Linear(in_features=hidden_size,out_features=1)
        
    def init_hidden(self, is_cuda):
        """
        Initialized the hidden state with 0.
        The reset of `hidden_state` and `cell_state` is needed after each episode.
        """
        # TODO: check if is_cuda actually works
        if is_cuda:
             # set device to that of the underlying network (it does not matter, the device of which layer is queried)
            hidden_state = cell_state = torch.zeros((self.batch_size, self.lstm_cell.hidden_size), device=self.lstm_cell.weight_ih.device)
        else:
            hidden_state = cell_state = torch.zeros((self.batch_size, self.lstm_cell.hidden_size), requires_grad=True)
        
        self.hidden = (hidden_state, cell_state)

    def forward(self,x):
        """
        The forward pass is used for calculation of our
        policy and our value function.
        
        Args:
            x(p_a, p_r, t) (dim=4):
                p_a: previous action, one hot encoded (dim=2)
                p_r: previous reward (dim=1)
                t: time (dim=1)

        Returns:
            policy (dim=2): The policy values for x (probabilities over all actions)  
            value: (dim=1): The value function for x

        """
        
        # TODO: why no activation function after LSTM Cell? - Are the nonlinearities inside the LSTM sufficient? 
        #x = F.relu(self.input_layer(x))
        self.hidden = self.lstm_cell(x, self.hidden)
        hidden_state, _ = self.hidden
        #hidden_state = F.elu(self.middle_layer(hidden_state))
        policy = F.softmax(self.actor(hidden_state), dim=1) # get the softmax over dim=1 (dim=0 is batch dim) -> returns 2 probabilities 
        value = self.critic(hidden_state) # get the value function (for state a) -> returns a single value
        
        return (policy, value.squeeze())
