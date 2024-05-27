"""
Test the agent on the task
"""
import argparse
import json
import os 
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import sys 

sys.path.append('../../meta-rl_utils/')
from agent import A2COptim
from environment import ContextualBanditTask
from utils import AgentMemoryOptim
from runner import Runner
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='../data/test/') # set folder path to load params.json
    parser.add_argument('--test_eps', type=int, default=50)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--task-id', type=int, default=0)

    args = parser.parse_known_args()[0]
    return args

def read_params(args):
    file_path = os.path.join(args.folder_path, 'params.json')

    with open(file_path, "r") as infile:
        data = json.load(infile)

    # Add items from dictionary to the args namespace
    for key, value in data.items():
        setattr(args, key, value)

def initialize_agent(args):

    if args.agent_model == 'Transformer':
        a2c = A2COptim(args.test_batch_size, 
                       args.hidden_size, 
                       args.input_size,  
                       args.learning_rate, 
                       args.is_cuda, 
                       num_layers = args.num_layers_transformer, 
                       n_heads = args.n_heads)

    return a2c


if __name__ == '__main__':
    """Argument parsing"""
    args = get_args()
    folder_path = args.folder_path
    read_params(args)
    args.is_cuda = True if torch.cuda.is_available() else False


    # Initialize the bandit environment
    debug = False #True if args.exp_name == 'test' else False
    bandit = ContextualBanditTask(batch_size=args.test_batch_size, 
                                  mode='test',
                                  success=args.success, 
                                  fail=args.fail,
                                  debugging=debug)

    # initialize agent
    agent = initialize_agent(args)
   
    writer = SummaryWriter(log_dir=folder_path+'/test/')

    memory = AgentMemoryOptim(is_cuda=args.is_cuda,
                              writer=writer,
                              return_coef=args.return_coef,
                              value_coef=args.value_coef,
                              return_fn=args.return_fn,)

    runner = Runner(agent,
                    bandit,
                    memory,
                    writer,
                    folder_path,
                    args.with_time)

    print('start testing')
    runner.test(args.test_eps,
                folder_path)
    print('finished testing')
