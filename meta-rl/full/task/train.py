"""
This script is used to train the agent on the task.
"""
import argparse
from datetime import datetime
import json
import os 
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import torch

import sys
sys.path.append('../../meta-rl_utils/')
from agent import A2COptim
from environment import CompleteTask
from utils import AgentMemoryOptim, LogProgress
from runner import Runner

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--train_eps', type=int, default=10)

    parser.add_argument('--num_layers_transformer', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--return_coef', type=float, default=0.8)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--max_grad_norm', type=float, default=50)
    
    parser.add_argument('--entropy_final_value', type=float, default=0)
    parser.add_argument('--return_fn', type=str, default='discounted_return')
    parser.add_argument('--agency_test', type=str, default='mask_policy_value_loss')
    parser.add_argument('--with_time', type=str, default='true')
    parser.add_argument('--env_mode', type=str, default='train')
    parser.add_argument('--success', type=int, default=1)
    parser.add_argument('--fail', type=int, default=-1)
    parser.add_argument('--feedback', type=str, default="full")
    parser.add_argument('--agency', type=str, default="forced")
    parser.add_argument('--agent_model', type=str, default='Transformer')

    parser.add_argument('--task-id', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

def make_directories(args):

    folder_path = f'../data/{args.exp_name}/'#/{args.model_name}_{args.date_time}/'
    
    args.folder_path = folder_path

    if args.exp_name == 'test':
        os.makedirs(folder_path, exist_ok=True)
    else:
        os.makedirs(folder_path, exist_ok=False) 

def save_params(args):
    dict_args = vars(args)
    with open(args.folder_path+"params.json", "w") as outfile:
        json.dump(dict_args, outfile)

def set_input_size(args):
    args.input_size = 7 # cue(one_hot [2]), prev_cue [2], prev_action (one_hot [2]), prev_reward 

    if args.with_time == 'true':
        args.input_size += 1
        args.with_time = True
    else:
        args.with_time = False

    if args.feedback == 'full':
        args.input_size += 1 # add dim if we concat prev_reward and counter_reward

def get_datetime(args):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d_%H:%M")
    args.date_time = date_time

def initialize_agent(args):

    if args.agent_model == 'Transformer':
        a2c = A2COptim(args.batch_size, 
                       args.hidden_size, 
                       args.input_size,  
                       args.learning_rate, 
                       args.is_cuda, 
                       num_layers = args.num_layers_transformer, 
                       n_heads = args.n_heads)

    return a2c

def free_gpu_cache():

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

if __name__ == '__main__':
    """Argument parsing"""
    args = get_args()
   
    get_datetime(args)
    set_input_size(args)
    args.is_cuda = True if torch.cuda.is_available() and args.agent_model == 'Transformer' else False
    
    if args.is_cuda:
        free_gpu_cache()  


    # Initialize the bandit environment
    debug = True if args.exp_name == 'test' else False
    bandit = CompleteTask(batch_size=args.batch_size, 
                         train=True,
                         success=args.success, 
                         fail=args.fail,
                         debugging=debug)
    # save env_params
    args.total_trials = bandit.total_trials

    # initialize agent
    args.hidden_size = args.input_size * 8
    agent = initialize_agent(args)

    make_directories(args)
    save_params(args)

    # initialize memory
    memory = AgentMemoryOptim(args.with_time, 
                              args.is_cuda,
                              args.return_coef,
                              args.value_coef,
                              args.return_fn,
                              args.agency_test)

    # configure logging
    logger = LogProgress(args.folder_path, 
                         args.batch_size,
                         args.train_eps,
                         train=True,
                         plot_freq=1000,
                         plot_window_size=30)

    # initialize training
    runner = Runner(agent,
                    bandit, 
                    memory,
                    logger)
    
    # start training
    print('start training')
    runner.training(args.train_eps, 
                    args.entropy_final_value, 
                    args.max_grad_norm)
    print('finished training')