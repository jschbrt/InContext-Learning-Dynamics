"""
As Anthropic changed its API, this script was used to run the new Claude models. 
This script is used to generate the data for the partial task.
The data is saved in a csv file in the folder data/engine/exp/ with the name run_{run}.csv.
"""

import gym
import pandas as pd
import argparse
import sys
import ipdb
import os
from torch.distributions import Binomial
import envs # this triggers the registration
sys.path.append('../llm_utils')
from generate_functions import *

# num2words = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
num2words = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}
env = gym.make('palminteri2017-v0')

def reset(context, num_trials):
    arm1_, arm2_ = env.arms[int(context)-1].keys()
    instructions = f"You are going to visit four different casinos (named 1, 2, 3 and 4) {int(num_trials/4)} times each. Each casino owns two slot machines which all return either 1 or 0 dollars stochastically with different reward probabilities. "\
                    
    trials_left = f"Your goal is to maximize the sum of received dollars within all {num_trials} visits."

    history = ""
   
    question = f"You are now in visit {env.t + 1} playing in Casino {num2words[int(context)]}." \
        f" Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine <your answer>.\")"
    

    return instructions, history, trials_left, question

def step(history, prev_machine, action, t):
    # get reward and next context
    observation, reward, done, _ = env.step(env.arms[int(prev_machine)-1][action])
    next_machine = observation[0, 3]
    # print(observation, reward, done)
    
    if t==0:
        history =  "You have received the following amount of dollars when playing in the past: \n"\
    # update history based on current action and trialss
    history += f"- Machine {action} in Casino {num2words[int(prev_machine)]} delivered {float(reward)} dollars.\n"
    
    # update trials left
    trials_left = f"Your goal is to maximize the sum of received dollars within {env.max_steps} visits."
    arm1_, arm2_ = env.arms[int(next_machine)-1].keys()
    if Binomial(1, 0.5).sample() == 1:  #randomly change their order for prompt
        arm1_ , arm2_ = arm2_, arm1_
    question = f"You are now in visit {env.t+1} playing in Casino {num2words[int(next_machine)]}. " \
        f"Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine <your answer>\")."
    
    return history, trials_left, next_machine, question, done

# Run
parser = argparse.ArgumentParser()
# ONLY FOR NEW CLAUDE MODELS USE n- as prefix
parser.add_argument('--engines', nargs='+', default=['n-claude-instant-1.2']) #'n-claude-3-haiku-20240307', 'n-claude-2.1', 'n-claude-3-sonnet-20240229', 'n-claude-3-opus-20240229', 'n-claude-instant-1.2'
parser.add_argument('--num_runs', type=int, default=50) # 50 participants
# parser.add_argument('--num_trials', type=int, default=20)
args = parser.parse_args()
engines = args.engines
num_runs = args.num_runs
for engine in engines:
    print(f'Engine :------------------------------------ {engine} ------------------------------------')
    # Load LLM
    llm, Q_, A_ = get_llm(engine, max_tokens=2, temp=0.0, arms=('A', 'B'))
    act = llm.generate

    # If already some run_files in path, start from the last one
    start_run = 0
    if os.path.exists(f'./data/{engine}/exp'):
        start_run = len(os.listdir(f'./data/{engine}/exp'))
    for run in range(start_run, num_runs):
        print(f'Run: {run}')
        data = []
        done = False
        actions = []
        env.reset()
        num_trials = env.max_steps
        instructions, history, trials_left, question = reset(env.contexts[0, 0], env.max_steps)
        current_machine = env.contexts[0, 0]
        for t in range(num_trials):
            instruction = instructions + trials_left
            quest = history + "\n" if t > 0 else history
            quest = quest + question
            print(f'Trial: {t}')
            #print(prompt)
            #print("\n")
            # LLM acts
            action = act(instruction, quest, arms= [i for i in env.arms[int(current_machine)-1].keys()], temp=0.0)
            i = 0
            while action not in env.arms[int(current_machine)-1].keys():
                i += 1
                print(f'Invalid action: {action}')
                action = act(instruction, quest, temp=1.0)[0]
                if i >= 9:
                    arm1_, arm2_ = env.arms[int(current_machine)-1].keys()
                    wrong_text = f"{action} is not in Casino {num2words[int(current_machine)]}. Let's repeat. Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine _\")."
                    action = act(instruction, quest, wrong_text=wrong_text, wrong_action=action)
                
                if i == 10:
                    import ipdb; ipdb.set_trace()
                    print('---')

            action_int = int(env.arms[int(current_machine)-1][action]) #torch.tensor([int(action)])
            rewards = env.rewards[0, t, action_int].item()
            #print(action_int)
            # save values
            row = [run, t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), action_int,  rewards]
            data.append(row)
            if not done:
                # step into the next trial
                history, trials_left, current_machine, question, done = step(history, current_machine, action, t)
        df = pd.DataFrame(data, columns=['run', 'trial', 'casino', 'mean0', 'mean1', 'reward0', 'reward1', 'choice', 'rewards'])
        #print(df)
        os.makedirs(f'./data/{engine}/exp', exist_ok=True)
        df.to_csv(f'./data/{engine}/exp/run_' + str(run) + '.csv')
