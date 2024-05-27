"""
Query the LLM for a given engine and save the simulated data in a csv file. 
This script is used to generate the data for the partial addition task with 2, 3 and 4 options. 
The data is saved in a csv file in the folder data/engine/ with the name exp_opt_{num_options}_{success}_{fail}.csv. 
The data is saved in the following format: run, trial, cue, action, reward, regret.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import random
import os
from torch.distributions import Binomial
from env import PartialTask
from utils import AgentMemoryOptim, LogProgress

# Run
parser = argparse.ArgumentParser()
parser.add_argument('--engines', nargs='+', default=['claude-1'])
parser.add_argument('--num_runs',  default=50) # 24 (trials) * 4 (modes) = 96
parser.add_argument('--success',  default=0.5)
parser.add_argument('--fail', default=-0.5)
parser.add_argument('--num_options', default=3)
args = parser.parse_args()  
engines = args.engines
success = float(args.success)
fail = float(args.fail)
num_options = int(args.num_options)
num_runs = int(args.num_runs)
sys.path.append('../llm_utils') # maybe not working
sys.path.append('llm_utils')
from generate_functions import *

def check_action(action, prompt, history):

    i = 0
    # debug action
    while action not in bandit.arms():
        i += 1
        print(f'Invalid action: {action}')
        action = act(prompt, temp=1.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
        if i >= 9:
            arm1_, arm2_ = bandit.arms().keys()
            text = prompt + " "+action + f".{Q_} Machine {action} is not in the casino. Let's repeat. Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine _\").\n\nA: Machine"
            action = act(text).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
        if i == 10:
            import ipdb; ipdb.set_trace()
            print('---')
        
    action_int = bandit.arms()[action]
    rewards, regrets = bandit.sample(action_int)
    
    memory.insert_data(trials_idx, 
                       rewards, 
                       regrets, 
                       action_int)

    if trials_idx == 0:
        history =  "You have received the following amount of dollars when playing in the past: \n"\

    ac = list(bandit.arms().keys())[action_int]

    history += f"- Machine {ac} in Casino {cue+1} delivered {float(rewards)} dollars.\n"

    bandit.step_forward()

    return history

for engine in engines:
    print(f'Engine :------------------------------------ {engine} ------------------------------------')
    # Load LLM
    llm, Q_, A_ = get_llm(engine, max_tokens=2, temp=0.0, arms=('A', 'B'))
    act = llm.generate

    # If already some run_files in path, start from the last one
    start_run = 0
    if os.path.exists(f'./data/{engine}/') and os.path.exists(f'./data/{engine}/exp_opt_{num_options}_{success}_{fail}.csv'):
            start_run = pd.read_csv(f'./data/{engine}/exp_opt_{num_options}_{success}_{fail}.csv')['run'].max()
            start_run = start_run + 1 if not np.isnan(start_run) else 0
    else:
        os.makedirs(f'./data/{engine}/', exist_ok=True)

    bandit = PartialTask(success=success, fail=fail, num_options=num_options)
    memory = AgentMemoryOptim()
    logger = LogProgress(folder_path=f'./data/{engine}/',
                         episodes=num_runs, name=f'exp_opt_{num_options}_{success}_{fail}')

    num_trials = bandit.total_trials

    logger.init_df(num_trials)

    for run in range(start_run, num_runs):
        bandit.setup()
        memory.init_tensors(num_trials)
        
        print(f'Run {run+1}/{num_runs}')
        
        if num_options == 2:
            instructions = f"You are going to visit four different casinos (named 1, 2, 3 and 4) {int(num_trials/4)} times each. Each casino owns two slot machines which all return either {success} or {fail} dollars stochastically with different reward probabilities. "\

            trials_left = f"Your goal is to maximize the sum of received dollars within all {num_trials} visits.\n"

            history = ""

            prompt=""

            for trials_idx in range(num_trials):

                if trials_idx % 10 == 0:
                    print(f'Trial {trials_idx+1}/{num_trials}\n')

                cue = bandit.cue()
                arm1_, arm2_ = bandit.arms().keys()

                memory.insert_cues(trials_idx, cue)

                if Binomial(1, 0.5).sample() == 1:  #randomly change their order for prompt
                    arm1_ , arm2_ = arm2_, arm1_

                question = f"{Q_} You are now in visit {trials_idx + 1} playing in Casino {cue+1}." \
                            f" Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine <your answer>.\")"\
                            f"{A_} Machine"

                prompt = instructions + trials_left + "\n" + history + "\n"+ question

                action = act(prompt, arms=[i for i in bandit.arms().keys()], temp=0.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
                
                #print(prompt+action)
                
                history = check_action(action, prompt, history)

        elif num_options == 3:
            instructions = f"You are going to visit three different casinos (named 1, 2 and 3) {int(num_trials/3)} times each. Each casino owns three slot machines which all return either {success} or {fail} dollars stochastically with different reward probabilities. "\

            trials_left = f"Your goal is to maximize the sum of received dollars within all {num_trials} visits.\n"

            history = ""

            prompt=""

            for trials_idx in range(num_trials):

                if trials_idx % 10 == 0:
                    print(f'Trial {trials_idx+1}/{num_trials}\n')

                cue = bandit.cue()
                arms = list(bandit.arms().keys())

                memory.insert_cues(trials_idx, cue)

                random.shuffle(arms)
                arm1_, arm2_, arm3_ = arms
                question = f"{Q_} You are now in visit {trials_idx + 1} playing in Casino {cue+1}." \
                            f" Which machine do you choose between Machine {arm1_}, Machine {arm2_} or Machine {arm3_}? (Give the answer in the form \"Machine <your answer>.\")"\
                            f"{A_} Machine"

                prompt = instructions + trials_left + "\n" + history + "\n"+ question

                action = act(prompt, arms=[i for i in bandit.arms().keys()], temp=0.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]

                print(prompt+action)

                history = check_action(action, prompt, history)

        elif num_options == 4:
            instructions = f"You are going to visit two different casinos (named 1 and 2) {int(num_trials/2)} times each. Each casino owns three slot machines which all return either {success} or {fail} dollars stochastically with different reward probabilities. "\

            trials_left = f"Your goal is to maximize the sum of received dollars within all {num_trials} visits.\n"

            history = ""

            prompt=""

            for trials_idx in range(num_trials):

                if trials_idx % 10 == 0:
                    print(f'Trial {trials_idx+1}/{num_trials}\n')

                cue = bandit.cue()
                arms = list(bandit.arms().keys())

                memory.insert_cues(trials_idx, cue)

                random.shuffle(arms)
                arm1_, arm2_, arm3_, arm4_ = arms

                question = f"{Q_} You are now in visit {trials_idx + 1} playing in Casino {cue+1}." \
                            f" Which machine do you choose between Machine {arm1_}, Machine {arm2_}, Machine {arm3_} or Machine {arm4_}? (Give the answer in the form \"Machine <your answer>.\")"\
                            f"{A_} Machine"

                prompt = instructions + trials_left + "\n" + history + "\n"+ question

                action = act(prompt, arms=[i for i in bandit.arms().keys()], temp=0.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
                
                print(prompt+action)

                history = check_action(action, prompt, history)

        simulated_data = memory.get_simulated_data()
        logger.save_test(run, simulated_data)