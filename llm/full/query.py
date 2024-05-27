"""
Query the LLM for a given engine and save the simulated data in a csv file. 
This script is used to generate the data for the full feedback task.
The data is saved in a csv file in the folder data/engine/ with the name sim.csv.
"""

import pandas as pd
import argparse
import sys
import ipdb
import os
from torch.distributions import Binomial
from env import CompleteTask
from utils import AgentMemoryOptim, LogProgress

# Run
parser = argparse.ArgumentParser()
parser.add_argument('--engines', nargs='+', default=['debug'])
parser.add_argument('--num_runs',  default=2) # 24 (part) * 4 (sess) = 96
args = parser.parse_args()  
engines = args.engines
num_runs = int(args.num_runs)
sys.path.append('../llm_utils') # maybe not working
sys.path.append('llm_utils')
from generate_functions import *
for engine in engines:
    print(f'Engine :------------------------------------ {engine} ------------------------------------')
    # Load LLM
    llm, Q_, A_ = get_llm(engine, max_tokens=2, temp=0.0, arms=('A', 'B'))
    act = llm.generate

    # If already some run_files in path, start from the last one
    start_run = 0
    if os.path.exists(f'./data/{engine}/') and os.path.exists(f'./data/{engine}/sim.csv'):
            start_run = pd.read_csv(f'./data/{engine}/sim.csv')['run'].max()
            start_run = start_run + 1 if not np.isnan(start_run) else 0
    else:
        os.makedirs(f'./data/{engine}/', exist_ok=True)

    bandit = CompleteTask()
    memory = AgentMemoryOptim()
    logger = LogProgress(folder_path=f'./data/{engine}/',
                         episodes=num_runs)

    logger.init_df(bandit.total_trials)

    for run in range(start_run, num_runs):
        
        bandit.setup_blocks()
        memory.init_tensors(bandit.total_trials)
        
        for block_idx in range(bandit.number_of_blocks):
            print(f'Run {run+1}/{num_runs} - Block {block_idx+1}/{bandit.number_of_blocks}')
            bandit.start_block(block_idx, train=False)

            # reset block

            arm1_, arm2_ = bandit.arms.keys()

            instructions = f"You will visit a casino {bandit.total_trials_block} times. "\
                           f"The casino has two slot machines that stochastically return either 1 or -1 with different reward probabilities. " \
                           f"You can only interact with one slot machine per visit. " \
                           f"Half of the time you visit the casino, you can play, the other someone else is playing. "

            if bandit.feedback_type == 1: # full feedback
                instructions += f"During visits where you, can play, you'll earn points from the chosen machine. You'll also learn what points would have been earned had the other machine been selected. " \
                                f"During visits where someone else is playing, you'll learn what points are earned on the chosen and what points would have been earned had the other machine been selected. " \
                                f"Nevertheless, you only accumulate points from the machine you choose to play. "
            else:
                instructions += f"During visits where you, can play, you'll earn points from the chosen machine. " \
                                f"During visits where someone else is playing, you'll learn what points are earned on the chosen machine. " \

            trials_left = f"Your goal is to maximize the total amount of points you receive in all {bandit.number_free} visits you can play."
            
            history = ""


            prompt=""

            for trials_idx in range(bandit.total_trials_block):
                idx = bandit.total_trials_idx
                memory.insert_cues(idx, trials_idx, block_idx, bandit.cue(trials_idx))

                forced = (bandit.cue(trials_idx) == 1)
                if not forced:

                    if Binomial(1, 0.5).sample() == 1:  #randomly change their order for prompt
                        arm1_ , arm2_ = arm2_, arm1_

                    question = f"{Q_} You are now in visit {bandit.step_in_block + 1}." \
                               f" Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine <your answer>.\")"\
                               f"{A_} Machine"

                    prompt = instructions + trials_left + "\n" + history + "\n"+ question
                    #print(prompt, end=" ")

                    action = act(prompt, arms=[i for i in bandit.arms.keys()], temp=0.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
                    i = 0
                    # debug action
                    while action not in bandit.arms:
                        i += 1
                        print(f'Invalid action: {action}')
                        action = act(prompt, temp=1.0).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
                        if i >= 9:
                            arm1_, arm2_ = bandit.arms
                            text = prompt + " "+action + f".{Q_} Machine {action} is not in the casino. Let's repeat. Which machine do you choose between Machine {arm1_} and Machine {arm2_}? (Give the answer in the form \"Machine _\").\n\nA: Machine"
                            action = act(text).replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')[0]
                        if i == 10:
                            import ipdb; ipdb.set_trace()
                            print('---')
                    
                    action_int = bandit.arms[action] #torch.tensor([int(action)])
                
                else:
                    action_int = None # will be replaced by forced action

                #print(action)
                #print("\n")

                actions_replaced, rewards, forgone_rewards, regrets, optimal_actions = bandit.sample(action_int, trials_idx)
                
                memory.insert_data(idx, 
                                bandit.feedback_type, 
                                bandit.reward_block_type, 
                                rewards, 
                                forgone_rewards, 
                                regrets, 
                                actions_replaced, 
                                optimal_actions,
                                prompt)

                if trials_idx == 0:
                    history = "During your previous visits you have observed the following: \n"\

                rew = rewards
                ac = list(bandit.arms.keys())[actions_replaced]

                f_rew = forgone_rewards
                f_ac = list(bandit.arms.keys())[1-actions_replaced]

                if bandit.feedback_type == 1:
                    if forced:
                        history += f"- On visit {trials_idx+1} someone else played Machine {ac} and earned {rew} point. On Machine {f_ac} the player would have earned {f_rew} point. \n"
        
                    else:
                        history += f"- On visit {trials_idx+1} you played Machine {ac} and earned {rew} point. On Machine {f_ac} you would have earned {f_rew} point. \n"
                else:
                    if forced:
                        history += f"- On visit {trials_idx+1} someone else played Machine {ac} and earned {rew} point.\n"
        
                    else:
                        history += f"- On visit {trials_idx+1} you played Machine {ac} and earned {rew} point.\n"

        simulated_data = memory.get_simulated_data()
        logger.save_test(run, simulated_data)