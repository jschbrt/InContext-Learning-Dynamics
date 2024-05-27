# %%
import numpy as np
import scipy.io
import pandas as pd

"""
This code is used to convert the data of Chambon from: https://github.com/spalminteri/agency.git in _original_

Each file contains a matrix M, which corresponds to the data of one participant. 
There were 24 participants in the second experiment;

In the matrix M, each line corresponds to one trial. 

Each column refers to:

- column 1: the session number. Each experiment contains different sessions, with 4 blocks corresponding to the 4 experimental conditions. Participants had a break between each session.

- column 2: the condition number. Conditions 1 and 2 correspond to high-reward blocks (reward probabilities were 90% and 60% for the two symbols). Conditions 3 and 4 correspond to low-reward blocks (reward probabilities were 40% and 10%). In the second experiment, conditions 1 and 3 correspond to trials in which only the factual outcome was shown, while both the factual and counterfactual outcomes were shown in conditions 2 and 4.

- column 3: the trial number within a block, e.g. 1 corresponds to the first trial of a block, the trial in which the participants sees a given pair of symbols for the first time.

- column 4: the outcome of the best rewarded stimulus. The outcomes could either be +1 or -1. « 0 » indicates that the trial was a forced-choice trial - the best rewarded symbol was not shown to the participant in this trial.

- column 5: the outcome of the least rewarded symbol. The outcomes could either be 1 (the participant wins one point) or -1 (the participant lost one point). "0" indicates that the trial was a forced-choice trial, and that the least rewarded symbol was not shown to the participant in this trial.

- column 6: the participant's choice. "1" indicates that the participant has chosen the best rewarded symbol, and "0" that she has chosen the least rewarded symbol.

- column 7: whether the trial was a free choice (1) or a forced choice (0). For experiment 4, this column indicates whether the trial was a "go" (-1) or a "no-go" (1) trial. 

- column 8: the reaction time for selecting the symbol

- column 9: the reaction time for confirming the outcome (see experimental protocol).


For experiment 2 this translates to: 
- 1: high, partial
- 2: high, full
- 3: low, partial
- 4: low, full
per block: 20 free, 20 forced
"""

# for each subject save data into a pandas dataframe
df = pd.DataFrame(columns=['run', 'idx', 'actions', 'rewards', 'context', 'counter_actions', 'forgone_rewards', 'high', 'block_feedback_type'])

for nsub in range(1,25):
    path = 'Experiment2/'

    # Replace 'load' with loading the data manually, example: np.load or any other method
    data = scipy.io.loadmat(path + 'passymetrieII_Suj' + str(nsub) + '.mat')
    M = data['M']

    data = np.empty((len(M), 9))

    for i in range(len(M)):
        data[i, 0] = M[i, 5] + 1 # participant choice (+1 -> 1 = left, 2 = right)
        data[i, 1] = M[i, 3] * (M[i, 5] == 1) + M[i, 4] * (M[i, 5] == 0) # outcome for chosen option
                    # 3 -> best_rewarded / 4 -> worst rewarded / 5 -> participant choice (0 (worst), 1 (best))
        data[i, 2] = (M[i, 2]) # + 4 * (M[i, 0] - 1)  # idx: block number (0-15)
        data[i, 3] = 1 - M[i, 6] # context: free (0) forced (1)
        data[i, 4] = (1 - M[i, 5]) + 1 # forgone choice)
        data[i, 5] = M[i, 3] * ((data[i,4]-1) == 1) + M[i, 4] * ((data[i,4]-1) == 0) # outcome for unchosen option
                    # 3 -> best_rewarded / 4 -> worst rewarded / 5 -> participant choice (0 (worst), 1 (best))
        data[i, 6] = M[i, 1] # high / low reward probs
        data[i, 8] = M[i,0]
     
    data[:, 7] = ([0]*40 + [1]*40)*8 # partial, full, partial,
    data = data.astype(int)

    df_sub = pd.DataFrame(data, columns=['actions', 'rewards', 'trial', 'context', 'counter_actions', 'forgone_rewards', 'high', 'block_feedback_type', 'session_number'])
    df_sub['run'] = nsub

    df = pd.concat([df, df_sub])

# make 0 based -> 0 = left, 1 = right
df['actions'] = df['actions'] - 1
df['counter_actions'] = df['counter_actions'] - 1
df['trial'] = df['trial'] -1
df['run'] = df['run'] - 1
df['high'].replace({1: 1, 2: 1, 3: 0, 4: 0}, inplace=True)
df['session_number'] = df['session_number'] - 1

# Set initial regret to 0
df['regret'] = 0
df['regret'] = np.where((df['actions'] == 0), 0.6, df['regret']) # 0 bad choice, 1 good choice
df['idx'] = df.groupby(['run', 'trial']).cumcount()

df.to_csv('full_human.csv')

# %%