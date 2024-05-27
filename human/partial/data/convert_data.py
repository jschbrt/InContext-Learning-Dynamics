import numpy as np
import scipy.io
import pandas as pd

"""
This code is used to convert the partial task data of Lefebvre (https://figshare.com/articles/dataset/Behavioral_data_and_data_extraction_code/4265408/1)
to the fit the format of the cognitive models.
"""

n_subjects = 50
conditions = np.zeros((n_subjects, 96))
choices = np.zeros((n_subjects, 96))
reward = np.zeros((n_subjects, 96))

for i in range(n_subjects):
    data = scipy.io.loadmat('b_data/exp1_' + str(i+1))
    data = data['data']
    conditions[i,:] = data[:,2] # 1 to 4 as per condition
    choices[i,:] = data[:, 6] / 2 + 1.5 # 1 for left, 2 for right
    reward[i,:] = data[:, 7] # - it is actually 0 or 0.5 euro but we convert it later!

conditions -= 1
choices -= 1

df = pd.DataFrame({'run': np.repeat(np.arange(0,n_subjects),96),
                   'trials_idx': np.tile(np.arange(0,96),n_subjects),
                   'context': conditions.flatten(),
                   'actions': choices.flatten(),
                   'rewards': reward.flatten()})
df.to_csv(f'exp1.csv',mode='w')