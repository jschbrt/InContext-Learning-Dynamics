"""
This script fits cognitive models to the data of the different agents.
"""

import pandas as pd
import numpy as np
from glob import glob 
from scipy.optimize import minimize
from cognitive_models import *
from numpy.random import rand
import argparse

REPS = 100

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll

def generate_init(num_alphas):
    epsilon = 1e-10 # to avoid bounds (0,1) for log calulations

    a_min, a_max = [0 + epsilon, 1 - epsilon]
    b_min, b_max = [1 + epsilon, 10 - epsilon]
    
    init = []

    b = (b_min + rand(1) * (b_max - b_min))[0]
    init.append(b)

    for _ in range(num_alphas):
        a = (a_min + rand(1) * (a_max - a_min))[0]
        init.append(a)
    return init

def partial_fitting(df, agent, q_initial, options=None, reward_name=None):

    fitted_models = pd.DataFrame(columns = ['agent',
                                            'cognitive_model',
                                            'run',
                                            'nll',
                                            'bic',
                                            'beta',
                                            'alpha',
                                            'alpha_pos',
                                            'alpha_neg'])
    
    CM = CM_Task1_Partial
    models = [CM.Model_1alpha, CM.Model_2alpha]

    for model in models:
        print(f'Fitting {model.__name__} for {agent}')       
        print(f'Amount of data: {df.run.max()+1}')
        for nsub in range(df.run.max()+1):
            print(f'{nsub+1}')

            M = df[df['run'] == nsub]

            data = np.empty((len(M), 3))
            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values # not used
            data[:, 2] = M.context.values + 1
            data = data.astype(int)

            rewards = np.empty((len(M), 1))
            rewards[:, 0] = M.rewards.values

            epsilon = 1e-10 # to avoid bounds (0,1) for log calulations
            if model.__name__ == 'Model_1alpha':

                for _ in range(REPS):
                    init_guess = generate_init(1)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 1
                    est = minimize(model, init_guess, (data, rewards, q_initial), bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha': est.x[1]}])
                if reward_name is not None:
                    temp_run['reward'] = reward_name
                    temp_run['options'] = options
                    temp_run['q_initial'] = q_initial

                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

            elif model.__name__ == 'Model_2alpha':

                for _ in range(REPS):
                    init_guess = generate_init(2)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 2
                    est = minimize(model, init_guess, (data, rewards, q_initial), bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_pos': est.x[1],
                                        'alpha_neg': est.x[2]}])
                
                if reward_name is not None:
                    temp_run['reward'] = reward_name
                    temp_run['options'] = options
                    temp_run['q_initial'] = q_initial

                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

    return fitted_models

def full_fitting_partial(df, agent):

    fitted_models = pd.DataFrame(columns = ['agent',
                                            'cognitive_model',
                                            'run',
                                            'nll',
                                            'bic',
                                            'beta',
                                            'alpha',
                                            'alpha_pos',
                                            'alpha_neg'])
    
    CM = CM_Task2_Partial
    models = [CM.Model_1alpha, CM.Model_2alpha]

    for model in models:
        print(f'Fitting {model.__name__} for {agent}')       
        print(f'Amount of data: {df.run.max()+1}')
        for nsub in range(df.run.max()+1):
            print(f'{nsub+1}')

            M = df[df['run'] == nsub]

            data = np.empty((len(M), 7))
            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values
            data[:, 2] = M.idx.values + 1
            data[:, 3] = 1- M.context.values.astype(bool).astype(int)
            data[:, 4] = M.counter_actions.values + 1
            data[:, 5] = M.forgone_rewards.values
            data[:, 6] = M.block_feedback_type.values
            
            data = data.astype(int)

            epsilon = 1e-10 # to avoid bounds (0,1) for log calulations
            if model.__name__ == 'Model_1alpha':

                for _ in range(REPS):
                    init_guess = generate_init(1)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 1
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha': est.x[1]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

            elif model.__name__ == 'Model_2alpha':

                for _ in range(REPS):
                    init_guess = generate_init(2)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 2
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_pos': est.x[1],
                                        'alpha_neg': est.x[2]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

    return fitted_models

def full_fitting_full(df, agent):

    fitted_models = pd.DataFrame(columns = ['agent',
                                            'cognitive_model',
                                            'run',
                                            'nll',
                                            'bic',
                                            'beta',
                                            'alpha_conf',
                                            'alpha_disconf',
                                            'alpha_pos_chosen',
                                            'alpha_pos_unchosen',
                                            'alpha_neg_chosen',
                                            'alpha_neg_unchosen'])
    
    CM = CM_Task2_Full
    models = [CM.Model_2alpha, CM.Model_4alpha]

    for model in models:
        print(f'Fitting {model.__name__} for {agent}')       
        print(f'Amount of data: {df.run.max()+1}')
        for nsub in range(df.run.max()+1):
            print(f'{nsub+1}')

            M = df[df['run'] == nsub]

            data = np.empty((len(M), 7))
            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values
            data[:, 2] = M.idx.values + 1
            data[:, 3] = 1- M.context.values.astype(bool).astype(int)
            data[:, 4] = M.counter_actions.values + 1
            data[:, 5] = M.forgone_rewards.values
            data[:, 6] = M.block_feedback_type.values
            
            data = data.astype(int)

            epsilon = 1e-10 # to avoid bounds (0,1) for log calulations

            if model.__name__ == 'Model_2alpha':


                for _ in range(REPS):
                    init_guess = generate_init(2)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 2
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_conf': est.x[1],
                                        'alpha_disconf': est.x[2]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)
            
            elif model.__name__ == 'Model_4alpha':

                for _ in range(REPS):
                    init_guess = generate_init(4)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 4
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_pos_chosen': est.x[1],
                                        'alpha_neg_chosen': est.x[2],
                                        'alpha_pos_unchosen': est.x[3],
                                        'alpha_neg_unchosen': est.x[4]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

    return fitted_models

def agency_fitting(df, agent):

    fitted_models = pd.DataFrame(columns = ['agent',
                                            'cognitive_model',
                                            'run',
                                            'nll',
                                            'bic',
                                            'beta',
                                            'alpha_forced',
                                            'alpha_pos_free',
                                            'alpha_neg_free',
                                            'alpha_pos_forced',
                                            'alpha_neg_forced'])
    
    CM = CM_Task3_Agency_Mixed
    models = [CM.Model_3alpha, CM.Model_4alpha]

    for model in models:
        print(f'Fitting {model.__name__} for {agent}')       
        print(f'Amount of data: {df.run.max()+1}')
        for nsub in range(df.run.max()+1):
            print(f'{nsub+1}')

            M = df[df['run'] == nsub]

            data = np.empty((len(M), 4))

            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values
            data[:, 2] = M.idx.values + 1
            data[:, 3] = 1- M.context.values.astype(bool).astype(int)
            data = data.astype(int)

            epsilon = 1e-10 # to avoid bounds of 0/1
            if model.__name__ == 'Model_3alpha':

                for _ in range(REPS):
                    init_guess = generate_init(3)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 3
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_pos_free': est.x[1],
                                        'alpha_neg_free': est.x[2],
                                        'alpha_forced': est.x[3]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

            elif model.__name__ == 'Model_4alpha':


                for _ in range(REPS):
                    init_guess = generate_init(4)
                    bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 4
                    est = minimize(model, init_guess, data, bounds=bounds)
                    best_est = np.inf
                    if est.fun < best_est:
                        best_est = est.fun
                        best_estimation = est

                est = best_estimation
                bic = calculate_BIC(len(init_guess), len(data), est.fun)
                temp_run = pd.DataFrame([{'agent': agent,
                                        'cognitive_model': model.__name__,
                                        'run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_pos_free': est.x[1],
                                        'alpha_neg_free': est.x[2],
                                        'alpha_pos_forced': est.x[3],
                                        'alpha_neg_forced': est.x[4]}])
                fitted_models = pd.concat([fitted_models, temp_run], axis=0)

    return fitted_models

def fit_partial_main():
    q_initial = 0.25

    # Fitting of partial task
    agent = 'human'
    df = pd.read_csv('../human/partial/data/exp1.csv')
    df = df[['run', 'actions', 'rewards', 'context']]
    df['rewards'] = df['rewards'] / 2
    fitted_models = partial_fitting(df, agent, q_initial)
    fitted_models.to_csv(f'../human/partial/data/CM_fit_q0.25.csv', index=False)

    agent = 'claude-1'
    df = pd.read_csv('../llm/partial_addition/data/claude-1/exp_opt_2_0.5_0.0.csv')
    df.rename(columns={'cues': 'context'}, inplace=True)
    df = df[['run', 'context', 'actions', 'rewards']]
    fitted_models = partial_fitting(df, agent, q_initial)
    fitted_models.to_csv(f'../llm/partial/data/claude-1/CM_fit_q0.25.csv', index=False)

    agent = 'meta-rl'
    df = pd.read_csv('../meta-rl/partial/data/exp/simulation_df.csv')
    df.rename(columns={'cues': 'context',
                    'test_part_idx': 'run'}, inplace=True)
    df = df[['run', 'context', 'actions', 'rewards']]
    fitted_models = partial_fitting(df, agent, q_initial)
    fitted_models.to_csv('../meta-rl/partial/data/CM_fit_q0.25.csv', index=False)

def fit_full():
    # full task
    agent = 'claude-1'
    file = '../llm/full/data/claude-1/sim.csv'
    df = pd.read_csv(file)
    df['counter_actions'] = 1.0 - df['actions']
    # create block_idx across multiple "sessions"
    df['part_run'] = ((df['run']) // 4) # 4 sessions per participant
    df['unique_run'] = df.groupby(['run', 'block_idx']).ngroup()
    df['idx'] = (df['unique_run'] % 16) # 16 blocks per participant


    df = df[['part_run', 
             'actions', 
             'rewards', 
             'idx', 
             'cues',
             'counter_actions',
             'forgone_rewards',
             'block_feedback_type']]
    df.rename(columns={'part_run': 'run',
                       'cues': 'context'}, inplace=True)
    
    fitted_models = full_fitting_partial(df, agent)
    fitted_models.to_csv('../llm/full/data/claude-1/CM_fit_partial.csv', index=False)

    fitted_models = full_fitting_full(df, agent)
    fitted_models.to_csv('../llm/full/data/claude-1/CM_fit_full.csv', index=False)

    agent = 'meta-rl'
    file = '../meta-rl/full/data/exp/test/simulation_df.csv'
    df = pd.read_csv(file)
    df['counter_actions'] = 1.0 - df['actions']
    # create block_idx across multiple "sessions"
    max_blocks = df.batch_idx.max()+1
    df['run'] = df['test_eps_idx'] * max_blocks + df['batch_idx']
    df['part_run'] = ((df['run']) // 4) # 4 sessions per participant
    df['unique_run'] = df.groupby(['run', 'block_idx']).ngroup()
    df['idx'] = (df['unique_run'] % 16) # 16 blocks per participant

    df = df[['part_run', 
             'actions', 
             'rewards', 
             'idx', 
             'cues',
             'counter_actions',
             'forgone_rewards',
             'block_feedback_type']]
    df.rename(columns={'part_run': 'run',
                       'cues': 'context'}, inplace=True)

    fitted_models = full_fitting_full(df, agent)
    fitted_models.to_csv('../meta-rl/full/data/CM_fit_full.csv', index=False)

    agent = 'human'
    df = pd.read_csv('../human/full/data/full_human.csv')
    fitted_models = full_fitting_full(df, agent)
    fitted_models.to_csv('../human/full/data/CM_fit_full.csv', index=False)

def fit_agency():

    agent = 'claude-1'
    file = '../llm/agency/data/claude-1/sim.csv'
    df = pd.read_csv(file)

    # create block_idx across multiple "sessions"
    df['part_run'] = ((df['run']) // 3)
    df['unique_run'] = df.groupby(['run', 'block_idx']).ngroup()
    df['idx'] = df['unique_run'] % 12 # 12 blocks per participant
    # select only forced blocks
    df = df[df.block_forced_type == 1.0] 

    df = df[['part_run', 
             'actions', 
             'rewards', 
             'idx', 
             'cues']]
    df.rename(columns={'part_run': 'run',
                       'cues': 'context'}, inplace=True)
    fitted_models = agency_fitting(df, agent)
    fitted_models.to_csv('../llm/agency/data/claude-1/CM_fit.csv', index=False)

    agent = 'meta-rl'
    df = pd.read_csv('../meta-rl/agency/data/exp/test/simulation_df.csv')

    # create block_idx across multiple "sessions"
    max_blocks = df.batch_idx.max()+1
    df['run'] = df['test_eps_idx'] * max_blocks + df['batch_idx']
    df['part_run'] = ((df['run']) // 3) # 3 sessions per participant
    df['unique_run'] = df.groupby(['run', 'block_idx']).ngroup()
    df['idx'] = df['unique_run'] % 12 # 12 blocks per participant
    # select only forced blocks
    df = df[df.block_forced_type == 1.0]

    df = df[['part_run', 
                'actions', 
                'rewards', 
                'idx', 
                'cues']]
    df.rename(columns={'part_run': 'run',
                        'cues': 'context'}, inplace=True)
    fitted_models = agency_fitting(df, agent)
    fitted_models.to_csv('../meta-rl/agency/data/CM_fit.csv', index=False)

    agent = 'human'
    file = '../human/agency/data/agency_human.csv'
    df = pd.read_csv(file)

    fitted_models = agency_fitting(df, agent)
    fitted_models.to_csv('../human/agency/data/CM_fit.csv', index=False)

def fit_partial_addition():

    base_path = '../llm/partial_addition/data/claude-1/'

    model_df = pd.DataFrame()

    agent = 'claude-1'

    options = [2, 3, 4]
    reward_names = {'0.5_-0.5':0.0, 
               '1.0_0.0':0.5}

    for option in options:
        for reward_name, q_initial in reward_names.items():
            try:
                df = pd.read_csv(base_path+f'exp_opt_{option}_{reward_name}.csv')
            except:
                pass
            df.rename(columns={'cues': 'context'}, inplace=True)
            df = df[['run', 'context', 'actions', 'rewards']]
            fitted_models = partial_fitting(df, agent, q_initial, option, reward_name)
            model_df = pd.concat([model_df, fitted_models], axis=0)
    model_df.to_csv(base_path+f'CM_fit_addition.csv', index=False)

def fit_partial_llms():

    q_initial = 0.5
    base_path = '../llm/partial/data/'
    llms = glob(base_path + '*/')
    llms = [llm.split('/')[-2] for llm in llms]
    
    for agent in llms: 
        df = pd.DataFrame(columns = ['run',
                                    'actions',
                                    'rewards',
                                    'context'])
        for run in range(0, 50):
            file = f'../llm/partial/data/{agent}/exp/run_{run}.csv'
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'choice': 'actions',
                                    'casino': 'context'}, inplace=True)
            df = pd.concat([df, temp_df], axis=0)
        fitted_models = partial_fitting(df, agent, q_initial)
        fitted_models.to_csv(f'../llm/partial/data/{agent}/CM_fit_q{q_initial}.csv', index=False)

def fit_partial_claudes():

    q_initial = 0.5
    base_path = '../llm/partial/data/'
    llms = glob(base_path + 'claude-*/')
    llms = [llm.split('/')[-2] for llm in llms]
    
    llms = ['claude-3-sonnet-20240229']


    for agent in llms: 
        df = pd.DataFrame(columns = ['run',
                                    'actions',
                                    'rewards',
                                    'context'])
        for run in range(0, 5):
            file = f'../llm/partial/data/{agent}/exp/run_{run}.csv'
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'choice': 'actions',
                                    'casino': 'context'}, inplace=True)
            df = pd.concat([df, temp_df], axis=0)
        fitted_models = partial_fitting(df, agent, q_initial)
        fitted_models.to_csv(f'../llm/partial/data/{agent}/CM_fit_q{q_initial}.csv', index=False)


if __name__ == '__main__':

    # Run
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits', nargs='+', default=['partial_main'])#['partial_main', 'full', 'agency', 'partial_llms', 'partial_addition'])
    args = parser.parse_args()  
    fits = args.fits

    if 'partial_main' in fits:
        fit_partial_main()
    if 'full' in fits:
        fit_full()
    if 'agency' in fits:
        fit_agency()

    if 'partial_llms' in fits:
        fit_partial_llms()
    if 'partial_addition' in fits:
        fit_partial_addition()

    if 'partial_claudes' in fits:
        fit_partial_claudes()