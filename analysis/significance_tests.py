"""
This script calculates the significance tests for the cognitive models and performance data.
"""
# %%
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def calculate_posterior_probabilities(df):
    """
    Calculate and return posterior probabilities from BIC values in the dataframe.
    """
    pivoted = df.pivot(index='run', columns='cognitive_model', values='bic')
    bic_diff = pivoted.sub(pivoted.min(axis=1), axis=0)
    unnormalized_pp = np.exp(-0.5 * bic_diff)
    normalized_pp = unnormalized_pp.div(unnormalized_pp.sum(axis=1), axis=0)
    # convert to plotting format
    reset = normalized_pp.reset_index()
    reset.head()
    melt = reset.melt(id_vars='run', 
                value_vars=reset.columns[1:],
                var_name='cognitive_model',
                value_name='pp')
    return melt

def load_models():
    partial_path = ['../llm/partial/data/claude-1/CM_fit_q0.25.csv',
                    '../human/partial/data/CM_fit_q0.25.csv',
                    '../meta-rl/partial/data/CM_fit_q0.25.csv']
    full_path = ['../llm/full/data/claude-1/CM_fit_full.csv',
                 '../human/full/data/CM_fit_full.csv',
                 '../meta-rl/full/data/CM_fit_full.csv']
    agency_path = ['../llm/agency/data/claude-1/CM_fit.csv',
                   '../human/agency/data/CM_fit.csv',
                   '../meta-rl/agency/data/CM_fit.csv']
    paths = [partial_path, full_path, agency_path]
    exp_names = ['partial', 'full', 'agency']
    agent_names = ['llm', 'human', 'meta-rl']
    
    merged = pd.DataFrame()

    for i, path_group in enumerate(paths):
        for j, path in enumerate(path_group):
            df = pd.read_csv(path)
            df['exp_name'] = exp_names[i]
            df['agent'] = agent_names[j]
            merged = pd.concat([merged, df], ignore_index=True)
    
    return merged

def performance_partial_convert():
    """
    Convert and preprocess partial performance data for human, LLM, and meta-RL agents.
    """
    # Human data
    human_partial = pd.read_csv('../human/partial/data/exp1.csv')
    # Set initial regret to 0
    human_partial['regret'] = 0
    # Update regret based on specific conditions
    human_partial['regret'] = np.where((human_partial['context'] == 1) & (human_partial['actions'] == 1), 0.375, human_partial['regret'])
    human_partial['regret'] = np.where((human_partial['context'] == 2) & (human_partial['actions'] == 0), 0.375, human_partial['regret'])
    # Count trials for each casino separately from 0 to 23
    human_partial['trial_cue'] = human_partial.groupby(['run', 'context']).cumcount()
    # Swap values 1 and 2 in the 'context' column
    # this is because cue 1 and cue 2 have been switched in all other analyses
    human_partial['context'] = human_partial['context'].replace({1: 2, 2: 1})
    human_partial['rewards'].replace({1.0:0.5}, inplace=True)
    human_partial = human_partial.rename(columns={'context': 'cue',
                                                  'actions': 'action',
                                                  'rewards': 'reward',
                                                  'trial_cue': 'cue_idx'})
    human_partial.drop(columns=['Unnamed: 0'], inplace=True)

    # LLM data
    dfs = [pd.read_csv(path) for path in glob('../llm/partial/data/claude-1/exp/run_*.csv')]
    llm_partial = pd.concat(dfs, ignore_index=True).drop(columns=['Unnamed: 0'])
    llm_partial['casino'] = llm_partial['casino']-1
    llm_partial['cues'] = llm_partial['casino']
    llm_partial['rewards'].replace({0: 0, 1: 0.5}, inplace=True)
    # add regret
    llm_partial['regret'] = 0
    llm_partial['regret'] = np.where((llm_partial['cues'] == 1) & (llm_partial['choice'] == 0), 0.375, llm_partial['regret'])
    llm_partial['regret'] = np.where((llm_partial['cues'] == 2) & (llm_partial['choice'] == 1), 0.375, llm_partial['regret'])
    # count trials for each casino seperately from 0 to 23
    llm_partial['trial_cue'] = llm_partial.groupby(['run', 'casino']).cumcount()
    llm_partial = llm_partial.rename(columns={'run': 'run',
                                            'cues': 'cue',
                                            'choice': 'action',
                                            'rewards': 'reward',
                                            'regret': 'regret',
                                            'trial_cue': 'cue_idx',
                                            'trial': 'trials_idx'})
    llm_partial.drop(columns=['casino', 'mean0', 'mean1', 'reward0', 'reward1'], 
                     inplace=True)

    # Meta-RL data
    meta_rl = pd.read_csv('../meta-rl/partial/data/exp/simulation_df.csv')
    meta_rl['cue_idx'] = meta_rl.groupby(['test_part_idx', 'cues']).cumcount()
    meta_rl.rename(columns={'test_part_idx': 'run',
                            'cues': 'cue',
                            'actions': 'action',
                            'rewards': 'reward',
                            'regrets': 'regret',
                            'trial_cue': 'cue_idx',
                            'trial': 'trials_idx'}, inplace=True)
    
    # Merge data
    partial_perf = pd.concat([human_partial, llm_partial, meta_rl], 
                             keys=['human', 'llm', 'meta-rl'], 
                             names=['agent'])
    partial_perf.reset_index(inplace=True)
    partial_perf.drop(columns=['level_1'], inplace=True)

    return partial_perf

def performance_full_convert():
    """
    Convert and preprocess full performance data for LLM, human, and meta-RL agents.
    """
    # LLM data
    llm = pd.read_csv('../llm/full/data/claude-1/sim.csv')
    llm['participants'] = ((llm['run']) // 4) # 4 sessions per participant
    llm['unique_run'] = llm.groupby(['run', 'block_idx']).ngroup()
    llm['idx'] = llm['unique_run'] % 16 # 16 blocks per participant
    llm['counter_actions'] = 1 - llm['actions']
    llm.drop(['block_idx','prompt', 'run'], axis=1, inplace=True)
    llm.rename(columns={'trials_idx' : 'trial',
                        'participants': 'run',
                        'cues': 'context',
                        'regrets': 'regret',
                        'opt_actions': 'optimal_choice'}, inplace=True)

    # Meta-RL data
    meta_rl = pd.read_csv('../meta-rl/full/data/exp/test/simulation_df.csv')
    max_blocks = meta_rl.batch_idx.max()+1
    meta_rl['run'] = meta_rl['test_eps_idx'] * max_blocks + meta_rl['batch_idx'] 
    meta_rl['participants'] = ((meta_rl['run']) // 4) # 4 sessions per participant
    meta_rl['unique_run'] = meta_rl.groupby(['run', 'block_idx']).ngroup()
    meta_rl['idx'] = meta_rl['unique_run'] % 16 # 16 blocks per participant
    meta_rl['counter_actions'] = 1 - meta_rl['actions']
    meta_rl.drop(['run'], axis=1, inplace=True)
    meta_rl.rename(columns={'trials_idx' : 'trial',
                            'participants': 'run',
                            'cues': 'context',
                            'regrets': 'regret',
                            'opt_actions': 'optimal_choice'}, inplace=True)

    # Human data
    human = pd.read_csv('../human/full/data/full_human.csv')
    human.rename(columns={'high' : 'block_reward_type',}, inplace=True)
    human.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Merge data
    df = pd.concat([llm, human, meta_rl], 
                   keys=['llm', 'human', 'meta-rl'], 
                   names=['agent']).reset_index(level=0)
    return df

def performance_agency_convert():
    """
    Convert and preprocess agency performance data for LLM, human, and meta-RL agents.
    """
    # LLM data
    llm = pd.read_csv('../llm/agency/data/claude-1/sim.csv')
    llm['participants'] = ((llm['run']) // 3) # 3 sessions per participant
    llm['unique_run'] = llm.groupby(['run', 'block_idx']).ngroup()
    llm['idx'] = llm['unique_run'] % 12 # 12 blocks per participant
    llm.drop(columns=['run', 'block_idx', 'opt_actions', 'prompt'], inplace=True)
    llm.rename(columns={'trials_idx' : 'trial',
                        'participants': 'run',
                        'cues': 'context',
                        'regrets': 'regret',
                        'opt_actions': 'optimal_choice'}, inplace=True)

    # Meta-RL data
    meta_rl = pd.read_csv('../meta-rl/agency/data/exp/test/simulation_df.csv')
    max_blocks = meta_rl.batch_idx.max()+1
    meta_rl['run'] = meta_rl['test_eps_idx'] * max_blocks + meta_rl['batch_idx'] 
    meta_rl['participants'] = ((meta_rl['run']) // 3) # 4 sessions per participant
    meta_rl['unique_run'] = meta_rl.groupby(['run', 'block_idx']).ngroup()
    meta_rl['idx'] = meta_rl['unique_run'] % 12
    meta_rl['counter_actions'] = 1 - meta_rl['actions']
    meta_rl.drop(['run'], axis=1, inplace=True)
    meta_rl.rename(columns={'trials_idx' : 'trial',
                            'participants': 'run',
                            'cues': 'context',
                            'regrets': 'regret',
                            'opt_actions': 'optimal_choice'}, inplace=True)

    # Human data
    human = pd.read_csv('../human/agency/data/agency_human.csv')
    human.rename(columns={'high': 'block_reward_type'}, inplace=True)
    human.drop(columns=['Unnamed: 0', 'counter_actions'], inplace=True)
    
    # Merge data
    df = pd.concat([llm, human, meta_rl], 
                   axis=0, 
                   keys=['llm', 'human', 'meta-rl'], 
                   names=['agent']).reset_index()
    df['plot_idx'] = df.groupby(['agent', 'run', 'idx', 'context']).cumcount()
    return df

def performance_data():
    """
    Merge partial, full, and agency performance data into a single DataFrame.
    """
    partial_df = performance_partial_convert()
    full_df = performance_full_convert()
    agency_df = performance_agency_convert()
    agency_df.drop(columns=['level_1'], inplace=True)

    df = pd.concat([partial_df, full_df, agency_df], 
                   axis=0, 
                   keys=['partial', 'full', 'agency'], 
                   names=['exp_name']).reset_index()
    return df

def performance_regret(t, exp_name, agent, trial_idx, llm_only=''):
    """
    Calculate performance regret and perform t-tests for LLM vs. human comparison.
    """
    if llm_only != '':
        t_max = t[trial_idx].max()
        t = t[[trial_idx, llm_only, 'regret']]
        t = t[t[trial_idx] == t_max]
        t = t.drop(columns=[trial_idx])
        td_mean = t.groupby([llm_only]).mean().rename(columns={'regret': 'mean'})
        td_sem = t.groupby(llm_only).sem().reset_index().rename(columns={'regret': 'sem'})
        td = pd.merge(td_mean, td_sem, on=llm_only)
        td['exp_name'] = exp_name
        td['agent'] = agent
        td['llm_only'] = llm_only

        # t-test regret at end of trial
        N = len(t[t[llm_only] == 0])-1
        statistic, pvalue = ttest_rel(t[t[llm_only] == 0]['regret'], 
                                      t[t[llm_only] == 1]['regret'], 
                                      axis=0)
        df = pd.DataFrame([[exp_name, 
                            agent, 
                            llm_only,
                            N, 
                            statistic.round(1), 
                            pvalue.round(4)]], 
                            columns=['exp_name', 
                                     'agent', 
                                     'llm_only', 
                                     'N-1', 
                                     'statistic', 
                                     'pvalue'])

    else:
        t_min = t[trial_idx].min()
        t_max = t[trial_idx].max()
        t = t[[trial_idx, 'regret']]
        t = t[t[trial_idx].isin([t_min, t_max])]
        td_mean = t.groupby(trial_idx).mean().reset_index().rename(columns={'regret': 'mean'})
        td_sem = t.groupby(trial_idx).sem().reset_index().rename(columns={'regret': 'sem'})
        td = pd.merge(td_mean, td_sem, on=trial_idx)
        td['exp_name'] = exp_name
        td['agent'] = agent
        td['llm_only'] = False

        # t-test regret at beginning vs end of trial
        N = len(t[t[trial_idx] == 0])-1
        statistic, pvalue = ttest_rel(t[t[trial_idx] == t_min]['regret'], 
                                      t[t[trial_idx] == t_max]['regret'], 
                                      axis=0, )
        df = pd.DataFrame([[exp_name, 
                            agent, 
                            llm_only, 
                            N, 
                            statistic.round(1), 
                            pvalue.round(4)]],
                            columns=['exp_name', 
                                     'agent', 
                                     'llm_only', 
                                     'N-1', 
                                     'statistic', 
                                     'pvalue'])
    return td, df


def regret_llm_vs_human(t, exp_name, trial_idx):
    """
    Calculate and compare regret between LLM and human agents.
    """
    t_max = t[trial_idx].max()
    t = t[t.agent.isin(['llm', 'human'])]
    t = t[['agent', trial_idx, 'regret']]
    t = t.query(f"{trial_idx} == {t_max}")
    td = t.drop([trial_idx], axis=1)
    td_mean = td.groupby('agent').mean().rename(columns={'regret': 'mean'})
    td_sem = td.groupby('agent').sem().reset_index().rename(columns={'regret': 'sem'})
    td = pd.merge(td_mean, td_sem, on='agent')
    td['exp_name'] = exp_name
    td['llm_only'] = False
    td['context_idx'] = t_max

    # t-test regret at beginning vs end of trial
    N = len(t[t['agent'] == 'llm'])-1

    statistic, pvalue = ttest_ind(t[t['agent'] == 'llm']['regret'], t[t['agent'] == 'human']['regret'], axis=0, )
    df = pd.DataFrame([[exp_name, 
                        'llm vs human at end', 
                        N, 
                        statistic.round(4), 
                        pvalue.round(1)]], 
                        columns=['exp_name', 
                                 'comparison', 
                                 'N-1', 
                                 'statistic', 
                                 'pvalue'])
    return td, df

# %%

if __name__ == "__main__":
    # Performance comparison for all experiments (t-test, mean and sem)
    p_df = performance_data()
    df_ttest = pd.DataFrame(columns=['exp_name', 'agent'])
    df_mean_sem = pd.DataFrame(columns=['exp_name', 'agent'])

    # Partial experiment
    t = p_df[(p_df['agent'] == 'llm') & (p_df['exp_name'] == 'partial')]
    mean_sem, ttest = performance_regret(t, 'partial', 'llm', 'cue_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)

    # Full experiment
    t = p_df[(p_df['agent'] == 'llm') & (p_df['exp_name'] == 'full')]
    t['context_idx'] = t.groupby(['run', 'idx', 'context']).cumcount()
    t = t[(t['block_feedback_type'] == 1) & (t['context'] == 0)]
    mean_sem, ttest = performance_regret(t, 'full', 'llm', 'context_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)
   
    t = p_df[(p_df['agent'] == 'llm') & (p_df['exp_name'] == 'full')]
    t['context_idx'] = t.groupby(['run', 'idx', 'context']).cumcount()
    t = t[(t['context'] == 0)]
    mean_sem, ttest = performance_regret(t, 'full', 'llm', 'context_idx', llm_only='block_feedback_type')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)

    # Agency experiment
    t = p_df[(p_df['agent'] == 'llm') & (p_df['exp_name'] == 'agency')]
    t['context_idx'] = t.groupby(['run', 'idx', 'context']).cumcount()
    t = t[(t['block_forced_type'] == 1) & (t['context'] == 0)]
    mean_sem, ttest = performance_regret(t, 'agency', 'llm', 'context_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)

    t = p_df[(p_df['agent'] == 'llm') & (p_df['exp_name'] == 'agency')]
    t['context_idx'] = t.groupby(['run', 'idx', 'context']).cumcount()
    t = t[(t['context'] == 0)]
    mean_sem, ttest = performance_regret(t, 'agency', 'llm', 'context_idx', llm_only='block_forced_type')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)

    # LLM vs Human comparison
    t = p_df[(p_df['exp_name'] == 'agency')]
    t['context_idx'] = t.groupby(['agent','run', 'idx', 'context']).cumcount()
    t = t[(t['block_forced_type'] == 1) & (t['context'] == 0)]
    mean_sem, ttest = regret_llm_vs_human(t, 'agency', 'context_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)
   

    t = p_df[(p_df['exp_name'] == 'full')]
    t['context_idx'] = t.groupby(['agent','run', 'idx', 'context']).cumcount()
    t = t[(t['block_feedback_type'] == 1) & (t['context'] == 0)]
    mean_sem, ttest = regret_llm_vs_human(t, 'full', 'context_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)
   
    t = p_df[(p_df['exp_name'] == 'partial')]
    mean_sem, ttest = regret_llm_vs_human(t, 'partial', 'cue_idx')
    df_ttest = pd.concat([df_ttest, ttest], ignore_index=True)
    df_mean_sem = pd.concat([df_mean_sem, mean_sem], ignore_index=True)
   
    # Save results
    df_ttest.to_csv('regret_ttest.csv', index=False)
    df_mean_sem.to_csv('regret_mean_sem.csv', index=False)

    print('---------------------')
    print('Model comparison')

    # Model comparison
    c_df = load_models()
    df = pd.DataFrame(columns=['exp_name', 'agent', 'cognitive_model'])

    # Calculate means and standard errors for PP, NLL, BIC, beta and alpha(s) for each cognitive model
    for exp_name in ['partial', 'full', 'agency']:
        for agent in ['llm', 'human', 'meta-rl']:
            t = c_df[(c_df['exp_name'] == exp_name) & (c_df['agent'] == agent)]
            t2 = calculate_posterior_probabilities(t)
            t = pd.merge(t2, t, on=['run', 'cognitive_model'])
            for cognitive_model in t.cognitive_model.unique():
                t_cm = t[t['cognitive_model'] == cognitive_model]
                t_cm = t_cm.dropna(axis=1)
                t_cm = t_cm.drop(columns=['run'])

                td_mean = t_cm.mean().reset_index()
                td_mean = td_mean.rename(columns={'index':'measure', 0: 'mean'})
                td_sem = t_cm.sem().reset_index()
                td_sem = td_sem.rename(columns={'index':'measure',0: 'sem'})
                t_cm = pd.merge(td_mean, td_sem, on='measure', )

                # merge with df
                t_cm['exp_name'] = exp_name
                t_cm['agent'] = agent
                t_cm['cognitive_model'] = cognitive_model
                df = pd.concat([df, t_cm], ignore_index=True)
    df = df.round(3)
    df.to_csv('model_comparison_means.csv', index=False)

    # Paired t-test comparing the cognitive models for each agent
    df = pd.DataFrame(columns=['exp_name', 'agent', 'cognitive_models', 'N', 'statistic', 'pvalue'])     # transform dataframe so that each measure is a column
    for exp_name in ['partial', 'full', 'agency']:
        for agent in ['llm', 'human', 'meta-rl']:
            t = c_df[(c_df['exp_name'] == exp_name) & (c_df['agent'] == agent)]
            t2 = calculate_posterior_probabilities(t)
            t = pd.merge(t2, t, on=['run', 'cognitive_model'])
            t = t.pivot(index='run', columns='cognitive_model', values='pp')
            c_mod1 = t.columns.unique()[0]
            c_mod2 = t.columns.unique()[1]
            first_model = t[c_mod1]
            second_model = t[c_mod2]
            t['is_a'] = (first_model - second_model) > 0
            value_counts = t['is_a'].value_counts()[True] if True in t['is_a'].value_counts() else 0
            max = t['is_a'].count()
            percentage = t['is_a'].value_counts()[True]/len(t) if True in t['is_a'].value_counts() else 0

            N = len(first_model) -1
            statistic, pvalue = ttest_rel(first_model, second_model, axis=0)
            statistic = statistic.round(1)
            pvalue = pvalue.round(4)
            t = pd.DataFrame([[exp_name, 
                               agent, 
                               f'paried t-test {c_mod1} vs {c_mod2}', 
                               N, 
                               statistic, 
                               pvalue,
                               value_counts,
                               max,
                               percentage]], 
                               columns=['exp_name', 
                                        'agent', 
                                        'cognitive_models', 
                                        'N-1', 
                                        'statistic', 
                                        'pvalue',
                                        'value_counts',
                                        'max',
                                        'percentage'])
            df = pd.concat([df, t], ignore_index=True)
    df.to_csv('model_comparison_t-test_pp.csv', index=False)

    # T-test of significant difference of the alpha pairs for each cognitive model
    df = pd.DataFrame(columns=['exp_name', 'agent', 'cognitive_models'])
    for exp_name in ['partial', 'full', 'agency']:
        for agent in ['llm', 'human', 'meta-rl']:
            t = c_df[(c_df['exp_name'] == exp_name) & (c_df['agent'] == agent)]
            if exp_name == 'partial':
                cognitive_model = 'Model_2alpha'
                tc = t[t['cognitive_model'] == cognitive_model]
                alphas = ['alpha_pos', 'alpha_neg']
                N = len(tc)-1
                statistic, pvalue = ttest_rel(tc[alphas[0]], tc[alphas[1]], axis=0)
                statistic = statistic.round(3)
                pvalue = pvalue.round(4)
                tm = pd.DataFrame([[exp_name, 
                                   agent, 
                                   cognitive_model, 
                                   f'paried t-test: {alphas[0]} vs {alphas[1]}',
                                   N, 
                                   statistic, 
                                   pvalue]], 
                                   columns=['exp_name', 
                                            'agent', 
                                            'cognitive_models', 
                                            'comparison',
                                            'N-1', 
                                            'statistic', 
                                            'pvalue'])
                df = pd.concat([df, tm], ignore_index=True)
            elif exp_name == 'full':
                cognitive_model = 'Model_2alpha'
                tc = t[t['cognitive_model'] == cognitive_model]
                alphas = ['alpha_conf', 'alpha_disconf']
                N = len(tc)-1
                statistic, pvalue = ttest_rel(tc[alphas[0]], tc[alphas[1]], axis=0)
                statistic = statistic.round(3)
                pvalue = pvalue.round(4)
                tm = pd.DataFrame([[exp_name, 
                                   agent, 
                                   cognitive_model, 
                                   f'paried t-test: {alphas[0]} vs {alphas[1]}',
                                   N, 
                                   statistic, 
                                   pvalue]], 
                                   columns=['exp_name', 
                                            'agent', 
                                            'cognitive_models', 
                                            'comparison',
                                            'N-1', 
                                            'statistic', 
                                            'pvalue'])
                df = pd.concat([df, tm], ignore_index=True)

                cognitive_model = 'Model_4alpha'
                tc = t[t['cognitive_model'] == cognitive_model]
                alphas_list = [['alpha_pos_chosen', 'alpha_neg_chosen'], ['alpha_pos_unchosen', 'alpha_neg_unchosen']]
                N = len(tc)-1
                for alphas in alphas_list:
                    statistic, pvalue = ttest_rel(tc[alphas[0]], tc[alphas[1]], axis=0)
                    statistic = statistic.round(3)
                    pvalue = pvalue.round(4)
                    tm = pd.DataFrame([[exp_name, 
                                    agent, 
                                    cognitive_model, 
                                    f'paried t-test: {alphas[0]} vs {alphas[1]}',
                                    N, 
                                    statistic, 
                                    pvalue]], 
                                    columns=['exp_name', 
                                                'agent', 
                                                'cognitive_models', 
                                                'comparison',
                                                'N-1', 
                                                'statistic', 
                                                'pvalue'])
                    df = pd.concat([df, tm], ignore_index=True)
            elif exp_name == 'agency':
                 
                cognitive_model = 'Model_3alpha'
                tc = t[t['cognitive_model'] == cognitive_model]
                alphas = ['alpha_pos_free', 'alpha_neg_free']
                N = len(tc)-1
                statistic, pvalue = ttest_rel(tc[alphas[0]], tc[alphas[1]], axis=0)
                statistic = statistic.round(3)
                pvalue = pvalue.round(4)
                tm = pd.DataFrame([[exp_name, 
                                   agent, 
                                   cognitive_model, 
                                   f'paried t-test: {alphas[0]} vs {alphas[1]}',
                                   N, 
                                   statistic, 
                                   pvalue]], 
                                   columns=['exp_name', 
                                            'agent', 
                                            'cognitive_models', 
                                            'comparison',
                                            'N-1', 
                                            'statistic', 
                                            'pvalue'])
                df = pd.concat([df, tm], ignore_index=True)

                cognitive_model = 'Model_4alpha'
                tc = t[t['cognitive_model'] == cognitive_model]
                alphas_list = [['alpha_pos_free', 'alpha_pos_forced'], ['alpha_neg_free', 'alpha_neg_forced']]
                N = len(tc)-1
                for alphas in alphas_list:
                    statistic, pvalue = ttest_rel(tc[alphas[0]], tc[alphas[1]], axis=0)
                    statistic = statistic.round(3)
                    pvalue = pvalue.round(4)
                    tm = pd.DataFrame([[exp_name, 
                                    agent, 
                                    cognitive_model, 
                                    f'paried t-test: {alphas[0]} vs {alphas[1]}',
                                    N, 
                                    statistic, 
                                    pvalue]], 
                                    columns=['exp_name', 
                                                'agent', 
                                                'cognitive_models', 
                                                'comparison',
                                                'N-1', 
                                                'statistic', 
                                                'pvalue'])
                    df = pd.concat([df, tm], ignore_index=True)

    df.to_csv('model_comparison_t-test_alphas.csv', index=False)
   # %% 