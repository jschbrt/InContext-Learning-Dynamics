"""
This script contains functions to generate and save plots for model comparison and learning rate comparison.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import scipy.stats
from glob import glob

# Configure environment and matplotlib settings
os.environ['PATH'] += os.pathsep + '$HOME/.TinyTeX/bin/x86_64-linux'
import matplotlib as mpl


dpi = 300
largesize = 10
smallsize = 9
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 10,
    'font.serif': 'CMU Serif',
    'xtick.labelsize': smallsize,
    'ytick.labelsize': smallsize,
    'axes.labelsize': largesize,
    'legend.fontsize': smallsize,
    'figure.dpi': dpi
})

errwidth = 0.9
capsize = 0.1

# Color palette settings
cllmp = "#85518aff"
cllmm = "#85518aff" # "#a26ca6ff"
cllmf = "#85518aff" # "#985d9dff"

chumanp = "#8f848fff"
chumanm = "#8f848fff" #"#b4a9b4ff"
chumanf = "#8f848fff" #"#a497a4ff"

cmetarlp = "#b96b87ff" # "#6a738cff"
cmetarlm = "#b96b87ff" # "#868ea7ff"
cmetarlf = "#b96b87ff" # "#6f7898ff"

cogmodel2 = "#9a802fff" # "#ab9960ff"
cogmodel1 = "#cdbe8cff" # "#b8aa79ff"

palette_single = {'human': chumanp,
                  'llm': cllmp,
                  'meta-rl': cmetarlp}

palette_two = {'human': [chumanp, chumanm],
               'llm': [cllmp, cllmm],
               'meta-rl': [cmetarlp, cmetarlm]}

palette_three = {'human': [chumanp, chumanm, chumanf],
                 'llm': [cllmp, cllmm, cllmf],
                 'meta-rl': [cmetarlp, cmetarlm, cmetarlf]}

def calculate_posterior_probabilities(df):
    """
    Calculate and return posterior probabilities from BIC values in the dataframe.
    """
    pivoted = df.pivot(index='run', columns='cognitive_model', values='bic')
    # calculate PP
    bic_diff = pivoted.sub(pivoted.min(axis=1), axis=0)
    unnormalized_pp = np.exp(-0.5 * bic_diff)
    normalized_pp = unnormalized_pp.div(unnormalized_pp.sum(axis=1), axis=0)
    # convert to plotting format
    reset = normalized_pp.reset_index()
    reset.head()
    melt = reset.melt(id_vars='run', 
                value_vars=reset.columns[1:],
                var_name='fitting_model',
                value_name='bic')
    return melt

def model_comparisons(data):
    """
    Generate and save model comparison plots for different agents and experimental conditions.
    """
    exp_labels = {'partial': [r'$\alpha$', r'$2\alpha$'], 
                  'full': [r'$2\alpha$', r'$4\alpha$'], 
                  'agency': [r'$3\alpha$', r'$4\alpha$']}

    cm_partial_human_llm(data, exp_labels['partial'])

    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'full')]
    generate_model_comparison(df, exp_labels['full'], f'pp_full_human')
    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'full')]
    generate_model_comparison(df, exp_labels['full'], f'pp_full_llm')
    df = data[(data['agent'] == 'meta-rl') & (data['exp_name'] == 'full')]
    generate_model_comparison(df, exp_labels['full'], f'pp_full_meta-rl')

    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'agency')]
    generate_model_comparison(df, exp_labels['agency'], f'pp_agency_human')
    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'agency')]
    generate_model_comparison(df, exp_labels['agency'], f'pp_agency_llm')
    df = data[(data['agent'] == 'meta-rl') & (data['exp_name'] == 'agency')]
    generate_model_comparison(df, exp_labels['agency'], f'pp_agency_meta-rl')

def cm_partial_human_llm(df, labels):
    """
    Generate and save partial experiment model comparison plot for human and LLM agents.
    """
    df_llm = df[(df['agent'] == 'llm') & (df['exp_name'] == 'partial')]
    melt_llm = calculate_posterior_probabilities(df_llm)

    df_human = df[(df['agent'] == 'human') & (df['exp_name'] == 'partial')]
    melt_human = calculate_posterior_probabilities(df_human)

    melt = pd.concat([melt_llm, melt_human], keys=['llm', 'human'] , names=['agent'], axis=0).reset_index()

    fig, ax = plt.subplots(1,1, figsize=(2.15,2.05), dpi=dpi)
    colors = [cogmodel1, cogmodel2]

    sns.barplot(data=melt, 
                x='agent', 
                y='bic', 
                hue='fitting_model',
                palette=colors,
                saturation=1, 
                errwidth=errwidth, 
                capsize=capsize, 
                ax=ax)
    sns.despine(ax=ax)

    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('PP')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.hlines(y=melt.mean(numeric_only=True)['bic'], 
            xmin=-0.5, 
            xmax=1.5, 
            linestyle='--', 
            alpha=0.5, 
            color='grey', 
            linewidth=1)
    ax.set_xticklabels(['LLM', 'Human'])
    ax.set_xlabel('Model Comparison')

    # legend
    handles, labels = ax.get_legend_handles_labels()
    labels = []
    ax.legend(handles, labels, loc='upper right', fontsize=smallsize, frameon=False, title='')

    fig.tight_layout()
    fig.savefig(f'partial_pp_llm_human.pdf', bbox_inches='tight')

def lr_partial_human_llm(df):
    """
    Generate and save learning rate comparison plots for partial experiments.
    """
    df = df[((df['agent'] == 'llm') | (df['agent'] == 'human')) & (df['exp_name'] == 'partial')]
    df = df[df['cognitive_model'] == 'Model_2alpha']
    melted = df.melt(id_vars=['agent'], value_vars=['alpha_pos', 'alpha_neg'])
    melted['merged_lr_with_agent'] = melted['agent'] + melted['variable'] 

    fig, ax = plt.subplots(1,1, figsize=(2.1,1.9), dpi=dpi)
    colors = [cllmp, cllmm, chumanp, chumanm]

    sns.barplot(data=melted,
                x='agent',
                y='value',
                hue='variable',
                capsize=capsize,
                errwidth=errwidth,
                palette=colors,
                saturation=1,
                ax=ax)
    sns.despine(ax=ax)

    ax.set_ylabel('Learning rate', fontsize=largesize)
    ax.set_xlabel(' ', fontsize=smallsize)
    ax.tick_params(axis='y',labelsize=smallsize)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.0])

    ax.set_xticks([-0.2, 0.2, 0.8, 1.2])
    ax.set_xticklabels([r'$\alpha^+$', r'$\alpha^-$', r'$\alpha^+$', r'$\alpha^-$'], fontsize=largesize)

    # replace color of bar 2,3
    ax.patches[1].set_facecolor(chumanp)
    ax.patches[3].set_facecolor(chumanm)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    labels = ['LLM', 'Human']
    handles = ax.patches[0:2]
    ax.legend(handles, labels, loc='upper right', fontsize=smallsize, frameon=False, title='')

    fig.tight_layout()
    fig.savefig('partial_lr_llm_human.pdf', bbox_inches='tight')

def lr_partial_single(exp_name, name, df, ylim=None, ylabel=False):

    palette = palette_single

    df = df[df['cognitive_model'] == 'Model_2alpha']

    if exp_name == 'partial':
        melted = df.melt(value_vars=['alpha_pos', 'alpha_neg'])
        fig, ax = plt.subplots(1,1, figsize=(1.4,1.85), dpi=dpi)
    #ax.set_xticklabels([r'$\alpha^+$',r'$\alpha^-$'], fontsize=largesize)

    elif exp_name == 'full':
            melted = df.melt(value_vars=['alpha_conf', 'alpha_disconf'])
            fig, ax = plt.subplots(1,1, figsize=(1.3,1.85), dpi=dpi)

    sns.barplot(data=melted,
                x='variable',
                y='value',
                color=palette[name],
                capsize=capsize,
                errwidth=errwidth,
                saturation=1,
                ax=ax)
    sns.despine(ax=ax)

    ax.set_ylim(ylim) 

    if exp_name == 'partial':
        ax.set_xticklabels([r'$\alpha^+$',r'$\alpha^-$'], fontsize=largesize)
    elif exp_name == 'full':
        ax.set_xticklabels([r'$\alpha^{C}$',r'$\alpha^{D}$'], fontsize=smallsize)
    if ylabel:
        ax.set_ylabel('Learning rate', fontsize=largesize)
    else:
        ax.set_ylabel(' ', fontsize=largesize)
    ax.set_xlabel(' ', fontsize=smallsize)
    ax.tick_params(axis='y',labelsize=smallsize)

    fig.tight_layout()
    fig.savefig(f'{exp_name}_{name}_single.pdf', bbox_inches='tight')

def generate_model_comparison(df, labels, name):
    """
    Generate and save model comparison plots for a given dataframe.
    """
    melt = calculate_posterior_probabilities(df)

    if name.startswith('pp_full'):
        fig, ax = plt.subplots(1,1, figsize=(1.5,2.045), dpi=dpi)
    elif name.startswith('pp_agency'):
        fig, ax = plt.subplots(1,1, figsize=(1.6,2.05), dpi=dpi)
        
    sns.barplot(data=melt, 
                x='fitting_model', 
                y='bic', 
                color='#b89f53', 
                errwidth=errwidth, 
                capsize=capsize, 
                ax=ax)
    sns.despine(ax=ax)

    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(2), labels)
    ax.set_xlabel('Cognitive Model')
    ax.set_ylabel('PP')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.hlines(y=melt.mean(numeric_only=True)['bic'], 
            xmin=-0.5, 
            xmax=1.5, 
            linestyle='--', 
            alpha=0.5, 
            color='grey', 
            linewidth=1)

    fig.tight_layout()
    fig.savefig(f'{name}.pdf', bbox_inches='tight')

def load_models():
    """
    Load and return the model comparison and learning rate comparison dataframes for all experiments.
    """
    partial_path = ['../llm/partial/data/claude-1/CM_fit_q0.25.csv',
            '../human/partial/data/CM_fit_q0.25.csv',
            '../meta-rl/partial/data/CM_fit_q0.25.csv']
    full_path = ['../llm/full/data/claude-1/CM_fit_full.csv',
            '../human/full/data/CM_fit_full.csv',
            '../meta-rl/full/data/CM_fit_full.csv']
    agency_path = ['../llm/agency/data/claude-1/CM_fit.csv',
            '../human/agency/data/CM_fit.csv',
            '../meta-rl/agency/data/CM_fit.csv']
    path = [partial_path, full_path, agency_path]
    exp = ['partial', 'full', 'agency']
    names = ['llm', 'human', 'meta-rl']

    # create empty pandas df
    merged = pd.DataFrame()

    for i, p in enumerate(path):
        for j, f in enumerate(p):
            df = pd.read_csv(f)
            df['exp_name'] = exp[i]
            df['agent'] = names[j]
            merged = pd.concat([merged, df], ignore_index=True)
    return merged

def four_learning_rates(exp_name, name, df, ylim=None, legend=False, ylabel=False):
    """
    Generate and save learning rate comparison plots for four learning rates.
    """
    palette = palette_two
    df = df[df['cognitive_model'] == 'Model_4alpha']

    if exp_name == 'full':
        df = df.melt(value_vars=['alpha_pos_chosen', 'alpha_pos_unchosen', 'alpha_neg_chosen', 'alpha_neg_unchosen'])
        # add a column with the condition chosen/unchosen
        df['condition'] = np.where(df['variable'].str.contains('unchosen'), 'unchosen', 'chosen')
        # rename variabe to be just free + or free -
        df['variable'] = df['variable'].str.replace('_chosen', '')
        df['variable'] = df['variable'].str.replace('_unchosen', '')

    elif exp_name == 'agency':
        df = df.melt(value_vars=['alpha_pos_free', 'alpha_neg_free', 'alpha_pos_forced', 'alpha_neg_forced'])
        df['condition'] = np.where(df['variable'].str.contains('free'), 'free', 'forced')
        df['variable'] = df['variable'].str.replace('_free', '')
        df['variable'] = df['variable'].str.replace('_forced', '')

    sns.catplot(x="condition", 
                y="value", 
                hue="variable", 
                data=df, 
                kind="bar", 
                palette=palette[name], 
                legend=False,  
                height=2, 
                aspect=1,
                errwidth=errwidth,
                saturation=1,
                capsize=capsize)

    plt.xticks([-0.2,0.2, 0.8, 1.2])
    # add alpha + and alpha - as text to x axis 
    y_pos = -0.07
    if ylim == (0, 0.7): 
        y_pos -= 0.02    
    plt.text(-0.3, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(0.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(0.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(1.1, y_pos, r'$\alpha^-$', fontsize=largesize)

    labels = ['Chosen', 'Unchosen'] if exp_name == 'full' else ['Free', 'Forced']
    plt.annotate(labels[0], xy=(0.25, -0.14), xytext=(0.25, -0.3), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.annotate(labels[1], xy=(0.75, -0.14), xytext=(0.75, -0.3), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.ylim(ylim)
    plt.yticks(fontsize=smallsize)

    if legend: 
        legend_elements = [Patch(facecolor=cllmp, label='LLM'),
                            Patch(facecolor=chumanp, label='Human')]
        plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=smallsize)
        
    plt.xlabel(' ')
    plt.ylabel(' ')
    if ylabel:
        plt.ylabel('Learning rate')

    plt.savefig(f'{exp_name}_{name}_lr_ylim_{ylim}.pdf', bbox_inches='tight')

def confidence_interval(data, confidence: float = 0.95):
    """
    Calculate the confidence interval for a given set of data
    """

    a: np.ndarray = 1.0 * np.array(data)
    n: int = len(a)

    m, se = a.mean(), scipy.stats.sem(a)
    tp = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    h = se * tp
    return h

def three_learning_rates(name, df, legend=False, ylabel=False):
    """
    Generate and save learning rate comparison plots for three learning rates.
    """
    palette = palette_three

    df = df[df['cognitive_model'] == 'Model_3alpha']

    df = df.melt(value_vars=['alpha_pos_free', 'alpha_neg_free', 'alpha_forced'])
    df['condition'] = np.where(df['variable'].str.contains('free'), 'free', 'forced')
    df['variable'] = df['variable'].str.replace('_free', '')
    df['variable'] = df['variable'].str.replace('_forced', '')

    fig, ax = plt.subplots(1,1, figsize=(1.25,1.7), dpi=dpi)
    sns.despine()
    df = df.groupby(['variable', 'condition'])['value'].agg(['mean', confidence_interval]).reset_index()
    # reverse order rows to alpha_pos, alpha_neg, alpha
    # Define a custom sorting order
    sort_order = {'alpha_pos': 1, 'alpha_neg': 2, 'alpha': 3}
    # Sort the DataFrame using the custom order
    df = df.iloc[df['variable'].map(sort_order).argsort()]
    ax.bar([0,0.5,1.25], df['mean'], 
        color=palette[name], 
        width=0.5, 
        yerr=df['confidence_interval'],
        error_kw={'elinewidth': errwidth, 'capsize': 3, 'ecolor':'#424242ff'})

    # set ticklabels to be alpha + and alpha -
    plt.xticks((0,0.5,1.25), labels=[], fontsize=largesize)
    y_pos = -0.06  
    plt.text(-0.11, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(0.39, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(1.14, y_pos, r'$\alpha$', fontsize=largesize)

    plt.ylim(0, 0.5)

    plt.annotate('Free', xy=(0.305, -0.14), xytext=(0.305, -0.3), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))
    plt.annotate('Forced', xy=(0.825, -0.14), xytext=(0.825, -0.3), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=1.05, lengthB=0.5', lw=0.5, color='k'))

    if legend: 
        legend_elements = [Patch(facecolor=cllmp, label='LLM'),
                            Patch(facecolor=chumanp, label='Human')]
        # position legend outside of plot
        ax.legend(handles=legend_elements, loc='upper right', frameon=False, bbox_to_anchor=(1.3, 1), fontsize=smallsize)
        #ax.legend(handles=legend_elements, loc='upper right', frameon=False, xpos=1.1)

    if ylabel:
        plt.ylabel('Learning rate')
        fig.savefig(f'agency_{name}_3lr_ylabel.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'agency_{name}_3lr.pdf', bbox_inches='tight')

def learning_rate_comparison(data):
    """
    Generate and save learning rate comparison plots for different agents and experimental conditions.
    """
    print('partial 2 learning rates')
    lr_partial_human_llm(data)

    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'partial')]
    #lr_partial_single('partial', 'llm', df, ylim=(0,0.9), ylabel=True)
    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'partial')]
    lr_partial_single('partial', 'meta-rl', df, ylim=(0,0.9), ylabel=True)

    print('full 2/conf learning rates')
    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'full')]
    lr_partial_single('full', 'llm', df, ylim=(0,0.5), ylabel=True)
    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'full')]
    lr_partial_single('full', 'human', df, ylim=(0,0.5), ylabel=True)
    df = data[(data['agent'] == 'meta-rl') & (data['exp_name'] == 'full')]
    lr_partial_single('full', 'meta-rl', df, ylim=(0,0.5), ylabel=True)
    
    print('full 4 learning rates')
    ylim = (0, 0.55)
    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'full')]
    four_learning_rates('full', 'llm', df, (0, 0.55), ylabel=True)
    #four_learning_rates('full', 'llm', df, (0, 0.5), ylabel=True)
    #four_learning_rates('full', 'llm', df, (0, 0.7), ylabel=True)
    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'full')]
    four_learning_rates('full', 'human', df, (0, 0.55), legend=True)

    df = data[(data['agent'] == 'meta-rl') & (data['exp_name'] == 'full')]
    four_learning_rates('full', 'meta-rl', df, (0, 0.7), ylabel=True)
    four_learning_rates('full', 'meta-rl', df, (0, 0.6), ylabel=True)


    print('agency 3 learning rates')
    df = data[(data['agent'] == 'llm') & (data['exp_name'] == 'agency')]
    three_learning_rates('llm', df, ylabel=True)
    three_learning_rates('llm', df, ylabel=False)
    df = data[(data['agent'] == 'human') & (data['exp_name'] == 'agency')]
    three_learning_rates('human', df, legend=True)
    df = data[(data['agent'] == 'meta-rl') & (data['exp_name'] == 'agency')]
    three_learning_rates('meta-rl', df, ylabel=True)

def performance_partial_convert():
    """
    Convert and return the performance data for partial experiments.
    """
    # modify human data to include regret
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

    # Convert llm data
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
    llm_partial.drop(columns=['casino', 'mean0', 'mean1', 'reward0', 'reward1'], inplace=True)

    meta_rl = pd.read_csv('../meta-rl/partial/data/exp/simulation_df.csv')
    meta_rl['cue_idx'] = meta_rl.groupby(['test_part_idx', 'cues']).cumcount()
    meta_rl.rename(columns={'test_part_idx': 'run',
                            'cues': 'cue',
                            'actions': 'action',
                            'rewards': 'reward',
                            'regrets': 'regret',
                            'trial_cue': 'cue_idx',
                            'trial': 'trials_idx'}, inplace=True)
    
    # merge
    partial_perf = pd.concat([human_partial, llm_partial, meta_rl], keys=['human', 'llm', 'meta-rl'], names=['agent'])
    partial_perf.reset_index(inplace=True)
    partial_perf.drop(columns=['level_1'], inplace=True)

    return partial_perf

def performance_full_convert():
    """
    Convert and return the performance data for full experiments.
    """
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

    human = pd.read_csv('../human/full/data/full_human.csv')
    human.rename(columns={'high' : 'block_reward_type',}, inplace=True)
    human.drop(['Unnamed: 0'], axis=1, inplace=True)

    # merge llm and human data in column agent
    df = pd.concat([llm, human, meta_rl], keys=['llm', 'human', 'meta-rl'], names=['agent']).reset_index(level=0)
    return df

def performance_agency_convert():
    """
    Convert and return the performance data for agency experiments.
    """
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

    human = pd.read_csv('../human/agency/data/agency_human.csv')
    human.rename(columns={'high': 'block_reward_type'}, inplace=True)
    human.drop(columns=['Unnamed: 0', 'counter_actions'], inplace=True)
    df = pd.concat([llm, human, meta_rl], axis=0, keys=['llm', 'human', 'meta-rl'], names=['agent']).reset_index()
    df['plot_idx'] = df.groupby(['agent', 'run', 'idx', 'context']).cumcount()
    return df

def performance_agents(exp_name, agents, df, x, y):
    """
    Generate and save performance plots for different agents and experimental conditions.
    """

    if exp_name == 'partial':
        figsize = (2.2,2)
    else:
        figsize = (2,2)
    
    if agents[-1] =='meta-rl':
        figsize = (2.2,2)

    fig, ax = plt.subplots(1,1, dpi=dpi, figsize=figsize)
    colors = palette_single
    label = {'llm': 'LLM',
            'human': 'Human',
            'meta-rl': 'Meta-RL'}
        
    colors = [colors[agent] for agent in agents]
    label = [label[agent] for agent in agents]

    df = df[df['agent'].isin(agents)]
    
    sns.lineplot(data=df, 
                x=x, 
                y=y, 
                hue='agent', 
                hue_order=agents,
                ax=ax, 
                palette=colors)
    sns.despine(ax=ax)

    # setup ticks
    ax.set_xlabel('Trial')
    ax.set_ylabel('Regret')

    if exp_name == 'partial':
        ax.set_yticks([0.05, 0.10])
    elif exp_name == 'full':
        ax.set_ylim(0, 0.4)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    elif exp_name == 'agency':
        ax.set_ylim(0, 0.3)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    # setup legend
    handles, labels = ax.get_legend_handles_labels()
    labels = label # rename labels 
    labels = labels[::-1] # reorder labels
    handles = handles[::-1]
    ax.legend(handles, labels, loc='upper right', frameon=False, fontsize=smallsize)

    fig.tight_layout()
    fig.savefig(f'{exp_name}_perf_{agents}.pdf', bbox_inches='tight')

def performance_llm_only(exp_name, df, x, y, hue):
    """
    Generate and save performance plots for LLM only.
    """
    colors = [cllmp, cllmm]
    fig, ax = plt.subplots(1,1, figsize=(2,2), dpi=dpi)

    sns.lineplot(data=df, 
                 x=x, 
                 y=y, 
                 hue=hue, 
                 style=hue, 
                 style_order=[1,0], 
                 hue_order=[0,1], 
                 ax=ax, 
                 palette=colors)
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('Regret')
    sns.despine()

    # legend for ax1 remove border
    handles, labels = ax.get_legend_handles_labels()

    if exp_name == 'full':
        labels = ['Partial', 'Full']
    else:
        labels = ['Free', 'Mixed']
        ax.set_ylim(0, 0.3)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])

    # reorder labels
    handles = handles[::-1]
    labels = labels[::-1]

    ax.legend(handles, labels, loc='upper right', frameon=False,)

    fig.tight_layout()
    fig.savefig(f'{exp_name}_perf_llm_only.pdf', bbox_inches='tight')

def performance():
    """
    Generate and save performance plots for all experiments.
    """

    partial_df = performance_partial_convert()
    full_df = performance_full_convert()
    agency_df = performance_agency_convert()

    performance_agents('partial', ['human', 'llm'], partial_df, 'cue_idx', 'regret')
    performance_agents('partial', ['human', 'llm', 'meta-rl'], partial_df, 'cue_idx', 'regret')

    df = full_df[(full_df['block_feedback_type'] == 1) & (full_df['context'] == 0)] # plot full feedback blocks (1) and free trials only (0)
    performance_agents('full', ['human', 'llm'], df, 'trial', 'regret')
    performance_agents('full', ['human', 'llm', 'meta-rl'], df, 'trial', 'regret')


    df = agency_df[(agency_df['block_forced_type'] == 1) & (agency_df['context'] == 0)] # plot forced blocks (1) and free trials only (0)
    performance_agents('agency', ['human', 'llm'], df, 'plot_idx', 'regret')
    performance_agents('agency', ['human', 'llm', 'meta-rl'], df, 'plot_idx', 'regret')

    df = full_df[(full_df['agent'] == 'llm') & (full_df['context'] == 0)] 
    performance_llm_only('full', df, 'trial', 'regret', 'block_feedback_type')
    df = agency_df[(agency_df['agent'] == 'llm') & (agency_df['context'] == 0)]
    performance_llm_only('agency', df, 'plot_idx', 'regret', 'block_forced_type')

def compare_llms(df, model1, model2, names, ylim=None, legend=False, ylabel=False):
    """
    Generate and save learning rate comparison plots for two learning models.
    """
    df = df[df['cognitive_model'] == 'Model_2alpha']

    df = df[(df['llm'] == model1) | (df['llm'] == model2)]

    df = df.melt(value_vars=['alpha_pos', 'alpha_neg'], id_vars=['llm'])
    df.llm = df.llm.replace({model1: '2.1', model2: '3 Haiku'})

    sns.catplot(x="llm", 
                y="value", 
                hue="variable", 
                data=df, 
                kind="bar", 
                palette=[cllmp, cllmp], 
                order=['2.1', '3 Haiku'],
                legend=False,  
                height=2, 
                aspect=1,
                errwidth=errwidth,
                saturation=1,
                capsize=capsize)

    plt.xticks([-0.2,0.2, 0.8, 1.2])
    # add alpha + and alpha - as text to x axis 
    y_pos = -0.12
    plt.text(-0.3, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(0.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(0.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(1.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    
    # add text as title above each bar
    plt.text(0.5, 1.1, names, fontsize=largesize, ha='center', va='bottom')
   
    if model1.startswith('llama'):
        plt.annotate('Base', xy=(0.25, -0.15), xytext=(0.25, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))
        plt.annotate('Chat', xy=(0.75, -0.15), xytext=(0.75, -0.31), xycoords='axes fraction', 
                    fontsize=smallsize, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))
    elif model1.startswith('claude'):
        plt.annotate('Claude 2.1', xy=(0.25, -0.15), xytext=(0.25, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))
        plt.annotate('3 Haiku', xy=(0.75, -0.15), xytext=(0.75, -0.31), xycoords='axes fraction', 
                    fontsize=smallsize, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.ylim(ylim)
    plt.yticks(fontsize=smallsize)

    if legend: 
        legend_elements = [Patch(facecolor=cllmp, label='LLM'),
                            Patch(facecolor=chumanp, label='Human')]
        plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=smallsize)
        
    plt.xlabel(' ')
    plt.ylabel(' ')
    if ylabel:
        plt.ylabel('Learning rate')

    plt.savefig(f'partial_llms_{model1}_vs_{model2}_lr_ylim_{ylim}.pdf', bbox_inches='tight')

def compare_claude(df, ylim=None, legend=False, ylabel=False):
    """
    Generate and save learning rate comparison plots for Claude.
    """
    df = df[df['llm'].str.contains('claude')]
    df = df[df['cognitive_model'] == 'Model_2alpha']

    df = df.melt(value_vars=['alpha_pos', 'alpha_neg'], id_vars=['llm'])
    df.llm = df.llm.replace({'claude-2.1': 'claude-2', 'claude-3-haiku-20240307': 'claude-3'})
    df['llm'] = pd.Categorical(df['llm'], categories=['claude-1', 'claude-2', 'claude-3'], ordered=True)
    sns.catplot(x="llm", 
                y="value", 
                hue="variable", 
                data=df, 
                kind="bar", 
                palette=[cllmp, cllmp], 
                legend=False,  
                height=2, 
                aspect=1.3,
                errwidth=errwidth,
                saturation=1,
                capsize=capsize)

    plt.xticks([-0.2,0.2, 0.8, 1.2, 1.8, 2.2])
    # add alpha + and alpha - as text to x axis 
    y_pos = -0.12  
    plt.text(-0.3, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(0.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(0.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(1.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(1.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(2.1, y_pos, r'$\alpha^-$', fontsize=largesize)

    #plt.text(-1.3, -0.315, f'Success/failure', fontsize=smallsize, ha='center', va='bottom')
    #if ylabel:
    #    plt.text(1.0, -0.4, f'\# slot machines', fontsize=smallsize, ha='center', va='bottom')

    labels = ['1.2', '2.1', '3 Haiku']
    plt.annotate(labels[0], xy=(0.17, -0.15), xytext=(0.17, -0.315), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.annotate(labels[1], xy=(0.5, -0.15), xytext=(0.5, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.annotate(labels[2], xy=(0.83, -0.15), xytext=(0.83, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.ylim(ylim)
    plt.yticks(fontsize=smallsize)

    if legend: 
        legend_elements = [Patch(facecolor=cllmp, label='LLM'),
                            Patch(facecolor=chumanp, label='Human')]
        plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=smallsize)
        
    plt.xlabel(' ')
    plt.ylabel(' ')
    if ylabel:
        plt.ylabel('Learning rate')

    plt.title(f'Claude', fontsize=largesize)

    plt.savefig(f'claudes.pdf', bbox_inches='tight')

def llm_single(df, model, name, ylim=None, ylabel=False):
    """
    Plot learning rate comparison plots for a single LLM on the partial task.
    """
    palette = palette_single
    df = df[df['cognitive_model'] == 'Model_2alpha']
    df = df[df['llm'] == model]

    melted = df.melt(value_vars=['alpha_pos', 'alpha_neg'])
    fig, ax = plt.subplots(1,1, figsize=(1.333,2.013), dpi=dpi)

    sns.barplot(data=melted,
                x='variable',
                y='value',
                color=palette['llm'],
                capsize=capsize,
                errwidth=errwidth,
                saturation=1,
                ax=ax)
    sns.despine(ax=ax)

    ax.set_ylim(ylim) 

    ax.set_xticklabels([r'$\alpha^+$',r'$\alpha^-$'], fontsize=largesize)
    
    if ylabel:
        ax.set_ylabel('Learning rate', fontsize=largesize)
    else:
        ax.set_ylabel(' ', fontsize=largesize)
    ax.set_xlabel(' ', fontsize=smallsize)
    ax.tick_params(axis='y',labelsize=smallsize)
    ax.get_yaxis().set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
    plt.title(name, fontsize=largesize)

    fig.tight_layout()
    fig.savefig(f'partial_llm_{model}.pdf', bbox_inches='tight')

def load_llms():
    """
    Load and return the LLM data for the partial task.
    """
    base_path = '../llm/partial/data/'
    llms = glob(base_path + '*/')
    llms.remove("../llm/partial/data/claude-3-sonnet-20240229/")
    merged = pd.DataFrame()
    for llm in llms:
        df = pd.read_csv(f'{llm}CM_fit_q0.5.csv')
        df['llm'] = llm.split('/')[-2]
        merged = pd.concat([merged, df], ignore_index=True)
    return merged

def plot_llm(): 
    """
    Generate and save learning rate comparison plots for LLMs.
    """
    data = load_llms()
    #llm_single(data, 'claude-1', 'Claude-1', ylim=(0, 0.9), ylabel=True)
    llm_single(data, 'gpt-4', 'GPT-4', ylim=(0, 0.9), ylabel=True)
    #compare_llms(data, 'claude-1', 'gpt-4',ylabel=True, ylim=(0, 0.9))
    compare_llms(data, 'llama-2-7', 'llama-2-7-chat', 'Llama-2-7B', ylabel=True, ylim=(0, 0.9))
    compare_llms(data, 'llama-2-70', 'llama-2-70-chat', 'Llama-2-70B', ylabel=True, ylim=(0, 0.9))
    compare_claude(data, ylabel=True, ylim=(0, 0.9))

def plot_addition(): 
    """
    Generate and save learning rate comparison plots for Claude on the additional robustness task.
    """
    data = pd.read_csv('../llm/partial_addition/data/claude-1/CM_fit_addition.csv')
    compare_addition(data, option="1.0_0.0",ylabel=True, ylim=(0, 0.9))
    compare_addition(data, option="0.5_-0.5",ylabel=True, ylim=(0, 0.9))

def compare_addition(df, option, ylim=None, legend=False, ylabel=False):
    """
    Plot generation for the additional robustness task.
    """
    df = df[df['cognitive_model'] == 'Model_2alpha']
    df = df[df['reward'] == '1.0_0.0']

    df = df.melt(value_vars=['alpha_pos', 'alpha_neg'], id_vars=['options'])
    df['options'] = pd.Categorical(df['options'], [2,3,4])

    sns.catplot(x="options", 
                y="value", 
                hue="variable", 
                data=df, 
                kind="bar", 
                palette=[cllmp, cllmp], 
                legend=False,  
                height=2, 
                aspect=1.3,
                errwidth=errwidth,
                saturation=1,
                capsize=capsize)

    plt.xticks([-0.2,0.2, 0.8, 1.2, 1.8, 2.2])
    # add alpha + and alpha - as text to x axis 
    y_pos = -0.12  
    plt.text(-0.3, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(0.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(0.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(1.1, y_pos, r'$\alpha^-$', fontsize=largesize)
    plt.text(1.7, y_pos, r'$\alpha^+$', fontsize=largesize)
    plt.text(2.1, y_pos, r'$\alpha^-$', fontsize=largesize)

    #plt.text(-1.3, -0.315, f'Success/failure', fontsize=smallsize, ha='center', va='bottom')
    if ylabel:
        plt.text(1.0, -0.4, f'\# slot machines', fontsize=smallsize, ha='center', va='bottom')

    labels = ['2', '3', '4']
    plt.annotate(labels[0], xy=(0.17, -0.15), xytext=(0.17, -0.315), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.annotate(labels[1], xy=(0.5, -0.15), xytext=(0.5, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.annotate(labels[2], xy=(0.83, -0.15), xytext=(0.83, -0.31), xycoords='axes fraction', 
                fontsize=smallsize, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.1, lengthB=0.5', lw=0.5, color='k'))

    plt.ylim(ylim)
    plt.yticks(fontsize=smallsize)

    if legend: 
        legend_elements = [Patch(facecolor=cllmp, label='LLM'),
                            Patch(facecolor=chumanp, label='Human')]
        plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=smallsize)
        
    plt.xlabel(' ')
    plt.ylabel(' ')
    if ylabel:
        plt.ylabel('Learning rate')

    plt.title(f'{option}'.replace('_', '/'), fontsize=largesize)

    plt.savefig(f'partial_addition_rew_{option}_lr_ylim_{ylim}.pdf', bbox_inches='tight')

if __name__ == "__main__":
    data = load_models()
    model_comparisons(data)
    learning_rate_comparison(data)
    performance()
    plot_llm()
    plot_addition()
