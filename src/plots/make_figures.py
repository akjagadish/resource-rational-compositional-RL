import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from matplotlib.ticker import MaxNLocator
import torch.nn as nn
from groupBMC.groupBMC import GroupBMC
sns.set_context('paper')
# from tueplots import bundles
# from tueplots import figsizes
# # plt.rcParams.update(bundles.neurips2022())
# with plt.rc_context(bundles.neurips2022()):
#     pass
import sys
# sys.path.append('/plots/' )
sys.path.append('../models/policy/' )
sys.path.append('../models/metaRL/')
from tasks.sample_data import sample_rewards
from utils import load_data, FitSoftmax
from model_comparison import return_model_comparison_metrics, return_mll_estimates, return_mll_estimates_per_trial
from load_human_behavior import load_behavior, load_percondition_metrics
import statsmodels.formula.api as smf
from torch.nn import KLDivLoss 
from analyze_errors import return_errors
import pickle

# COLORS = {'rl3':'#FF6F00', #"#D55E00", 
#           'rl2':'#E29152', #'#0571D0', 
#           'mean_tracker':'#EFD381', #FFC107', 
#           'mean_tracker_compositional':'#FFC107', #A1E6DB',
#           'rbf_nocontext_nomemory':'#D26E92', 
#           'simple_grammar_constrained':'#CE1558',
#           #'simple_grammar_constrained_noncompositonal':'#EF9EBB',
#           'compositional':'#035043', 
#           'noncompositional':'#A1E6DB',
#           'linear':'#332288',#882255
#           'periodic':'#AA4499',
#           'optimal': '#E69F00'}


COLORS = {'compositional':'#117733', 
          'noncompositional':'#96CAA7',
          'linear':'#88CCEE',#882255
          'periodic':'#CC6677',
          'optimal': '#D6BF4D',
          'mean_tracker':'#882255', #332288
          'mean_tracker_compositional':'#882255', #AA4499',
          'rbf_nocontext_nomemory':'#44AA99', 
          'simple_grammar_constrained':'#44AA99',
          #'simple_grammar_constrained_noncompositonal':'#EF9EBB',
          'rl2':'#E2C294', #'#0571D0', 
          'rl3':'#DA9138', #"#D55E00", 
          }

list_of_labels = {'optimal': 'Optimal arm', 
                        'corner': 'Corner arms', 
                        'corner_optimal': 'Optimal corner arms', 
                        'non_corner_optimal': 'Optimal Non-corner  arms',  
                        'corner_non_optimal': 'Non-optimal corner arms', 
                        'non_corner_non_optimal': 'Neither optimal nor corner arm', 
                        'non_optimal': 'Non-optimal arms',
                        'phasic_non_optimal': 'Non-optimal phasic arms',
                        'neither': 'Neither of the above'}

def generate_reward_structures(sub_task=2, n_runs=1, FONTSIZE=20, FIGSIZE=(6,4)):
    
    condition = 'compositional'
    seed = 1

    #set up the color scheme
    c = np.arange(n_runs) + 1
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    colormaps = mpl.cm.Blues if sub_task==0 else mpl.cm.Reds if sub_task==1 else mpl.cm.grey
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=colormaps)
    cmap.set_array([])
    round_label = np.arange(1, n_runs+1, 10)

    for rule in ['add', 'changepoint']:
        # set up figure for kernels
        #f, ax = plt.subplots(1, 1, figsize=FIGSIZE
        # use these for kernels: ['linear']:, ['periodic']: 
        for lin in  ['linneg', 'linpos']: 
            for per in  ['perodd', 'pereven']:
                f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
                # sample data
                block = [lin, per, 'linperiodic']
                description = block if condition == 'compositional' else ['linperiodic']
                _, Y, _ = sample_rewards(n_runs, block, rule, seed_val=seed, noise_per_arm=True, noise_var=0.00)
                ax.plot(Y[0, sub_task].T.numpy(), c='k', lw=3)
                # uncomment for kernels
                # for run in range(n_runs):
                #     ax.plot(Y[run, sub_task].T.numpy(), lw=3, c=cmap.to_rgba(run + 1))
                arms = ['S', 'D', 'F', 'J', 'K', 'L']
                plt.xticks(np.arange(6), arms)
                plt.yticks([])
                plt.xticks(fontsize=FONTSIZE)
                sns.despine()
                f.tight_layout()
                f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/rewards_{}_{}_{}_{}.svg'.format(sub_task, block[0], block[1], rule), bbox_inches = 'tight')
                plt.show()

def simulated_models_meanregrets(trials=np.arange(0,15), FONTSIZE=25, FIGSIZE=(7,7), path='/eris/u/ajagadish/compositional-reinforcement-learning/modelfits/simulated_data_preds/optimal/'):
    SCALE = 4.
    NUM_TRIALS = 5
    plot_trials = np.arange(NUM_TRIALS)+1
    TASK_ID = 2
    mean_random_regrets = {'changepoint': 2.89, 'add': 4.05} #2.89
    for rule in ['add', 'changepoint']:
        changepoint = True if rule=='changepoint' else False
        #TODO: non-curriculum conditions
        # RL3
        _, _, _, rl3_regrets_compositional, _ = torch.load(f'../../RL3NeurIPS/simulations/stats_changepoint={changepoint}_jagadish2022curriculum-v0_unconstrainedFalse_entropylossTrue_policyzeros.pth')
        #_, _, _, rl3_regrets_noncompositional, true_best_action = torch.load(f'')
        
        # RL2
        rl2_regrets_compositional = rl3_regrets_compositional[-1][:100, trials]  * SCALE # last dl is now the new RL2
        #rl2_regrets_noncompositional = rl3_regrets_noncompositional[-1].mean(0)[trials].sum() # last dl is now the new RL2

        # Mean-Tracker
        mt_regrets_compositional = np.load(f'{path}/mean_tracker/regrets_mean_tracker_simulated_greedy_compositional_{rule}_0.0.npy') * SCALE
        #mt_regrets_noncompositional, _ = np.load()
        #n_samples = mt_regrets_compositional.shape[0]

        # Mean-Tracker Compositional
        mtc_regrets_compositional = np.load(f'{path}/mean_tracker_compositional/regrets_mean_tracker_compositional_simulated_greedy_compositional_{rule}_0.0.npy')  * SCALE
        #mtc_regrets_noncompositional, _ = np.load()

        # GP-RBF
        gp_rbf_regrets = np.load(f'{path}/gp_rbf/regrets_gp_rbf_simulated_greedy_compositional_{rule}_0.0.npy') * SCALE
        #gp_regrets_nc = np.load(f'{path}/gp_rbf/regrets_gp_rbf_simulated_greedy_noncompositional_{rule}_0.0.npy')
        #gp_rbf_regrets = gp_rbf_regrets * SCALE * 2 

        # Grammar Model
        gp_compositional_regrets = np.load(f'{path}/gp_compositional/RLDM/regrets_simple_grammar_simulated_compositional_{rule}.npy')  * SCALE * 2
        #gp_compositional_regrets_nc = np.load(f'{path}/gp_compositional/regrets_simple_grammar_simulated_greedy_noncompositional_{rule}_0.0.npy')

        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        n_samples = mt_regrets_compositional.shape[3]
        ax.errorbar(plot_trials+0.01*2, mt_regrets_compositional.mean((1,2,3)).squeeze()[TASK_ID], 
                    yerr=mt_regrets_compositional.mean((1,2))[0,:, TASK_ID].std(0)/np.sqrt(n_samples), fmt="o--",
                    color=COLORS['mean_tracker'], label="BMT", lw=3)
        ax.errorbar(plot_trials+0.01*3, mtc_regrets_compositional.mean((1,2,3)).squeeze()[TASK_ID], 
                    yerr=mtc_regrets_compositional.mean((1,2))[0,:, TASK_ID].std(0)/np.sqrt(n_samples), fmt="o-",
                    color=COLORS['mean_tracker_compositional'], label='Compositional BMT', lw=3, linestyle='solid')
        ax.errorbar(plot_trials+0.01*2, gp_rbf_regrets.mean((1,2,3)).squeeze()[TASK_ID], 
                    yerr=gp_rbf_regrets.mean((1,2))[0,:, TASK_ID].std(0)/np.sqrt(n_samples), fmt="o--",
                    color=COLORS['rbf_nocontext_nomemory'], label="GPR", lw=3)
        ax.errorbar(plot_trials+0.01*2, gp_compositional_regrets.mean((1,2,3)).squeeze()[TASK_ID],
                    yerr=gp_compositional_regrets.mean((1,2))[0,:, TASK_ID].std(0)/np.sqrt(n_samples), fmt="o-",
                    color=COLORS['simple_grammar_constrained'], label="Compositional GPR", lw=3)
        ax.errorbar(plot_trials+0.01*3, rl2_regrets_compositional.mean(0), 
                    yerr=rl2_regrets_compositional.std(0)/np.sqrt(n_samples), fmt="o-",
                    color=COLORS['rl2'], label='RL$^2$', lw=3, linestyle='solid') #(Curriculum)
        ax.hlines(mean_random_regrets[rule], 1, 5, color='k', linestyles='dotted', lw=5)#, label='Random')
        if rule=='changepoint':
            plt.legend(fontsize=FONTSIZE-4,  loc="upper center", bbox_to_anchor=(.6, 1.06), frameon=False)
        ax.set_xlabel('Trials', fontsize=FONTSIZE)
        ax.set_ylabel('Mean regret', fontsize=FONTSIZE) 
        ax.set_ylim(ymin=0., ymax=5)
        ax.set_xticks(np.arange(1, 6))
        ax.set_xticklabels(np.arange(1, 6, dtype=int), fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/Simulated_Model_MeanRegretTrials_{}.svg'.format(rule), bbox_inches = 'tight')
        plt.show()
        
def simulated_dls_meanregrets(trials=np.arange(0,15), FONTSIZE=25, FIGSIZE=(7,7)):
    use_trials = 'all' if (trials[0]==0 and len(trials)==15) else 'composed'
    step_size = 2
    dls = np.linspace(10, 10000, 1000)[::step_size]
    SCALE = 4.
    for rule in ['add', 'changepoint']:
        changepoint = True if rule=='changepoint' else False
        actions, _, _, rl3_regrets, true_best_action = torch.load(f'../../RL3NeurIPS/simulations/stats_changepoint={changepoint}_jagadish2022curriculum-v0_unconstrainedFalse_entropylossTrue_policyzeros.pth')
        rl2_sim = rl3_regrets[-1].mean(0)[trials].sum() * SCALE
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)#, dpi=100)
        rl3_sim = pd.DataFrame.from_dict({'dls': np.log10(dls), 'regrets': rl3_regrets.mean(1)[::step_size, trials].sum(1)*SCALE})
        graph=sns.regplot(x='dls',y='regrets', data=rl3_sim, logx=False, scatter_kws={"color": COLORS['rl3'], "s": 100}, line_kws={"color": COLORS['compositional']})
        graph.axhline(rl2_sim, lw=4, ls=':', color=COLORS['rl2'])
        ax.set_xlim(xmin=0.9, xmax=4.1)
        ax.set_ylim(ymin=0., ymax=21.)
        ax.text(1.1, rl2_sim+0.05, 'RL$^2$', color=COLORS['rl2'], fontsize=FONTSIZE)
        if rule == 'add':
            plt.legend(loc='upper right', fontsize=FONTSIZE-2, frameon=False)
        ax.set_xlabel('Description length (in log-scale)', fontsize=FONTSIZE)
        ax.set_ylabel(f'Mean regret', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/SimulatedDescriptionLengths_MeanRegrets_{}_{}.svg'.format(use_trials, rule), bbox_inches = 'tight')
        plt.show()

def simulated_dls_meanprobs(trials=np.arange(0,15), FONTSIZE=25, FIGSIZE=(7,7)):
    use_trials = 'all' if (trials[0]==0 and len(trials)==15) else 'first' if (len(trials)==1 and len(trials)==10) else 'composed'
    step_size = 2
    dls = np.linspace(10, 10000, 1000)[::step_size]

    for rule in ['add', 'changepoint']:
        changepoint = True if rule=='changepoint' else False
        actions, _, _, rl3_regrets, true_best_action = torch.load(f'../../RL3NeurIPS/simulations/stats_changepoint={changepoint}_jagadish2022curriculum-v0_unconstrainedFalse_entropylossTrue_policyzeros.pth')
        rl3_actions = (actions==true_best_action).float()
        
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)#, dpi=100)
        rl3_sim = pd.DataFrame.from_dict({'dls': np.log10(dls), 'regrets': rl3_actions.mean(1)[::step_size, trials].mean(1)})
        sns.regplot(x='dls',y='regrets', data=rl3_sim, logx=False, scatter_kws={"color": COLORS['rl3'], "s": 100}, line_kws={"color": COLORS['compositional']})
        ax.set_xlim(xmin=0.9, xmax=4.1)
        if rule == 'add':
            plt.legend(loc='upper right', fontsize=FONTSIZE-2, frameon=False)
        
        ax.set_xlabel('Description length (in log-scale)', fontsize=FONTSIZE)
        name_trials = '1' if len(trials)==1 else 'last'
        ax.set_ylabel(f'p(optimal choice)', fontsize=FONTSIZE) #f'p($a_{name_trials}$)'
        plt.xticks(fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/SimulatedDescriptionLengths_MeanProbs_{}_{}.svg'.format(use_trials, rule), bbox_inches = 'tight')
        plt.show()

        
def human_meanregrets_trials(FONTSIZE=25, FIGSIZE=(8,8)):
    # plt.rcParams.update(figsizes.neurips2022(nrows=2, ncols=3))
    
    EXPS = ['compositional', 'noncompositional']
    LABELS = {'compositional': 'Curriculum','noncompositional':'Non-curriculum'} 
    NUM_TRIALS = 5
    plot_trials = np.arange(NUM_TRIALS)
    mean_random_regrets = {'changepoint': 2.89, 'add': 4.05} #2.89

    for rule in ['add', 'changepoint']:

        optimal_actions_dict, pooled_rewards_dict, \
        pooled_regrets_dict, pooled_times_dict, pooled_rewards_tasks_dict, \
        pooled_regrets_tasks_dict, pooled_times_tasks_dict, n_subjs = load_percondition_metrics(rule)

        f, ax = plt.subplots(1, 1, figsize=FIGSIZE) #, dpi=100)
        for nn, (exp, label) in enumerate(zip(EXPS, LABELS)):
            ax.errorbar(plot_trials+1+0.1*nn, pooled_regrets_dict[exp].mean(1).mean(0), yerr=pooled_regrets_dict[exp].mean(1).std(0)/np.sqrt(n_subjs[exp]), 
                        color=COLORS[exp], label=LABELS[exp], lw=3, linestyle='solid', fmt='o')
            print(exp, pooled_regrets_dict[exp].mean(1).mean(0), pooled_regrets_dict[exp].mean(1).std(0)/np.sqrt(n_subjs[exp]))
        plt.legend(labels = ['Curriculum', 'Non-curriculum'],  frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.06), ncol=1, fontsize=FONTSIZE-2)
        # if rule == 'changepoint':
        #     plt.legend(labels = ['Curriculum', 'Non-Curriculum'],  frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.06), ncol=1, fontsize=FONTSIZE)
        # # else:
        #     plt.legend(labels = ['_nolegend_', 'Non-Curriculum'], frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=FONTSIZE,)
        ax.hlines(mean_random_regrets[rule], 1, 5, color='k', linestyles='dotted', lw=5, label='Random')
        ax.set_xlabel('Trials', fontsize=FONTSIZE)
        ax.set_ylabel('Mean regret', fontsize=FONTSIZE) 
        ax.set_ylim(ymin=0., ymax=5)
        plt.xticks(fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/Human_MeanRegretTrials_{}.svg'.format(rule), bbox_inches = 'tight')
        plt.show()

def human_meanprobs(first=True, FONTSIZE=25, FIGSIZE=(5,5)):
    
    for rule in ['add', 'changepoint']:
        data = load_behavior(rule)
        optimal_actions_dict, pooled_rewards_dict, \
        pooled_regrets_dict, pooled_times_dict, pooled_rewards_tasks_dict, \
        pooled_regrets_tasks_dict, pooled_times_tasks_dict, n_subjs = load_percondition_metrics(rule)
        data_comp = data[(data['experiment']=='compositional')]
        data_noncomp = data[(data['experiment']=='noncompositional')]
        prob_opt_noncompositonal_first = np.stack([data_noncomp.iloc[subj].optimal_actions[:, 0].mean() if first else data_noncomp.iloc[subj].optimal_actions.mean(1) for subj in range(n_subjs['noncompositional'])])
        prob_opt_compositional_first = np.stack([data_comp.iloc[subj].optimal_actions[2::3, 0].mean() if first else data_comp.iloc[subj].optimal_actions[2::3].mean(1) \
                                                  for subj in range(n_subjs['compositional'])])

        means = [prob_opt_compositional_first.mean(), prob_opt_noncompositonal_first.mean()]
        std_errors = [prob_opt_compositional_first.std()/np.sqrt(n_subjs['compositional']), prob_opt_noncompositonal_first.std()/np.sqrt(n_subjs['noncompositional'])]
        conditions = ['Curriculum', 'Non-curriculum']
        colors = [COLORS['compositional'], COLORS['noncompositional']]

        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        #ax.errorbar(np.arange(0,2), means, yerr=std_errors, color=[COLORS['compositional'], COLORS['noncompositional']], lw=3, fmt='o') #linestyle='solid')
        ax.bar(np.arange(0,2), means, color=colors, label=conditions)
        ax.errorbar(np.arange(0,2), means, yerr=std_errors, color='k', lw=3, fmt='o') #linestyle='solid')
        plt.xticks(np.arange(0,2))
        plt.yticks(fontsize=FONTSIZE-2)
        ax.set_xlabel('Conditions', fontsize=FONTSIZE)
        trials = '1' if first else 'last'
        ax.set_ylabel(f'p(optimal choice)', fontsize=FONTSIZE)#$a_{trials}$
        ax.set_xticklabels(conditions, fontsize=FONTSIZE-2)#['', '']
        ax.set_ylim(ymin=0., ymax=.4 if first else .5)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/Human_MeanProbs_{}.svg'.format(rule), bbox_inches = 'tight')
        plt.show()

def histogram_peoplechoices(trials=np.array([0]), FONTSIZE=25, FIGSIZE=(7, 7), include_optimal=False):
    use_trial = 'first' if len(trials) == 1 else 'all' 

    for rule in ['add', 'changepoint']:
        data = load_behavior(rule)
        comp_data = data[(data['experiment']=='compositional')]
        best_actions =np.stack(comp_data.best_actions)[:, 2::3].reshape(-1)
        
        # load noncompositonal actions
        experiment = 'noncompositional'
        data = load_behavior(rule)
        orig_noncomp_actions = np.stack(data[(data['experiment']==experiment)].actions)

        if rule == 'add':
            orig_actions, _ = torch.load('/u/ajagadish/compositional-reinforcement-learning/modelfits/rl3_to_participants/rl3_add_curriculum.pth')  
        else:
            orig_actions, _, linfirst = torch.load('/u/ajagadish/compositional-reinforcement-learning/modelfits/rl3_to_participants/rl3_changepoint_curriculum.pth')

        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        if include_optimal:
            ax.hist([orig_actions[..., 10+trials].reshape(-1), orig_noncomp_actions[..., trials].reshape(-1), best_actions], 
                bins=np.arange(7), density=True, width=0.267, color=[COLORS['compositional'], COLORS['noncompositional'], COLORS['optimal']], align='left', label=['Curriculum', 'Non-curriculum', 'Optimal']);
        else:
            ax.hist([orig_actions[..., 10+trials].reshape(-1), orig_noncomp_actions[..., trials].reshape(-1)], 
                bins=np.arange(7), density=True, width=0.39, color=[COLORS['compositional'], COLORS['noncompositional']], align='left', label=['Curriculum', 'Non-curriculum']);
        
        plt.ylim([0., .5])
        ax.set_xlabel('Arms', fontsize=FONTSIZE)
        #trials = '1'if len(trials) == 1 else 'all'
        ax.set_ylabel(f'p(arm)', fontsize=FONTSIZE) #f'p($a_{trials}$)'
        arms = ['S', 'D', 'F', 'J', 'K', 'L']
        plt.xticks(np.arange(6), arms, fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        # if rule == 'add':
        #     plt.legend(fontsize=FONTSIZE, frameon=False)
        plt.legend(fontsize=FONTSIZE-2, frameon=False, loc="upper left", bbox_to_anchor=(0.25, 1.), ncol=1)#(0.5, 1.15)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/HumanChoices_{}_{}.svg'.format(use_trial, rule), bbox_inches = 'tight')
        plt.show()

def posterior_model_frequency_mlls_perrule(fit_trials='first', horizontal=True, FIGSIZE=(5,5), FONTSIZE=25):
    for rule in ['add', 'changepoint']:
        result = {}
        models = ['mean_tracker', 'mean_tracker_compositional','rbf_nocontext_nomemory', 'simple_grammar_constrained', 'rl2', 'rl3']  #
        if horizontal:
            models.reverse()
        for fit_subtasks in ['composed']:
            #L = return_mll_estimates(models, rule=rule, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
            L = return_mll_estimates_per_trial(models, rule=rule, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
            LogEvidence = np.stack(L)
            result[fit_subtasks] = GroupBMC(LogEvidence).get_result()

        # rename models for plot
        colors = [COLORS[model_name] for model_name in models]
        models = ['MLP' if model_id == 'nn_uvfa' else model_id for model_id in models]
        models = ['RR-RL$^2$' if model_id == 'rl3' else model_id for model_id in models]
        models = ['RL$^2$' if model_id == 'rl2' else model_id for model_id in models]
        models = ['GPR' if model_id == 'rbf_nocontext_nomemory' else model_id for model_id in models]#Gaussian Process Regression
        models = ['Compositional'+"\n"+'GPR' if model_id == 'simple_grammar_constrained' else model_id for model_id in models]
        models = ['BMT' if model_id == 'mean_tracker' else model_id for model_id in models]#Bayesian Mean-Tracker
        models = ['Compositional'+"\n"+'BMT' if model_id == 'mean_tracker_compositional' else model_id for model_id in models]
        models = ['Random' if model_id == 'random' else model_id for model_id in models]
        
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        
        if horizontal:
            # composed
            ax.barh(np.arange(len(models)), result['composed'].frequency_mean, xerr=result['composed'].frequency_var, align='center', color=colors, height=0.6)#, edgecolor='k')#, hatch='//', label='Compostional Subtask')
            # plt.legend(fontsize=FONTSIZE-4, frameon=False)
            ax.set_ylabel('Models', fontsize=FONTSIZE)
            # ax.set_xlim(0, 0.7)
            ax.set_xlabel('Posterior model frequency', fontsize=FONTSIZE) 
            plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-3.)
            plt.xticks(fontsize=FONTSIZE-2)
        else:
            # composed
            ax.bar( np.arange(len(models))-0.22, result['composed'].frequency_mean, align='center', color='w', width=0.4, edgecolor='k', hatch='//', label='Compostional Subtask')
            ax.errorbar(np.arange(len(models))-0.22, result['composed'].frequency_mean, result['composed'].frequency_var, c='red',fmt='.r', lw=3)
            # plt.legend(fontsize=FONTSIZE, frameon=False)
            ax.set_xlabel('Models', fontsize=FONTSIZE)
            ax.set_ylim(0, 0.7)
            ax.set_ylabel('Posterior model frequency', fontsize=FONTSIZE) 
            plt.xticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-3.)#, rotation=45)
            plt.yticks(fontsize=FONTSIZE-2)

        print(models, result['composed'].frequency_mean, result['composed'].frequency_var)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/PosteriorModelFrequency_MLLS_{}_{}.svg'.format(fit_subtasks, rule), bbox_inches = 'tight')
        plt.show()

def exeedance_probability_mlls_perrule(fit_trials='first', horizontal=True, FIGSIZE=(5,5), FONTSIZE=25):
    for rule in ['add', 'changepoint']:
        result = {}
        models = ['mean_tracker', 'mean_tracker_compositional', 'rbf_nocontext_nomemory', 'simple_grammar_constrained', 'rl2', 'rl3']
        if horizontal:
            models.reverse()
        for fit_subtasks in ['composed']:
            #L = return_mll_estimates(models, rule=rule, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
            L = return_mll_estimates_per_trial(models, rule=rule, fit_subtasks=fit_subtasks, fit_trials=fit_trials)
            LogEvidence = np.stack(L)
            result[fit_subtasks] = GroupBMC(LogEvidence).get_result()

        # rename models for plot
        colors = [COLORS[model_name] for model_name in models]
        models = ['MLP' if model_id == 'nn_uvfa' else model_id for model_id in models]
        models = ['RR-RL$^2$' if model_id == 'rl3' else model_id for model_id in models]
        models = ['RL$^2$' if model_id == 'rl2' else model_id for model_id in models]
        models = ['GPR' if model_id == 'rbf_nocontext_nomemory' else model_id for model_id in models]#Gaussian Process Regression
        models = ['Compositional'+"\n"+'GPR' if model_id == 'simple_grammar_constrained' else model_id for model_id in models]
        models = ['BMT' if model_id == 'mean_tracker' else model_id for model_id in models]#Bayesian Mean-Tracker
        models = ['Compositional'+"\n"+'BMT' if model_id == 'mean_tracker_compositional' else model_id for model_id in models]
        models = ['Random' if model_id == 'random' else model_id for model_id in models]
        
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        
        
        if horizontal:
            # composed
            ax.barh(np.arange(len(models)), result['composed'].exceedance_probability, align='center', color=colors, height=0.6)#, hatch='//', label='Compostional Subtask')
            # plt.legend(fontsize=FONTSIZE-4, frameon=False)
            ax.set_ylabel('Models', fontsize=FONTSIZE)
            # ax.set_xlim(0, 0.7)
            ax.set_xlabel('Exceedance probability', fontsize=FONTSIZE) 
            plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-3.)
            plt.xticks(fontsize=FONTSIZE-2)
        else:
            # composed
            ax.bar( np.arange(len(models))-0.22, result['composed'].exceedance_probability, align='center', color=colors, width=0.4, edgecolor='k')#, hatch='//', label='Compostional Subtask')
            # plt.legend(fontsize=FONTSIZE, frameon=False)
            ax.set_xlabel('Models', fontsize=FONTSIZE)
            ax.set_ylim(0, 0.7)
            ax.set_ylabel('Exceedance probability', fontsize=FONTSIZE) 
            plt.xticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-5.5)#, rotation=45)
            plt.yticks(fontsize=FONTSIZE-2)
        print(models, result['composed'].exceedance_probability)
        sns.despine()
        f.tight_layout()
        # f.savefig('/eris/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/ExceedanceProbability_MLLS_{}.svg'.format(fit_subtasks), bbox_inches = 'tight')
        plt.show()

def fitted_model_simulated_meanprobs(fit_subtasks='all', first=True, FONTSIZE=25, FIGSIZE=(8,8)):
    
    EXPS = ['compositional']#, 'noncompositional']
    LABELS = {'compositional': 'Curriculum'}#,'noncompositional':'Non-Curriculum'} 
    mean_random_regrets = {'changepoint': 3., 'add': 4.05} #2.89 from 1000 simulations

    for rule in ['add', 'changepoint']:
        data = load_behavior(rule)
        optimal_actions_dict, pooled_rewards_dict, \
        pooled_regrets_dict, pooled_times_dict, pooled_rewards_tasks_dict, \
        pooled_regrets_tasks_dict, pooled_times_tasks_dict, n_subjs = load_percondition_metrics(rule)
        data_comp = data[(data['experiment']=='compositional')]
        data_noncomp = data[(data['experiment']=='noncompositional')]
        means, std_errors = [], []

        # people probs
        prob_opt_noncompositonal_first = np.stack([data_noncomp.iloc[subj].optimal_actions[:, 0].mean() if first else data_noncomp.iloc[subj].optimal_actions.mean(1) for subj in range(n_subjs['noncompositional'])])
        prob_opt_compositional_first = np.stack([data_comp.iloc[subj].optimal_actions[2::3, 0].mean() if first else data_comp.iloc[subj].optimal_actions[2::3].mean(1) \
                                                  for subj in range(n_subjs['compositional'])])
        means.append(prob_opt_compositional_first.mean())
        std_errors.append(prob_opt_compositional_first.std()/np.sqrt(n_subjs['compositional']))

        # RL3
        n_participants = 92 if rule=='add' else 109
        SCALE = 4.
        changepoint = False if rule == 'add' else True
        full = True if fit_subtasks=='all' else False
        actions, _, _, _, true_best_action = torch.load(f'../../RL3NeurIPS/data/RL3Fits/fitted_simulations/all_stats_changepoint={changepoint}_full={full}_entropyTruejagadish2022curriculum-v0_pertrial0.pth')
        sim_rl3_probs = (actions==true_best_action).to(torch.float)[:, :20]
        means.append(sim_rl3_probs.mean(1).mean(0)[10])
        std_errors.append(sim_rl3_probs.mean(1).std(0)[10]/np.sqrt(sim_rl3_probs.shape[0]))

        # Mean-tracker-compositional
        TASK_ID = 2
        if rule == 'changepoint':
            actions, _, _, best = np.load(f'/notebooks/modelfits/simulated_data_preds/mean_tracker_compositional/stats_mean_tracker_compositional_simulated_compositional_{rule}_composed.npy')
            mtc_probs = (actions==best).squeeze().mean(1).mean(1)[:, TASK_ID, 0]
            means.append(mtc_probs.mean(0))
            std_errors.append(mtc_probs.std(0)/np.sqrt(n_participants))
        else:
            actions, _, _, best = np.load(f'/notebooks/modelfits/simulated_data_preds/mean_tracker_compositional/stats_mean_tracker_compositional_simulated_compositional_{rule}_composed.npy')
            mtc_probs = (actions==best).squeeze().mean(1).mean(1)[:, TASK_ID, 0]
            means.append(mtc_probs.mean(0))
            std_errors.append(mtc_probs.std(0)/np.sqrt(n_participants))
           

        models = ['People', 'RR-RL$^2$', 'Compositional'+"\n"+'BMT']
        colors = [COLORS['compositional'], COLORS['rl3'], COLORS['mean_tracker_compositional']]
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        print(means, std_errors)
        #ax.hlines(mean_random_regrets[rule], -.5, 1.5, color='r', linestyles='dotted', lw=5, label='Random')
        ax.bar(np.arange(0, len(models)), means, color=colors, label=models)
        ax.errorbar(np.arange(0, len(models)), means, yerr=std_errors, c='k', lw=3, fmt='.') #linestyle='solid')
        plt.xticks(np.arange(0, len(models)))
        plt.yticks(fontsize=FONTSIZE-2)
        ax.set_xlabel('Models', fontsize=FONTSIZE)
        trials = '1' if first else 'last'
        ax.set_ylabel(f'p(optimal choice)', fontsize=FONTSIZE)#f'p($a_{trials}$)'
        ax.set_xticklabels(models, fontsize=FONTSIZE-2)
        # ax.set_ylim(ymin=0., ymax=.4)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/ModelSimulated_MeanProbs_{}_{}.svg'.format(fit_subtasks, rule), bbox_inches = 'tight')
        plt.show()


def fitted_dls_grouped_human_choices(trials=np.arange(0,15), FONTSIZE=25, FIGSIZE=(7, 7)):

    use_trials = 'all' if trials[0]==0 and len(trials)==15 else 'composed'
    for rule in ['add', 'changepoint']:
        data = load_behavior(rule)
        data_comp = data[(data['experiment']=='compositional')]
        order_rl3, order_baselines = np.load(f'/notebooks/modelfits/rl3_to_participants/rl3_subject_order_{rule}.npy'), np.load(f'/notebooks/modelfits/rl3_to_participants/baselines_subject_order_{rule}.npy')
        
        full = True if use_trials=='all' else False
        full_path = f'_full={full}_changepoint=True.pth' if rule == 'changepoint' else  f'_full={full}_changepoint=False.pth' 
        dls = torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/dls_rl3_firsttrial_{full_path}') if len(trials)==1 else torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/dls_rl3{full_path}')
        dls =(dls[order_rl3] + 1)*10

        # sort ref rewards and use those indices to sort regrets
        prob_opt_compositional_first = np.stack(data_comp.optimal_actions)[:, 2::3, 0].mean(1)[order_baselines] 
        prob_opt_compositional_last =  np.stack(data_comp.optimal_actions)[:, 2::3, 4].mean(1)[order_baselines]
        prob_opt_compositional = np.stack(data_comp.optimal_actions)[:, 2::3].mean(2).mean(1)[order_baselines] #if len(trials)==5 else NotImplementedError
        probs = prob_opt_compositional_first if len(trials)==1 else prob_opt_compositional

        rl3 = pd.DataFrame.from_dict( {'probs': probs, 'dls': np.log10(dls)}) #np.log10(dls)
        results = smf.ols("probs ~ dls", data=rl3)
        results = results.fit()

        
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        sns.regplot(x='dls',y='probs', data=rl3, scatter_kws={"color": COLORS['rl3'], "s": 100}, line_kws={"color": COLORS['compositional']})
        ax.set_xlim(xmin=0.9, xmax=4.1)
        ax.text(1.1, .7 if rule=='changepoint' else .89, '$r={}, p<.001$'.format(np.sqrt(results.rsquared).round(2)),  c='k', fontsize=FONTSIZE-4)

        if rule == 'add':
            plt.legend(loc='upper right', fontsize=FONTSIZE-2, frameon=False)
        ax.set_xlabel('Description length (in log-scale)', fontsize=FONTSIZE)
        name_trials = '1' if len(trials)==1 else 'last'
        ax.set_ylabel(f'p(optimal choice)', fontsize=FONTSIZE) #$a_{name_trials}$
        plt.xticks(fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/FittedDescriptionLengths_Human_MeanProbs_{}_{}.svg'.format(use_trials, rule), bbox_inches = 'tight')
        plt.show()
        print(results.summary())


def fitted_dls_grouped_histogram_peoplechoices(trials=np.array([0]), FONTSIZE=25, FIGSIZE=(7, 7), include_optimal=False):
    
    use_trial = 'all' if trials[0]==0 and len(trials)==15 else '1' if trials[0]==0 else 'composed'
    full = True if use_trial=='all' else False
    

    for rule in ['add', 'changepoint']:
        data = load_behavior(rule)
        comp_data = data[(data['experiment']=='compositional')]
        best_actions =np.stack(comp_data.best_actions)[:, 2::3].reshape(-1)
        full_path = f'_full={full}_changepoint=True.pth' if rule == 'changepoint' else  f'_full={full}_changepoint=False.pth' 
        dls = torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/dls_rl3_firsttrial_{full_path}') if len(trials)==1 else torch.load(f'../../RL3NeurIPS/data/RL3Fits/model_fits/dls_rl3{full_path}')
        dls = (dls + 1)*10
        if rule == 'add':
            orig_actions, _ = torch.load('/u/ajagadish/compositional-reinforcement-learning/modelfits/rl3_to_participants/rl3_add_curriculum.pth')  
        else:
            orig_actions, _, linfirst = torch.load('/u/ajagadish/compositional-reinforcement-learning/modelfits/rl3_to_participants/rl3_changepoint_curriculum.pth')

        high_dl_participants = (np.log10(dls)>3.)
        low_dl_participants = (np.log10(dls)<2.)
        low_dl_participants_actions = orig_actions[low_dl_participants]
        high_dl_participants_actions = orig_actions[high_dl_participants]

        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        if include_optimal:
            ax.hist([high_dl_participants_actions[..., 10+trials].reshape(-1), low_dl_participants_actions[..., trials].reshape(-1), best_actions], 
                bins=np.arange(7), density=True, width=0.267, color=[COLORS['compositional'], COLORS['noncompositional'], COLORS['optimal']], align='left', label=['Composers', 'Non-Composers', 'Optimal']);
        else:
            ax.hist([high_dl_participants_actions[..., 10+trials].reshape(-1), low_dl_participants_actions[..., trials].reshape(-1)], 
                bins=np.arange(7), density=True, width=0.39, color=[COLORS['compositional'], COLORS['noncompositional']], align='left', label=['Composers', 'Non-Composers']);

        plt.ylim([0., .65])
        ax.set_xlabel('Arms', fontsize=FONTSIZE)
        ax.set_ylabel('p(arm)', fontsize=FONTSIZE) 
        arms = ['S', 'D', 'F', 'J', 'K', 'L']
        plt.xticks(np.arange(6), arms, fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        plt.legend(fontsize=FONTSIZE-2, frameon=False, loc="upper left", bbox_to_anchor=(0.25, 1.), ncol=1)#(0.5, 1.15)
        sns.despine()
        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/FittedDescriptionLengths_HumanChoices_{}_{}.svg'.format(use_trial, rule), bbox_inches = 'tight')
        plt.show()

def fitted_models_analysis_of_regrets(model_name='rl3', FONTSIZE=20, FIGSIZE=(16, 8)):

    for rule in ['add', 'changepoint']:
        #load traces
        with open(f'../modelfits/analysis_regrets/traces_{rule}_{model_name}.pkl', 'rb') as fp:
            trace_first, list_of_errors = pickle.load(fp)

        # list_of_labels = {'optimal': 'Optimal arm', 
        #                 'corner': 'Corner arms', 
        #                 'corner_optimal': 'Optimal corner arms', 
        #                 'non_corner_optimal': 'Optimal Non-corner  arms',  
        #                 'corner_non_optimal': 'Non-optimal corner arms', 
        #                 'non_corner_non_optimal': 'Neither optimal nor corner arm', 
        #                 'non_optimal': 'Non-optimal arms',
        #                 'phasic_non_optimal': 'Non-optimal phasic Arms',
        #                 'neither': 'Neither of the Above'}

        f, axes = plt.subplots(2, 2, figsize=FIGSIZE)#, dpi=100)
        for idx, (ax, error) in enumerate(zip(axes.reshape(-1), list_of_errors)):
            lin = trace_first[error].posterior.w_lin.values.reshape(-1) 
            per = trace_first[error].posterior.w_per.values.reshape(-1) 
            ax.vlines(x=0., ymin=0, ymax=60, color='k', linewidth=3)
            ax.hist(lin, bins=100, density=True, label=['Linear sub-task regrets'], c=COLORS['linear'])
            ax.hist(per, bins=100, density=True, label=['Periodic sub-task regrets'], c=COLORS['periodic'])
            ax.set_title(f'{list_of_labels[error]}', fontsize=FONTSIZE)#f'{error}#'f'{error}'
            ax.set_xlim([-0.2, .2])
            if idx==0:
                # labels on the first axes
                ax.legend(frameon=False, loc='upper right', fontsize=FONTSIZE-5)
                ax.set_ylabel('#samples', fontsize=FONTSIZE) 
            if idx<(len(list_of_errors)-2):
                ax.set_xticks([])
                ax.tick_params(axis='x', which='both',bottom=False)
            else:
                ax.set_xlabel('Posterior regression coefficients', fontsize=FONTSIZE) 
            ax.tick_params(axis='x', labelsize=FONTSIZE-4)
            ax.tick_params(axis='y', labelsize=FONTSIZE-4)
            sns.despine()
            

        f.tight_layout()
        f.savefig(f'/eris/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/Model_RegretsAnalysis_{rule}_{model_name}.svg', bbox_inches = 'tight')
        plt.show()

def analysis_of_regrets(FONTSIZE=20, FIGSIZE=(16, 8)):

    for rule in ['add', 'changepoint']:
        #load traces
        with open(f'../modelfits/analysis_regrets/traces_{rule}.pkl', 'rb') as fp:
            trace_first, list_of_errors = pickle.load(fp)



        f, axes = plt.subplots(2, 2, figsize=FIGSIZE)#, dpi=100)
        for idx, (ax, error) in enumerate(zip(axes.reshape(-1), list_of_errors)):
            lin = trace_first[error].posterior.w_lin.values.reshape(-1) 
            per = trace_first[error].posterior.w_per.values.reshape(-1) 
            ax.vlines(x=0., ymin=0, ymax=60, color='k', linewidth=3)
            ax.hist(lin, bins=100, density=True, label=['Linear sub-task regrets'], color=COLORS['linear'])
            ax.hist(per, bins=100, density=True, label=['Periodic sub-task regrets'], color=COLORS['periodic'])
            ax.set_title(f'{list_of_labels[error]}', fontsize=FONTSIZE)#f'{error}#'f'{error}'
            ax.set_xlim([-0.2, .2])
            if idx==0:
                # labels on the first axes
                ax.legend(frameon=False, loc='upper right', fontsize=FONTSIZE-5)
                ax.set_ylabel('#samples', fontsize=FONTSIZE) 
            if idx<(len(list_of_errors)-2):
                ax.set_xticks([])
                ax.tick_params(axis='x', which='both',bottom=False)
            else:
                ax.set_xlabel('Posterior regression coefficients', fontsize=FONTSIZE) 
            ax.tick_params(axis='x', labelsize=FONTSIZE-4)
            ax.tick_params(axis='y', labelsize=FONTSIZE-4)
            sns.despine()
            

        f.tight_layout()
        f.savefig('/u/ajagadish/compositional-reinforcement-learning/figs/PLOS/Human_RegretsAnalysis_{}.svg'.format(rule), bbox_inches = 'tight')
        plt.show()