import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
import pdb
import seaborn as sns
sns.set_style("whitegrid")
import sys

from matplotlib import pyplot as plt
from utils import natural_sort


"""Collects and reports timing performance of all models.

  Collects the perf.txt files in each results directory,
  and builds summarizing plots to show data and time efficiency.

"""


parser = argparse.ArgumentParser()
parser.add_argument('--measure', type=str, default='mmd2',
    choices=['mmd2', 'energy', 'kl'])
args = parser.parse_args()
measure = args.measure

print('*** {} *** tail performance\n'.format(measure))
measure_str_v = '{}_v'.format(measure)
measure_str_t = '{}_t'.format(measure)


def boxplots(df_, data_dim, hue_order=None, measure='mmd', gan_type=None):
    # CONVERGENCE PLOT. Plot loss on y-axis and step on x-axis. Subdivide by model.
    p = sns.boxplot(
        x='step', y=measure, hue='model_shorttag', data=df_,
        hue_order=hue_order,
        palette=sns.color_palette("hls", len(hue_order)),
        width=1, fliersize=2, linewidth=0.5)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    fig = p.get_figure()
    fig.set_size_inches(10,7)
    fig.suptitle('data_dim={}, Measure={}'.format(data_dim, measure))
    fig.savefig('results/plots/boxplot_{}_d{}_loss_{}.png'.format(
        gan_type, data_dim, measure))
    plt.close()

    # TIME PLOT. Plot time on y-axis and model on x-axis.
    p = sns.boxplot(
        x='model_shorttag', y='time', data=df_, 
        order=hue_order,
        palette=sns.color_palette("hls", len(hue_order)),
        width=1, fliersize=2, linewidth=0.5)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    fig = p.get_figure()
    fig.set_size_inches(10,7)
    fig.suptitle('gan type={}, data_dim={}'.format(gan_type, data_dim))
    fig.savefig('results/plots/boxplot_{}_d{}_time.png'.format(gan_type, data_dim))
    plt.close()


def run_boxplots(df, measure, data_dim, case='all'):
    # CE vs Upsample.
    df_ce = df.loc[df['model_shorttag'].isin(
        ['ce_iw', 'ce_miw', 'ce_sniw', 'ce_upsample'])]
    df_ce_iw_vs_up = df.loc[df['model_shorttag'].isin(
        ['ce_iw', 'ce_upsample'])]
    df_mmd = df.loc[df['model_shorttag'].isin(
        ['mmd_iw', 'mmd_miw', 'mmd_sniw', 'mmd_upsample'])]
    df_ce_baseline = df.loc[df['model_shorttag'].isin(['ce_baseline'])]

    assert df.count > 0, 'df is empty'
    assert df_ce.count > 0, 'df_ce is empty'
    assert df_ce_iw_vs_up.count > 0, 'df_ce_iw_vs_up is empty'
    assert df_mmd.count > 0, 'df_mmd is empty'
    assert df_baseline.count > 0, 'df_baseline is empty'

    if case == 'all':
        boxplots(df, data_dim, 
            hue_order=['ce_iw', 'ce_miw', 'ce_sniw', 'ce_conditional', 'ce_upsample',
                       'mmd_iw', 'mmd_miw', 'mmd_sniw', 'mmd_upsample'],
            measure=measure, gan_type='all')
    elif case == 'no_cgan':
        boxplots(df, data_dim, 
            hue_order=['ce_iw', 'ce_miw', 'ce_sniw', 'ce_upsample',
                       'mmd_iw', 'mmd_miw', 'mmd_sniw', 'mmd_upsample'],
            measure=measure, gan_type='all')
    elif case == 'ce':
        boxplots(df_ce, data_dim,
                 hue_order=['ce_iw', 'ce_miw', 'ce_sniw', 'ce_upsample'],
                 measure=measure, gan_type='ce')
    elif case == 'ce_iw_vs_up':
        boxplots(df_ce_iw_vs_up, data_dim,
                 hue_order=['ce_iw', 'ce_upsample'],
                 measure=measure, gan_type='ce')
    elif case == 'mmd':
        boxplots(df_mmd, data_dim,
                 hue_order=['mmd_iw', 'mmd_miw', 'mmd_sniw', 'mmd_upsample'],
                 measure=measure, gan_type='mmd')
    elif case == 'baseline':
        boxplots(df_baseline, data_dim,
                 hue_order=['baseline'],
                 measure=measure, gan_type='ce')


# START SCRIPT.
def main():
    results_dir = 'results'
    base_path = '/home/maurice/iwgn_multivariate/{}'.format(results_dir)
    plot_path = os.path.join(base_path, 'plots')
    if not os.path.exists(base_path):
        sys.exit('Need a results dir to read from.')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Run evaluation for each dimension separately.
    for data_dim in [2, 4, 10]:
        # Set up datafram to hold all run info, and get run names.
        df = pd.DataFrame() 
        model_run_names = [n for n in os.listdir(base_path) if 
            (('dim'+str(data_dim) in n) and ('run' in n))]
        if len(model_run_names) == 0:
            print('No model run names found for dim{}'.format(data_dim))
            continue

        # For each model in the desired set, add the performance data to an array.
        for model_run_name in model_run_names:
            run_path = os.path.join(base_path, model_run_name)
            filename = os.path.join(run_path, 'perf.txt')

            # Extract performance info from the log file.
            run_performance = np.loadtxt(filename,
                dtype={
                    'names': (
                        'model_type', 'model_subtype', 'dim', 'run',
                        'step', 'g_loss', 'mmd2_v', 'energy_v', 'kl_v',
                        'mmd2_t', 'energy_t', 'kl_t', 'time'),
                    'formats': (
                        '|S24', '|S24', np.int, np.int, np.int, np.float,
                        np.float, np.float, np.float, np.float, np.float,
                        np.float, np.float)},
                delimiter=',', skiprows=0)
            if len(run_performance.shape) == 0:
                run_performance = np.atleast_1d(run_performance)
            assert isinstance(run_performance, np.ndarray), ('file info must '
                'come as list, probably need more than one result per run.')

            # Add row to df.
            for row in run_performance:
                # Add row to dataframe.
                model = '{}_{}'.format(
                    row['model_type'], row['model_subtype'])
                row_df = pd.DataFrame(
                    [[model, row['model_type'], row['model_subtype'],
                      row['dim'], row['run'], row['step'], row['g_loss'],
                      row['mmd2_v'], row['energy_v'], row['kl_v'], row['mmd2_t'],
                      row['energy_t'], row['kl_t'], row['time']]], 
                    columns=['model', 'model_type', 'model_subtype', 'dim', 'run',
                        'step', 'g_loss', 'mmd2_v', 'energy_v', 'kl_v', 'mmd2_t',
                        'energy_t', 'kl_t', 'time'])
                df = df.append(row_df, ignore_index=True)


        ##############################################
        ##############################################
        # PLOT PERFORMANCE ASSOCIATED WITH BEST VALIDATION DISCREPANCY 

        # For each model, find run that performed best, based on validation.
        model_names = df.model.unique()
        performance_per_model = []

        for model_name in model_names:
            # Get all runs for that model, e.g. 10 runs.
            runs = natural_sort(
                [n for n in model_run_names if n.startswith(model_name)])

            # For each run, compute average g_loss and discrepancy measure in a
            # tail of available log steps. A run_outcomes array will store 
            # g_loss and measure per run.
            tail = 2
            run_outcomes = np.zeros((len(runs), 2))  
            for i, run_name in enumerate(runs):
                # Subset dataframe for this run.
                dim = int(run_name.split('_')[2].replace('dim', ''))
                run_num = int(run_name.split('_')[3].replace('run', ''))
                df_run_subset = df.loc[df['model'] == model_name]
                df_run_subset = df_run_subset.loc[df['dim'] == dim]
                df_run_subset = df_run_subset.loc[df['run'] == run_num]


                # Get tail performance for this run.
                validation_loss_ = np.mean(df_run_subset[measure_str_v][-tail:])
                reported_loss_ = np.mean(df_run_subset[measure_str_t][-tail:])

                # Store performance for this run.
                run_outcomes[i] = [validation_loss_, reported_loss_]

            # For this model, get best validation_loss_ among runs, and store
            # associated name and reported_loss. Also store average
            # reported_loss among runs.
            best_validation_run_idx = np.argmin(run_outcomes[:,0])
            best_validation_name = runs[best_validation_run_idx]
            best_validation_reported_loss = \
                run_outcomes[best_validation_run_idx, 1]
            avg_model_loss = np.mean(run_outcomes[:,1])

            # Store final info for this model.
            performance_per_model.append(
                [best_validation_name,
                 best_validation_reported_loss,
                 avg_model_loss])


        # Plot performance per model.
        fig = plt.figure()
        for model_perf in performance_per_model:
            n = best_validation_name = model_perf[0]
            best_validation_reported_loss = model_perf[1]
            avg_model_loss = model_perf[2]

            model_name = n.split('_dim')[0] 
            dim = int(n.split('_')[2].replace('dim', ''))
            run_num = int(n.split('_')[3].replace('run', ''))
            df_run_subset = df.loc[df['model'] == model_name]
            df_run_subset = df_run_subset.loc[df['dim'] == dim]
            df_run_subset = df_run_subset.loc[df['run'] == run_num]

            # Plot the test loss for the best validation model.
            plt.plot(df_run_subset.step, df_run_subset[measure_str_t], label=model_name)

        plt.xlabel('Step')
        plt.ylabel(measure)
        plt.legend()
        plt.savefig(os.path.join(
            base_path, 'plots', 'best_perf_dim{}.png'.format(data_dim)))
        plt.close()

        print(df.shape)
        for p in performance_per_model:
            print(('best_validation_reported_loss, avg_model_loss: '
                   '{:.5f}, {:.5f}, model: {}').format(
                       p[1], p[2], p[0]))


        ####################################
        # PLOT JUST CE, JUST MMD, and ALL.

        #for measure in ['mmd', 'energy', 'kl']:
            #run_boxplots(df, measure, data_dim, case='special')
            #run_boxplots(df, measure, data_dim, case='ce')
            #run_boxplots(df, measure, data_dim, case='ce_iw_vs_up')
            #run_boxplots(df, measure, data_dim, case='no_cgan')

        email = 0
        if email:
            # The CE results.
            os.system((
                'echo "{}" | mutt momod@utexas.edu -s '
                '"test_upsample" '
                '-a results/plots/boxplot_ce_d{}_loss_energy.png '
                '-a results/plots/boxplot_ce_d{}_loss_kl.png '
                '-a results/plots/boxplot_ce_d{}_loss_mmd.png '
                '-a results/plots/boxplot_ce_d{}_time.png').format(
                    os.getcwd(), data_dim, data_dim, data_dim, data_dim))


main()
