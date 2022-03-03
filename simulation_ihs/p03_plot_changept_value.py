import platform, sys, os, pickle, subprocess, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

gamma_list = [0.9, 0.95]
gamma_label_list = [re.sub("\.", "", str(gamma)) for gamma in gamma_list]
nsim = 100
Ns = 100


#%% functions for summarization
def replace_na(x):
    if x == -999:
        return np.nan
    else:
        return x

def load_replace_na(file_name):
    try:
        out = pickle.load(open(file_name, "rb"))
        return out
    except:
        print("cannot open", file_name)
        return np.nan

file_name_parts = [
    ['overall', ''],
    ['sequential', ''],
    ['observed', ''],
    ['oracle', ''],
    ['random', '']
]
type_est_labels = ['Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random']
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
for N in Ns:
    append_name = '_N' + str(N) + '_ihs'
    method = '_int_emp'

    # create a table to be saved to csv
    df_table = pd.DataFrame(columns=['$\\gamma$'] + type_est_labels)
    diff_cols = ['proposed - oracle', 'proposed - overall', 'proposed - random']
    df = pd.DataFrame(columns=['id', 'gamma'] + diff_cols + ['changept_proposed'])
    row_idx_table = 0
    row_idx = 0
    for j in range(len(gamma_list)):
        gamma = gamma_list[j]
        setting_label = ["gamma = " + str(gamma)]

        data_path0 = 'data/sim_result_gamma' + re.sub("\.", "", str(gamma)) + append_name

        value_overall_list = []
        value_proposed_list = []
        value_behavior_list = []
        value_oracle_list = []
        value_random_list = []
        for nrep in range(1, nsim + 1):
            data_path = data_path0 + '/sim_result_gamma' + re.sub("\.", "", str(gamma)) + append_name + '_' + str(nrep)

            value_overall = load_replace_na(data_path + '/value_overall_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
            value_proposed = load_replace_na(data_path + '/value_sequential_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
            value_behavior = load_replace_na(data_path + '/value_observed_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
            value_oracle = load_replace_na(data_path + '/value_oracle_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
            value_random = load_replace_na(data_path + '/value_random_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
            changept_proposed = load_replace_na(data_path + '/changept_sequential.dat')

            value_overall_list.append(value_overall)
            value_proposed_list.append(value_proposed)
            value_behavior_list.append(value_behavior)
            value_oracle_list.append(value_oracle)
            value_random_list.append(value_random)

            df.loc[row_idx] = [nrep] + setting_label + \
                              [value_proposed - value_oracle, value_proposed - value_overall,
                               value_proposed - value_random,
                               changept_proposed]
            row_idx += 1

        # compute mean and standard deviation from raw data
        means = list(df[diff_cols].apply(np.nanmean, axis=0))
        stds = list(df[diff_cols].apply(np.nanstd, axis=0))
        stds /= np.sqrt(nsim - 1)
        df_table.loc[row_idx_table] = [gamma] + \
                                      [i + " (" + j + ")" for i, j in zip([f"{mean:0.2f}" for mean in means], [f"{std:0.2f}" for std in stds])]
        row_idx_table += 1

    stdoutOrigin = sys.stdout
    sys.stdout = open("output/optimal_rewards_table_latex_N" + str(N) + ".txt", "w")
    # print(df_table.index)
    df_table_t = df_table.transpose()
    df_table_t.columns = df_table_t.iloc[0]
    df_table_t.columns = [r'$\gamma=$' + str(x) for x in list(df_table_t.columns)]
    df_table_t = df_table_t[1:]
    df_table_t.insert(loc=0, column='Method', value=type_est_labels)
    print(df_table_t.to_latex(index=False, sparsify=False, float_format="{:0.2f}".format,
                               multirow=True, escape=False, column_format='ccc'))
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    # print(df_table_t)