'''
Create a box plot of the estimated values in 1d simulation from multiple methods
'''
import platform, sys, os, pickle, subprocess, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

#%% need to specify the settings of data to plot
setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_full_list = [['Homogeneous', 'Piecewise Constant'],
                ['Homogeneous', 'Smooth'],
                ['Piecewise Constant', 'Homogeneous'],
                ['Smooth', 'Homogeneous']]
setting_label_list = [['Hm', 'PC'],
                ['Hm', 'Sm'],
                ['PC', 'Hm'],
                ['Sm', 'Hm']]
gamma_list = [0.9, 0.95]#
gamma_label_list = [re.sub("\.", "", str(gamma)) for gamma in gamma_list]#
# number of simulations
nsim = 100
# sample sizes
Ns = [25, 100]

# the name of the file that saves the values of the corresponding policy
file_name_parts = [
    'overall',
    'oracle',
    'proposed',
    'random',
    'kernel_02',
    'kernel_04',
    'kernel_08',
    'kernel_16'
]
# a list of strings, each being the label of the type of policy
policy_labels = ['Transition', 'Reward', '$\\gamma$',
                 'Proposed', 'Oracle', 'Overall', 'Random',
                 'Kernel ($\\h$=0.2)', 'Kernel ($\\h$=0.4)', #'Kernel ($\\h$=0.3)',
                 'Kernel ($\\h$=0.8)', 'Kernel ($\\h$=0.16)']
# the type of test statistic to use for detecting change point. Takes values
# in 'int_emp' (integral), '' (unnormalized max), and 'normalized' (normalized max)
method = '_int_emp'


#%% functions for plotting
def replace_na(x):
    if x == -999:
        return np.nan
    else:
        return x

def load_replace_na(file_name):
    try:
        out = replace_na(pickle.load(open(file_name, "rb")))
        return out
    except:
        print("cannot open", file_name)
        return np.nan

for N in Ns:
    append_name = '_N' + str(N) + '_1d'
    # create a table to be saved to csv
    df_table = pd.DataFrame(columns=policy_labels)
    add_col = 'column_names = ['
    for file_name in file_name_parts:
        if file_name != 'proposed' and file_name != file_name_parts[len(file_name_parts) - 1]:
            add_col += "'proposed - " + file_name + "', "
        elif file_name != 'proposed' and file_name == file_name_parts[len(file_name_parts) - 1]:
            add_col += "'proposed - " + file_name + "'"
    add_col += ']'
    exec(add_col)

    # create a table used for plotting
    df = pd.DataFrame(columns=['id', 'Setting', 'gamma'] + column_names + ['changept_proposed'])

    row_idx_table = 0
    row_idx = 0
    for i in range(len(setting_list)):
        setting = setting_list[i]
        print("setting =", setting)
        for j in range(len(gamma_list)):
            gamma = gamma_list[j]
            print("gamma =", gamma)
            gamma_name = re.sub("\.", "", str(gamma))
            setting_label = ["Transition: " + setting_label_list[i][0] + "\nReward: " + setting_label_list[i][1],
                             gamma]

            data_path0 = 'data/sim_result_trans' + setting[0] + '_reward' + setting[1] + '_gamma' + \
                        re.sub("\.", "", str(gamma)) + append_name

            for file_name in file_name_parts:
                exec('value_' + file_name + '_list = []')

            for nrep in range(1, nsim + 1):
                data_path = data_path0 + '/sim_result' + method +\
                            '_gamma' + gamma_name + append_name + '_' + str(nrep)
                for file_name in file_name_parts:
                    if "kernel" in file_name:
                        exec("value_" + file_name + " = load_replace_na(data_path + '/value_kernel_gamma' + gamma_name + '_bandwidth" +\
                            re.split("kernel_", file_name)[1] + ".dat')")
                    else:
                        exec(
                            "value_" + file_name + " = load_replace_na(data_path + '/value_' + file_name + '_gamma' + gamma_name + '.dat')")
                    changept_seq = 50
                    # if file_name == 'proposed':
                    #     value_proposed = value_oracle
                    exec('value_' + file_name + '_list.append(value_' + file_name + ')')
                add_data = 'df.loc[row_idx] = [nrep] + setting_label + ['
                for file_name in file_name_parts:
                    if file_name != 'proposed':
                        add_data += 'value_proposed - value_' + file_name + ', '
                add_data += 'changept_seq]'
                # print(add_data)
                exec(add_data)

                row_idx += 1

            # compute mean and standard deviation from raw data
            means = []
            stds = []
            for file_name in file_name_parts:
                exec( "means.append(np.nanmean(value_" + file_name + "_list))" )
                exec( "stds.append(np.nanstd(value_" + file_name + "_list))" )
            df_table.loc[row_idx_table] = setting_label_full_list[i] + [gamma] + [i + " (" + j + ")" for i, j in zip([f"{mean:0.3f}" for mean in means], [f"{std:0.2f}" for std in stds])]
            row_idx_table += 1

    stdoutOrigin = sys.stdout
    sys.stdout = open("output/optimal_rewards_table_latex" + str(N) + ".txt", "w")
    print(df_table.to_latex(index=True, sparsify=False, float_format="{:0.2f}".format,
                               multirow=True, escape=False))
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    # %% box plots of differences between estimated optimal rewards and the oracle optimal rewards
    df_plot = df.reset_index()
    df_plot = pd.melt(df_plot, id_vars=['Setting', 'gamma'],
                      value_vars=column_names)
    df_plot['variable'] = df_plot['variable'].astype('category')
    df_plot['variable'] = df_plot['variable'].cat.reorder_categories(
        column_names)
    print(df_plot)
    df_plot.to_csv('output/data_optvalue_dt_N' + str(N) + '.csv', index=False)

    # summary statistics
    sum_stat = df_plot.groupby(['Setting', 'gamma', 'variable'])['value'].\
        describe()[['mean', 'std', '50%']].unstack().reset_index()
    print(sum_stat.transpose())
    subprocess.call(["Rscript", "--vanilla", "p03_plot_changept_value.R", str(N)])