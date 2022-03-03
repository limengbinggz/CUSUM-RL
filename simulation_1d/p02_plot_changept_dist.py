'''
Create a bar plot of the distribution of detected change points.
'''
import platform, sys, os, pickle, subprocess, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_list = ['Transition: Homo\nReward: PC',
                'Transition: Homo\nReward: Sm',
                'Transition: PC\nReward: Homo',
                'Transition: Sm\nReward: Homo']
gamma_list = [0.9, 0.95]
nsim = 100
Ns = [25,100]
method_list = ["_normalized", "", "_int_emp"] #
method_label_list = ["Normalised", "Unnormalised", "Integral"]
# method used in change point detection. 'isoreg' for isotonic regression as used in the paper
type = 'isoreg' # or 'sequential'


#%%
gamma_label_list = [re.sub("\\.", "", str(gamma)) for gamma in gamma_list]
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
kappa_list = np.arange(25, 76, step=5, dtype=np.int32)

for N in Ns:
    append_name = '_N' + str(N) + '_1d'
    df = pd.DataFrame(columns=['setting', 'rep', 'gamma', 'method', 'changept'])
    for s in range(len(setting_list)):
        setting = setting_list[s]
        setting_label = setting_label_list[s]

        for j in range(len(gamma_list)):
            gamma = gamma_list[j]
            gamma_name = re.sub("\\.", "", str(gamma))
            data_path0 = 'data/sim_result_trans' + setting[0] + '_reward' + setting[1] +\
                          '_gamma' + gamma_name + append_name

            for nrep in range(1, nsim + 1):
                df_rep = pd.DataFrame(columns=['rep', 'gamma', 'method', 'changept'])
                change_points1 = []
                for method in method_list:
                    data_path = data_path0 + '/sim_result' + method + '_gamma' + gamma_name + append_name + '_' + str(nrep)
                    try:
                        change_points1.append(int(pickle.load(open(data_path + '/changept_' + type + '.dat', "rb"))))
                    except OSError:
                        print('cannot open', data_path)
                df_rep['changept'] = change_points1
                df_rep['setting'] = setting_label
                df_rep['rep'] = nrep
                df_rep['gamma'] = gamma
                df_rep['method'] = method_label_list
                df = pd.concat([df, df_rep])
    print("df =\n", df)

    df.to_csv('output/changept_data_N' + str(N) + '.csv', index = False)

    # run R code to use ggplot
    # os.system("module load R")
    subprocess.call("Rscript --vanilla p02_plot_changept_dist.R " + str(N), shell=True)