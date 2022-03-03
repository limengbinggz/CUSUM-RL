'''
Combine p values from multiple simulations using method in
Nicolai Meinshausen, Lukas Meier & Peter Bühlmann (2009) p-Values for
High-Dimensional Regression, Journal of the American Statistical Association, 104:488, 1671-1681,
DOI: 10.1198/jasa.2009.tm08647
First read in saved p-value data from multiple random seeds, and aggregate p-values
with specified quantiles
'''
import pickle, platform, sys, os, re
import numpy as np
import pandas as pd
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

#%%

# a list of all kappa's that have been tested
kappa_list = np.arange(25, 76, step=5)
# create a list of all settings.
seeds = [1,2,3,4]
gamma_list = [0.9,0.95]#,0.95
Ns = [25,100]
# number of replications
nsim = 100
# gamma_quantile_inv_list = [1 / 0.05, 1 / 0.1, 1 / 0.15, 1 / 0.2]
# quantile in aggretating multiple p values
gamma_quantile_inv_list = [1 / 0.1]
setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_list = ['Transition: Homo\nReward: PC',
                'Transition: Homo\nReward: Sm',
                'Transition: PC\nReward: Homo',
                'Transition: Sm\nReward: Homo']
# a list of random seeds used in RBFSampler
RBFSampler_random_states = np.arange(1,4+1)


#%% compute rejection rate and sd
date = 'ls_altseed'
np_random_seeds = np.arange(1, nsim+1)
method_list = ['_int_emp', '', '_normalized']
method_label_list = ['Integral', 'Max', 'Normalized']

def calculate_rejection(kappa_list, gamma, date, np_random_seeds, method_list = ['']):
    gamma_name = re.sub("\\.", "", str(gamma))
    row_idx = 0
    append_name = '_N' + str(N) + '_1d'
    for m in range(len(method_list)):
        method = method_list[m]
        print(method)
        method_label = method_label_list[m]
        for s in range(len(setting_list)):
            setting = setting_list[s]
            print(setting)
            setting_label = setting_label_list[s]
            adjusted_p_values = {}
            for k in range(len(kappa_list)):
                rejections = []
                kappa = kappa_list[k]
                adjusted_p_values_kappa = []
                n_invalid = 0
                print(kappa)
                for nrep in np_random_seeds:
                    adjusted_p_values_rep = []

                    for seed in RBFSampler_random_states:
                        path_name0 = 'data/sim_result_trans' + setting[0] + '_reward' + setting[1] +\
                                     '_gamma' + gamma_name + '_kappa' + str(kappa) \
                                     + '_N' + str(N) + '_1d_' + date + str(seed)
                        path_name = path_name0 + '/sim_result_trans' + setting[0] + '_reward' + setting[1] +\
                                     '_gamma' + gamma_name + '_kappa' + str(kappa) +\
                                    '_N' + str(N) + '_1d_' + str(nrep)
                        # read in data
                        file_name = path_name + '/reject' + method + '.dat'
                        try:
                            ST = pickle.load(open(path_name + '/ST' + method + '.dat', "rb"))
                            BT = pickle.load(open(path_name + '/BT' + method + '.dat', "rb"))
                            p_value = 1.0 - np.mean(ST > BT)
                            if (ST > 0) and (min(BT) > 0):
                                adjusted_p_values_rep.append(p_value * gamma_quantile_inv)
                            else:
                                n_invalid +=1
                        except OSError:
                            print('cannot open', file_name)

                    adjusted_p_values_rep = np.array(adjusted_p_values_rep)
                    try:
                        adjusted_p_values_rep = min(1.0, np.quantile(adjusted_p_values_rep, 1/gamma_quantile_inv))
                        adjusted_p_values_kappa.append(adjusted_p_values_rep)
                        row_idx += 1
                    except:
                        print("rep", nrep, "problematic")

                adjusted_p_values[kappa] = np.array(adjusted_p_values_kappa)
            df = pd.DataFrame.from_dict(adjusted_p_values)
            # save to file
            path_save = "output/combine_p_values_N" + str(N) + "/"
            if not os.path.exists(path_save):
                os.makedirs(path_save, exist_ok=True)
            path_save += 'trans' + setting[0] + '_reward' + setting[1] + '_gamma' + gamma_name + \
                         "_gquantile" + re.sub("\\.", "", str(round(1/gamma_quantile_inv, 2)))
            if not os.path.exists(path_save):
                os.makedirs(path_save, exist_ok=True)
            df.to_csv(path_save + "/p_values" + method + "_table.csv", index=False)

            adjusted_rejections = {}
            for kappa in adjusted_p_values.keys():
                adjusted_rejections[kappa] = np.array([np.mean(adjusted_p_values[kappa] < 0.05)])
            df = pd.DataFrame.from_dict(adjusted_rejections)
            df.to_csv(path_save + "/rejections" + method + "_table.csv", index=False)
    return 0


for N in Ns:
    for gamma in gamma_list:
        print("gamma =", gamma)
        for gamma_quantile_inv in gamma_quantile_inv_list:
             calculate_rejection(kappa_list, gamma, date, np_random_seeds, method_list)