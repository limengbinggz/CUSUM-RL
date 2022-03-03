'''
Combine p values from multiple simulations using method in
Nicolai Meinshausen, Lukas Meier & Peter Bühlmann (2009) p-Values for
High-Dimensional Regression, Journal of the American Statistical Association, 104:488, 1671-1681,
DOI: 10.1198/jasa.2009.tm08647
First read in saved p-value data from multiple random seeds, and aggregate p-values
with specified quantiles
'''
import pickle,platform, sys, os, re
import numpy as np
import pandas as pd
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_ihs")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")


#%% specify simulation settings
# a list of all kappa's that have been tested
kappa_list = np.arange(10, 41, step=5)
# create a list of all settings.
seeds = [1,2,3,4]
gamma_list = [0.9, 0.95]
Ns = [100]
# number of replications
nsim = 100
# gamma_quantile_inv_list = [1 / 0.05, 1 / 0.1, 1 / 0.15, 1 / 0.2]
# quantile in aggretating multiple p values
gamma_quantile_inv_list = [1 / 0.1]
# a list of random seeds used in RBFSampler
RBFSampler_random_states = np.arange(1,4+1)



#%% compute rejection rate and sd
date = 'ls_altseed'
np_random_seeds = np.arange(1, nsim+1)
method_list = ['_int_emp', '', '_normalized']
method_label_list = ['Integral', 'Max', 'Normalized']

def calculate_rejection(kappa_list, gamma, date, np_random_seeds, method = ''):
    # df_rej = pd.DataFrame(columns=['kappa', 'rej_rate', 'rej_rate_std', 'n_invalid'])

    row_idx = 0
    append_name = '_N' + str(N) + '_ihs'
    adjusted_p_values = {}
    for k in range(len(kappa_list)):
        rejections = []
        kappa = kappa_list[k]

        n_invalid = 0
        adjusted_p_values_kappa = []
        for nrep in np_random_seeds:
            adjusted_p_values_rep = []
            for seed in RBFSampler_random_states:
                path_name0 = 'data/sim_result_gamma' + re.sub("\.", "", str(gamma)) + '_kappa' + str(kappa) \
                             + '_N' + str(N) + '_ihs_' + date + str(seed)
                path_name = path_name0 + '/sim_result_gamma' + re.sub("\.", "", str(gamma)) + '_kappa' + str(kappa) +\
                            '_N' + str(N) + '_ihs_' + str(nrep)
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
            adjusted_p_values_rep = min(1.0, np.quantile(adjusted_p_values_rep, 1/gamma_quantile_inv))
            adjusted_p_values_kappa.append(adjusted_p_values_rep)
        adjusted_p_values[kappa] = np.array(adjusted_p_values_kappa)

    df = pd.DataFrame.from_dict(adjusted_p_values)
    # print(df)
    # save to file
    path_save = "output/combine_p_values/"
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)
    path_save += "gamma" + re.sub("\.", "", str(gamma)) + \
                 "_gquantile" + re.sub("\.", "", str(round(1/gamma_quantile_inv, 2)))
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
            print("gamma_quantile_inv =", gamma_quantile_inv)
            calculate_rejection(kappa_list, gamma, date, np_random_seeds, method = '')
            print("method = max")
            calculate_rejection(kappa_list, gamma, date, np_random_seeds, method='_normalized')
            print("method = normalized")
            calculate_rejection(kappa_list, gamma, date, np_random_seeds, method='_int_emp')
            print("method = int")
