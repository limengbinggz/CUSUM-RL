'''
Detect change points after running the CUMSUM-RL test on a set of kappas using the
aggregated p-values from multiple random seed.
We use isotonic regression in the paper, but we also supply codes for applying sequential
detection.
The detected change points will be saved as .dat file in the data folder.
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

# a list of sample sizes
Ns = [25,100]
# a np.array or list of kappas
kappa_list = np.arange(25, 76, step=5, dtype=np.int32)
# a list of discount factors
gamma_list = [0.9, 0.95]
# number of simulations
nsim = 100
# length of time line
T = 100
setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_list = ['Transition: Homo\nReward: PC',
                'Transition: Homo\nReward: Sm',
                'Transition: PC\nReward: Homo',
                'Transition: Sm\nReward: Homo']
gamma_quantile = 0.1

#%% detect change point
def find_firsthit(array, value):
    '''
    Find the index of element that is the first to be smaller than value in array
    If all elements in array are greater than value, then return -1
    '''
    array0 = np.asarray(array) - value
    idx = np.where(array0 <= 0)[0]
    if len(idx) > 0:
        idx = min(idx)
    else:
        idx = -1
    return idx

def changept_detection(kappa_list, gamma, nsim = 100, T = 100, N = 300):
    gamma_name = re.sub("\.", "", str(gamma))
    append_name = '_N' + str(N) + '_1d'
    method_list = ['', '_normalized', '_int_emp']
    for s in range(len(setting_list)):
        setting = setting_list[s]
        setting_label = setting_label_list[s]

        # create directory
        path_name_save0 = 'data/sim_result_trans' + setting[0] + '_reward' + setting[1] +\
                          '_gamma' + gamma_name + append_name
        if not os.path.exists(path_name_save0):
            os.makedirs(path_name_save0, exist_ok=True)

        path_name0 = 'output/combine_p_values_N' + str(N) + \
                     '/trans' + setting[0] + \
                     '_reward' + setting[1] + '_gamma' + gamma_name +'_gquantile' +\
                     re.sub("\.", "", str(gamma_quantile))
        for method in method_list:
            dt = pd.read_csv(path_name0+"/p_values" + method + "_table.csv")
            row_idx = 0
            print("method = ", method)
            for nrep in np.arange(0, nsim):  # for each replication
                path_name_save = path_name_save0 + '/sim_result' + method + '_gamma' + \
                                 gamma_name + append_name + '_' + str(nrep+1)
                if not os.path.exists(path_name_save):
                    os.makedirs(path_name_save, exist_ok=True)

                p_values = np.array(dt.iloc[row_idx])
                row_idx += 1
                # print("p_values =", p_values)
                rejections_seq = p_values < 0.05
                # %% sequential method: simply take the first kappa at which H0 is rejected
                if not any(rejections_seq):
                    # print("no rejection")
                    changept = 0
                else:
                    changept_idx = np.argmax(rejections_seq)
                    if changept_idx < 1:
                        changept = int(T - kappa_list[0])
                    else:
                        changept = T - kappa_list[changept_idx-1]
                pickle.dump(changept, open(path_name_save + "/changept_sequential.dat", "wb"))

                #%% isotonic regression on p values
                iso_reg = IsotonicRegression(y_min = 0, y_max = 1, increasing=False, out_of_bounds="clip").fit(X=kappa_list, y=p_values)
                predicted = iso_reg.predict(kappa_list)
                idx = find_firsthit(predicted, 0.05)
                if idx > -1:
                    changept = T - kappa_list[max(idx-1,0)]
                else:
                    changept = 0
                pickle.dump(changept, open(path_name_save + "/changept_isoreg.dat", "wb"))



#%% run function
for N in Ns:
    print('N =', N)
    for gamma in gamma_list:
        print('gamma =', gamma)
        changept_detection(kappa_list, gamma, nsim, T, N)