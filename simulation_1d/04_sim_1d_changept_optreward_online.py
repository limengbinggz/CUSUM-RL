'''
Simulate non-stationary time series data and apply Q-learning.
estimate the optimal reward after identifying change point
'''
#!/usr/bin/python
import platform, sys, os, re, pickle
from copy import deepcopy
from copy import copy
plat = platform.platform()
print(plat)
if plat == 'macOS-12.5-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core' or plat == 'Linux-3.10.0-1160.53.1.el7.x86_64-x86_64-with-centos-7.6.1810-Core':  # biostat cluster
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-4.18.0-305.45.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
# import test_stat.simulate_data_1d as sim
from functions.simulate_data_1d_flexible import *
from sklearn.tree import DecisionTreeRegressor
import functions.compute_test_statistics as stat_pvalue
from functions.evaluation_separateA import *
# parallel jobs
from joblib import Parallel, delayed
# Arguments passed

# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
trans_setting = sys.argv[2]
reward_setting = sys.argv[3]
# num_threads = int(sys.argv[4])
gamma = float(sys.argv[4])
type_est = str(sys.argv[5])
effect_size = str(sys.argv[6])
N = int(sys.argv[7])
#'overall', 'oracle', 'oracle80', 'oracle60', 'proposed', 'random'
# est_kernel = str(sys.argv[6]) == 'True'

# seed = 10
# trans_setting = 'homo'
# reward_setting = 'pwconst2'
# gamma = 0.9
# type_est = "proposed"
# effect_size = "strong"

# import importlib
# importlib.reload(sim)
# importlib.reload(stat_pvalue)
# importlib.reload(plotting)
# importlib.reload(test_stat.evaluation)

startTime = datetime.now()

np.random.seed(seed)
plot_value = seed < 5

# %% simulate data
# N = 200
# terminal timestamp
T_initial = 100
# delta=0.5
# gamma=0.95
# dimension of X0
p = 1
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5

# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0.25

# #%% environment for reading in detected change points
# date = '01202022'
# append_name = '_N' + str(N) + '_rbf_1d'
# method = '_int_emp'
#
# if not os.path.exists('data'):
#     os.makedirs('data', exist_ok=True)
# data_path = 'data/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\.", "", str(gamma)) + \
#                                              append_name + '_' + date
# if not os.path.exists(data_path):
#     os.makedirs(data_path, exist_ok=True)
# data_path += '/sim_result' + method + '_gamma' + re.sub("\.", "", str(gamma)) + \
#              append_name + '_' + str(seed)
# if not os.path.exists(data_path):
#     os.makedirs(data_path, exist_ok=True)
# data_path_readin = data_path


#%% environment for saving online values
date = '20221005'
method = ''.join([i for i in type_est if not i.isdigit()])
append_name = '_' + effect_size
# method = '_int_emp'
if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
data_path = 'data/' + date + '_N' + str(N) + '_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + \
            re.sub("\\.", "", str(gamma)) + append_name
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += '/gamma' + re.sub("\\.", "", str(gamma)) + \
             append_name + '_' + str(seed)
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)

# print(data_path)
# os.chdir(data_path)
# print("os.curdir =", os.curdir)
stdoutOrigin = sys.stdout
method_label = type_est + "_gamma" + re.sub("\\.", "", str(gamma))
sys.stdout = open(data_path + "/log_online_" + method_label + ".txt", "w")

qmodel = 'polynomial'
degree = 1

num_threads = 1
#%% generate data for estimating the optimal policy
if effect_size == "strong":
    effect_size_factor = 1.0
elif effect_size == "moderate":
    effect_size_factor = 0.5
elif effect_size == "weak":
    effect_size_factor = 0.2

def transition_function1(St, At, t, mean=0, cov=0.25):
    '''
    Generate time-homogeneous transition at time t
    :param St: State at time t
    :param At: action in {0,1} at time t
    :param t: time t
    :return: a scalar of state
    '''
    # noise = np.random.normal(mean, cov, 1)[0] #len(At)
    # new_St = 0.5 * (2.0 * At - 1.0) * St + noise
    # print("noise =", noise)
    # print("new_St =", new_St)
    return -0.5 * effect_size_factor * (2.0 * At - 1.0) * St

def transition_function2(St, At, t, mean=0, cov=0.25):
    '''
    Generate time-homogeneous transition at time t
    # :param St: State at time t
    # :param At: action in {0,1} at time t
    # :param t: time t
    :return: a scalar of state
    '''
    return 0.5 * effect_size_factor * St * (2.0 * At - 1.0)

def reward_function1(St, At, t):
    '''
    Generate piecewise constant reward function in time consisting of 2 pieces
    # :param St: State at time t
    # :param At: action in {0,1} at time t
    # :param t: time t
    :return: a scalar of reward at time t
    '''
    return -1.5 * effect_size_factor * (2.0 * At - 1.0) * St
def reward_function2(St, At, t):
    '''
    Generate piecewise constant reward function in time consisting of 2 pieces
    # :param St: State at time t
    # :param At: action in {0,1} at time t
    # :param t: time t
    :return: a scalar of reward at time t
    '''
    return 1.5 * (2.0 * At - 1.0) * St * effect_size_factor
def reward_function_homo(St, At, t):
    '''
    Generate time-homogeneous reward function in time consisting of 2 pieces
    t = the time at which we generate a reward
    :return: a scalar of reward at time t
    '''
    return 0.25*St**2 * (2.0 * At - 1.0) * effect_size_factor + 4*St

if trans_setting == 'homo' and reward_setting == 'pwconst2':
    system_settings = {'N': N, 'T': T_initial,
                       'changepoints': [50],
                       'state_functions': [transition_function2, transition_function2],
                       'reward_functions': [reward_function1, reward_function2],
                       'state_change_type': 'homogeneous',  # ,
                       'reward_change_type': 'piecewise_constant'}
elif trans_setting == 'homo' and reward_setting == 'smooth':
    system_settings = {'N': N, 'T': T_initial,
                       'changepoints': [50],
                       'state_functions': [transition_function2, transition_function2],
                       'reward_functions': [reward_function1, reward_function2],
                       'state_change_type': 'homogeneous',  # ,
                       'reward_change_type': 'smooth',
                       'delta': 0.1}
elif trans_setting == 'pwconst2' and reward_setting == 'homo':
    system_settings = {'N': N, 'T': T_initial,
                       'changepoints': [50],
                       'state_functions': [transition_function1, transition_function2],
                       'reward_functions': [reward_function_homo, reward_function_homo],
                       'state_change_type': 'piecewise_constant',#,
                       'reward_change_type': 'homogeneous'
                        }
elif trans_setting == 'smooth' and reward_setting == 'homo':
    system_settings = {'N': N, 'T': T_initial,
                       'changepoints': [50],
                       'state_functions': [transition_function1, transition_function2],
                       'reward_functions': [reward_function_homo, reward_function_homo],
                       'state_change_type': 'smooth',#,
                       'reward_change_type': 'homogeneous',
                       'delta': 0.1
                        }
States, Rewards, Actions = simulate(system_settings, seed = seed)
# print(States1[0:2,0:50,0])
# print(Rewards1[0:2,0:50])
def transform(x):
    return (x - np.mean(x)) / np.std(x)
# States_s = copy(States)
# for i in range(1):
#     States_s[:,:,i] = transform(States_s[:,:,i])


#%% use kernel method to estimate the optimal policy
def gaussian_rbf(x, bandwidth):
    return np.exp(- (x / bandwidth)**2 )

def FQI_kernel(States, Rewards, Actions, bandwidth, num_sampled_times, seed = 50, perform_cv = True):
    T = Rewards.shape[1]
    sampling_probs = [gaussian_rbf((T-t) / T, bandwidth) for t in range(T)]
    sampled_time_points = np.random.choice(range(T), num_sampled_times + 1, p=sampling_probs / sum(sampling_probs))

    if perform_cv:
        model = DecisionTreeRegressor(random_state=seed)
        out = select_model_cv(States, Rewards, Actions, param_grid, rbf_bw, qmodel='polynomial', gamma=gamma,
                              model=model, max_iter=200, tol=1e-4, nfold = 5, num_threads = 1,
                              sampled_time_points = sampled_time_points, kernel_regression=True, metric=metric)
        model = out['best_model']
    else:
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=seed)
    print(model)

    # obtain (St, At, Rt)
    # Q learning after change point
    q_all = stat.q_learning(States[:,sampled_time_points,:], Rewards[:,sampled_time_points[:-1]], Actions[:,sampled_time_points[:-1]],
                            qmodel, degree, gamma)

    # obtain (St+1)
    q_all.States1 = q_all.create_design_matrix(States = States[:,sampled_time_points+1,:], Actions= np.zeros((N, num_sampled_times), dtype='int32'), type='current')
    q_all_fit = q_all.fit(model, max_iter=300, tol = 1e-6)
    return q_all


#%% estimate the value of the estimated policy
rbf_bw = 0.1
metric = 'kerneldist'
param_grid = {"max_depth": [3, 5],#
              "min_samples_leaf": [50, 60, 70]}#
basemodel = DecisionTreeRegressor(random_state=seed)
# model=DecisionTreeRegressor(random_state=10, max_depth=3, min_samples_leaf = 60)
N_new = N
T_new = 300
# create a random list of change points
change_points_subsequent = [50, T_initial + np.random.poisson(50)]
change_point_next = change_points_subsequent[1]
while change_point_next < T_new - 30:
    change_point_next += np.random.poisson(50)
    change_points_subsequent.append(change_point_next)
change_points_subsequent = change_points_subsequent[:-1]
print("change_points =", change_points_subsequent)
change_points_subsequent.append(T_new)
# policy_method = ''.join([i for i in type_est if not i.isdigit()])

# model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=480)
def estimate_value(States, Rewards, Actions, type_est, param_grid, basemodel):
    method = ''.join([i for i in type_est if not i.isdigit()])

    cp_detect_interval = 25
    seed_new = seed
    np.random.seed(seed)
    system_settings_batch = deepcopy(system_settings)
    system_settings_batch['N'] = N_new
    system_settings_batch['T'] = cp_detect_interval
    print(change_points_subsequent)

    # States_updated = np.empty(shape = (N_new, 0, 1))
    # Rewards_updated = np.empty(shape = (N_new, 0))
    # Actions_updated = np.empty(shape = (N_new, 0), dtype = "int32")
    States_updated = copy(States)
    Rewards_updated = copy(Rewards)
    Actions_updated = copy(Actions)

    # number of data batches
    n_batch = int((T_new - T_initial) / cp_detect_interval)
    cp_index = 0
    cp_current = 0
    T_current = T_initial

    # # first detect change point
    # if method == "random":
    #     cp_current = 0

    for batch_index in range(n_batch):

        #%% first, detect change point
        # cp_current = change_points_subsequent[cp_index]
        if method == "overall":
            if T_current > 200 and T_current % 100 == 0:
                cp_current = max(0, T_current - 200)
                print("T_current - 200 =", T_current - 200)
                print("cp_current =", cp_current)
        if method == "random":
            # if np.random.rand() <= 0.5: # if selected to have a change point
            if batch_index == 0:
                cp_current = np.random.randint(0, T_initial - 1)
            else:
                random_cp = np.random.randint(cp_current, T_current - 1)
                cp_current = random_cp
                # cp_current = T_current - cp_detect_interval + random_cp
        if method == "oracle":
            if batch_index == 0:
                cp_current = min(T_current, change_points_subsequent[cp_index])
            else:
                cp_current = max(cp_current, change_points_subsequent[cp_index])
        if method == "kernel":
            cp_current = 0
        if method == "proposed":
            if batch_index == 0:
                cp_current = 0
                T_length = T_initial
                # kappa_list = np.arange(15, 50, step = 5, dtype='int32')
                kappa_list = np.arange(35, 66, step=5, dtype='int32')
            else:
                T_length = States_updated[:, cp_current:, :].shape[1] - 1
                kappa_list = np.arange(15, min(T_length - 10, 50), step=5, dtype='int32')
            States_s = copy(States_updated[:, cp_current:, :])
            for i in range(1):
                States_s[:, :, i] = transform(States_s[:, :, i])

            epsilon = 0.05
            while min(kappa_list) <= 2 * epsilon * T_length:
                epsilon = max(epsilon - 0.01, 0.02)
            # print(kappa_list)
            result = stat_pvalue.cusumRL_detect_changept(States_s,
                                                         Rewards_updated[:, cp_current:],
                                                         Actions_updated[:, cp_current:],
                                                         kappa_list, T_total=T_length,
                                                         detection_methods=['sequential', 'isotonic'],
                                                         qmodel='rbf', degree=4, rbf_dim=0, bandwidth=1.0,
                                                         gamma=gamma, u_list=None, num_changept=15, num_threads=1,
                                                         theta=0.5, J=10, epsilon=epsilon, nB=1000,
                                                         select_basis=False, select_basis_interval=10,
                                                         num_basis_list=[0, 1, 2, 3, 5],
                                                         criterion='ls', seed=0, RBFSampler_random_state=1,
                                                         cut_off={'unnormalized': None, 'normalized': None,
                                                                  'integral_ref': None,
                                                                  'integral_emp': None})
            change_point_detected = result.cp_result['sequential']
            cp_current += change_point_detected['integral_emp']
            print(change_point_detected)



        #%% estimate the optimal policy
        if method != "kernel":
            print("Standard FQI")
            if batch_index % 2 == 0:
                out = select_model_cv(States_updated[:, cp_current:, :], Rewards_updated[:, cp_current:], Actions_updated[:, cp_current:], param_grid, bandwidth=rbf_bw,
                                    qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                    nfold = 5, num_threads = 1, metric = metric)
                model = out['best_model']
            print(model)
            q_all = stat.q_learning(States_updated[:, cp_current:, :], Rewards_updated[:, cp_current:], Actions_updated[:, cp_current:], qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
            q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)
        else:
            print("Kernel FQI")
            bandwidth = float(''.join([i for i in type_est if i.isdigit()])) * 0.1
            if abs(bandwidth) < 1e-5: # if bandwidth == 0, use the last two observation
                cp_current = States_updated.shape[1] - 2
                while np.all(Actions_updated[:, cp_current:].flatten() == 1):
                    Actions_updated[:, cp_current:] = np.random.binomial(1, 0.95, N).reshape(N,1)
                # print(Actions_updated[:, cp_current:].flatten())
                # out = select_model_cv(States_updated[:, cp_current:, :], Rewards_updated[:, cp_current:], Actions_updated[:, cp_current:],
                #                       param_grid, bandwidth=rbf_bw,
                #                       qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                #                       nfold=5, num_threads=3, metric=metric)
                # model = out['best_model']
                # print(model)
                model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=seed)
                q_all = stat.q_learning(States_updated[:, cp_current:, :], Rewards_updated[:, cp_current:],
                                        Actions_updated[:, cp_current:], qmodel, degree, gamma, rbf_dim=degree,
                                        rbf_bw=rbf_bw)
                q_all_fit = q_all.fit(model, max_iter=200, tol=1e-6)
            else:
                num_threads = 1
                num_sampled_times = int(T_current * 1.5)
                if batch_index % 2 == 0:
                    perform_cv = True
                else:
                    perform_cv = False
                q_all = FQI_kernel(States_updated, Rewards_updated, Actions_updated, bandwidth, num_sampled_times, seed, perform_cv)


        #%% now we collect the next batch of data following the estimated policy
        T_current += cp_detect_interval
        print("batch_index =", batch_index, "cp_index =", cp_index, "cp_current =", cp_current, "T_to_simulate =", T_current)

        # if the next change point has not been encountered, just keep the current dynamics
        if T_current <= change_points_subsequent[cp_index+1]:
            # system_settings_batch['changepoints'] = [T_current]
            system_settings_batch['changepoints'] = [cp_detect_interval]
            print("same as before")
            if trans_setting == 'homo' and reward_setting == 'pwconst2':
                system_settings_batch['reward_functions'] = [system_settings_batch['reward_functions'][1]]*2
            elif trans_setting == 'homo' and reward_setting == 'smooth':
                system_settings_batch['reward_functions'] = [system_settings_batch['reward_functions'][1]] * 2
            elif trans_setting == 'pwconst2' and reward_setting == 'homo':
                system_settings_batch['state_functions'] = [system_settings_batch['state_functions'][1]] * 2
            elif trans_setting == 'smooth' and reward_setting == 'homo':
                system_settings_batch['state_functions'] = [system_settings_batch['state_functions'][1]] * 2

        # if the next change point is encountered, need to make it the change point
        else:
            # system_settings_batch['changepoints'] = [change_points_subsequent[cp_index+1]]
            system_settings_batch['changepoints'] = [change_points_subsequent[cp_index+1] - T_current + cp_detect_interval]
            if (trans_setting == 'homo' and reward_setting == 'pwconst2') or (trans_setting == 'homo' and reward_setting == 'smooth'):
                if cp_index % 2 == 0:
                    system_settings_batch['reward_functions'] = [system_settings['reward_functions'][1], system_settings['reward_functions'][0]]
                    print("+-")
                else:
                    system_settings_batch['reward_functions'] = [system_settings['reward_functions'][0], system_settings['reward_functions'][1]]
                    print("-+")
            elif (trans_setting == 'pwconst2' and reward_setting == 'homo') or (trans_setting == 'smooth' and reward_setting == 'homo'):
                if cp_index % 2 == 0:
                    system_settings_batch['state_functions'] = [system_settings['state_functions'][1], system_settings['state_functions'][0]]
                    print("+-")
                else:
                    system_settings_batch['state_functions'] = [system_settings['state_functions'][0], system_settings['state_functions'][1]]
                    print("-+")

            print("change point encountered")
            cp_index += 1
        print(system_settings_batch['changepoints'])

        # simulate new data following the estimated policy
        seed_new = int(np.sqrt(np.random.randint(1e6) + seed_new*np.random.randint(10)))
        # print(States_new[199,0:,0])
        # print(Rewards_new[199,0:])
        # print(Actions_new[199,0:])
        # print(States_updated[3,100:,0])
        # print(Rewards_updated[5,100:])
        # print(Actions_updated[5,100:])

        S0 = States_updated[:, -1, :]
        States_new, Rewards_new, Actions_new = simulate(system_settings_batch, seed=seed_new, S0 = S0, optimal_policy_model=q_all) #q_all
        # if batch_index == 0:
        #     States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        # else:
        States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        Rewards_updated = np.concatenate((Rewards_updated, Rewards_new), axis = 1)
        Actions_updated = np.concatenate((Actions_updated, Actions_new), axis = 1)

        sys.stdout.flush()

    #%% compute values
    values = {}
    # discounted reward
    estimated_value = 0.0
    for t in range(T_initial, Rewards_updated.shape[1]):
        estimated_value += Rewards_updated[:,t] * gamma**(t - T_initial)
    values['discounted_reward'] = np.mean(estimated_value)
    values['average_reward'] = np.mean(Rewards_updated[:, T_initial:])
    values['raw_reward'] = Rewards_updated
    return values



# %% run the evaluation
if type_est != 'behavior':
    method_list = ['oracle', 'proposed', 'overall', 'random', 'kernel0', 'kernel01', 'kernel02', 'kernel03', 'kernel04']
    value = estimate_value(States, Rewards, Actions, type_est=type_est, param_grid=param_grid, basemodel=basemodel)
    print(type_est, "discounted reward:", value['discounted_reward'], "\n")
    print(type_est, "average reward:", value['average_reward'], "\n")
    with open(data_path + "/value_online_" + type_est + "_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb") as f:
        pickle.dump(value, f)

# else:
#     system_settings['T'] = 200
#     States, Rewards, Actions = simulate(system_settings, seed=seed)
#     #%% compute values
#     values = {}
#     # discounted reward
#     estimated_value = 0.0
#     for t in range(T_initial, Rewards.shape[1]):
#         estimated_value += Rewards[:,t] * gamma**t
#     values['discounted_reward'] = np.mean(estimated_value)
#     values['discounted_reward'] = np.mean(Rewards[:, T_initial:])
#     values['raw_reward'] = Rewards_updated



sys.stdout.flush()
print('Finished. Time: ', datetime.now() - startTime)

sys.stdout.close()
sys.stdout = stdoutOrigin
