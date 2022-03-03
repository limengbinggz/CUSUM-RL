'''
Estimate the optimal reward after identifying change point.
The policies evaluated include:
proposed: using data on [T - kappa^*, T], where kappa^* is the change point detected by isotonic regression
overall: using data on [0, T]
behavior: the behavioral policy with A_t \in {-1, 1} and P(A_t = 1) = P(A_t = -1) = 0.5
random: pick a random change point kappa^**, and evaluate using data on [T - kappa^**, T]. Repeat the process
    for 20 times and take the average value
kernel: kernel regression method to deal with nonstationarity as described in the paper. Multiple bandwidths
    are taken: 0.2, 0.4, 0.8, and 1.6.
'''
#!/usr/bin/python
import platform, sys, os, re, pickle
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")
import functions.simulate_data_1d as sim
from functions.evaluation_separateA import *

'''
Arguments passed:
- seed: int. random seed to generate data
- trans_setting: string. scenario of the transition function. Takes value from 'homo', 'pwconst2', or 'smooth'
- reward_setting: string. scenario of the reward function. Takes value from 'homo', 'pwconst2', or 'smooth'
- gamma: float. discount factor for the cumulative discounted reward. between 0 and 1
- N: int. number of individuals
- type_est: string. the type of policy to be estimated. Takes values from 'overall', 'oracle', 'proposed', 'random',
    'kernel_02', 'kernel_04', 'kernel_08', 'kernel_16' (bandwidth = trailing numbers * 0.1; for example, 'kernel_02'
    means kernel method with bandwidth 0.2)  

Example:
seed = 30
trans_setting = 'homo'
reward_setting = 'smooth'
gamma = 0.9
N = int(25)
type_est = "proposed"
'''
seed = int(sys.argv[1])
trans_setting = sys.argv[2]
reward_setting = sys.argv[3]
gamma = float(sys.argv[4])
N = int(sys.argv[5])
type_est = str(sys.argv[6])


startTime = datetime.now()
np.random.seed(seed)

# criterion of cross validation. Takes value from 'ls' (least squares) or 'kerneldist' (kernel distance)
metric = 'ls'
# grids of hyperparameters of decision trees to search over in cross validation
param_grid = {"max_depth": [3, 5, 6],
              "min_samples_leaf": [50, 60, 70]}
# the type of test statistic to use for detecting change point. Takes values
# in 'int_emp' (integral), '' (unnormalized max), and 'normalized' (normalized max)
method = '_int_emp'
# basis functions. In evaluation, we use decision trees with only linear terms of states
qmodel = 'polynomial'
# degree of the basis function. degree = 1 or 0 for Linear term only
degree = 1
# true change point
time_change_pt_true = int(50)
# number of new individuals to simulate to calculate the discounted reward in infinite horizon
N_new = 300
# number of new time points to simulate to calculate the discounted reward in infinite horizon
T_new = 200 + time_change_pt_true


plot_value = seed < 5

# %% parameters to simulate data
# terminal timestamp
T = 100
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

#%% environment setup
append_name = '_N' + str(N) + '_1d'
if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
data_path = 'data/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                                             append_name
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += '/sim_result' + method + '_gamma' + re.sub("\\.", "", str(gamma)) + \
             append_name + '_' + str(seed)
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)
stdoutOrigin = sys.stdout
sys.stdout = open("log_" + type_est + ".txt", "w")

num_threads = 1
time_terminal = T

#%% generate data for estimating the optimal policy
def simulate(N=25, optimal_policy_model = None, S0=None, A0=None, T0=0, T1=T_new):
    w = 0.01
    delta = 1 / 10
    sim_dat = sim.simulate_data(N, T, delta)

    if trans_setting == 'homo' and reward_setting == 'pwconst2':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, sim_dat.reward_pwconstant2, seed, S0,
                                                    optimal_policy_model=optimal_policy_model, A0=A0, T0=T0, T1=T1)
    elif trans_setting == 'homo' and reward_setting == 'smooth':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        def myreward_function(t):
            return sim_dat.reward_smooth2(t, w)
        States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed, S0,
                                                    optimal_policy_model=optimal_policy_model, A0=A0, T0=T0, T1=T1)
    elif trans_setting == 'pwconst2' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_pwconstant2(t, mean, cov)
        def myreward_function(t):
            return sim_dat.reward_homo()
        States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed, S0,
                                                    optimal_policy_model=optimal_policy_model, A0=A0, T0=T0, T1=T1)
    elif trans_setting == 'smooth' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_smooth2(t, mean, cov, w)
        def myreward_function(t):
            return sim_dat.reward_homo()
        States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed, S0,
                                                    optimal_policy_model=optimal_policy_model, A0=A0, T0=T0, T1=T1)

    return States, Rewards, Actions

States, Rewards, Actions = simulate(N=N, optimal_policy_model = None, T0=0, T1=T)
basemodel = DecisionTreeRegressor(random_state=seed)


#%% estimate the value of the estimated policy
rbf_bw = None
def estimate_value(States, Rewards, Actions, param_grid, basemodel):
    out = select_model_cv(States, Rewards, Actions, param_grid, bandwidth=rbf_bw,
                        qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                        nfold = 5, num_threads = num_threads, metric = metric)
    model = out['best_model']
    print(model)
    q_all = stat.q_learning(States, Rewards, Actions, qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
    q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)
    if plot_value:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for a in range(2):
            tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
            axs[a].set_title('Action ' + str(2*a-1), loc='left')
        fig.savefig('plot_policy' + type_est + '.pdf', bbox_inches='tight', pad_inches = 0.5)
        plt.close('all')
        # plt.show()

    _, Rewards_new, _ = simulate(N_new, optimal_policy_model=q_all, T0=time_change_pt_true)
    estimated_value = 0.0
    for t in range(T_new - time_change_pt_true):
        estimated_value += Rewards_new[:,t] * gamma**t
    return estimated_value



#%% estimate the value of the behavioral policy
if type_est == 'overall':
    estimated_value = 0.0
    for t in range(T):
        estimated_value += Rewards[:,t] * gamma**t
    pickle.dump(np.mean(estimated_value), open("value_observed_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    print("Observed reward:", np.mean(estimated_value), "\n")
    sys.stdout.flush()



#%% overall policy: assume stationarity throughout
if type_est == 'overall':
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_overall = estimate_value(States, Rewards, Actions, param_grid, basemodel=model)
    estimated_value_overall = np.mean(estimated_value_overall)
    print("Overall estimated reward:", estimated_value_overall, "\n")
    pickle.dump(estimated_value_overall, open("value_overall_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_overall, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of overall values')
        plt.savefig("hist_value_overall_gamma" + re.sub("\\.", "", str(gamma)) + ".png")



# %% estimate the oracle policy: piecewise Q function before and after change point
#%% fit the Q model with known change point
if type_est == 'oracle':
    time_change_pt = time_change_pt_true
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model)
    estimated_value_oracle = np.mean(estimated_value_oracle)
    print("Oracle estimated reward:", estimated_value_oracle, "\n")
    pickle.dump(estimated_value_oracle, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_oracle, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")


# %% estimate the proposed policy: piecewise Q function before and after the estimated change point
if type_est == 'proposed':
    method = "isoreg"
    time_change_pt_seq = int(pickle.load(open('changept_' + method + '.dat', "rb")))
    print("isotonic:", time_change_pt_seq)
    if time_change_pt_seq == time_change_pt_true:
        try:
            estimated_value_oracle = pickle.load(open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "rb"))
            estimated_value_seq = estimated_value_oracle
            print("Sequential estimated reward:", estimated_value_seq)
            pickle.dump(estimated_value_seq,
                        open("value_" + type_est + "_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
            sys.stdout.flush()
        except:
            print(data_path + ": oracle value not estimated yet")
    else:
        time_change_pt = time_change_pt_seq
        model = DecisionTreeRegressor(random_state=seed)
        estimated_value_seq = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:],
                                             Actions[:, time_change_pt:], param_grid, model)
        estimated_value_seq = np.mean(estimated_value_seq)
        print("Sequential estimated reward:", estimated_value_seq)
        pickle.dump(estimated_value_seq, open("value_" + type_est + "_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
        sys.stdout.flush()



# %% estimate the random policy
if type_est == 'random':
    num_random_cp = 20
    method = "random"
    estimated_value_random_list = []
    for num in range(num_random_cp):
        np.random.seed(seed + num)
        time_change_pt_random = np.random.randint(0, T-1)
        time_change_pt = time_change_pt_random
        print("number:", num, "random:", time_change_pt)
        model = DecisionTreeRegressor(random_state=seed)
        estimated_value_random = estimate_value(States[:, time_change_pt_random:, :], Rewards[:, time_change_pt_random:],
                                                Actions[:, time_change_pt_random:], param_grid, model)
        estimated_value_random = np.mean(estimated_value_random)
        estimated_value_random_list.append(estimated_value_random)
        print("Random estimated reward:", estimated_value_random, "\n")
        sys.stdout.flush()
    estimated_value_random_list = np.array(estimated_value_random_list)
    estimated_value_random = np.mean(estimated_value_random_list)
    pickle.dump(estimated_value_random, open("value_" + method + "_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))



# %% estimate the kernel policy
if ''.join([i for i in type_est if not i.isdigit()]) == 'kernel':
    bandwidth = float(''.join([i for i in type_est if i.isdigit()])) * 0.1
    method = 'kernel'
    num_threads = 1
    param_grid = {"max_depth": [3, 5, 6],#
                  "min_samples_leaf": [50, 60, 70, 80, 90]}#
    def gaussian_rbf(x, bandwidth):
        return np.exp(- (x / bandwidth)**2 )

    def FQI_kernel(bandwidth, seed = 50):
        sampling_probs = [gaussian_rbf((T-t) / T, bandwidth) for t in range(T)]
        sampled_time_points = np.random.choice(range(T), num_sampled_times + 1, p=sampling_probs / sum(sampling_probs))

        model = DecisionTreeRegressor(random_state=seed)
        out = select_model_cv(States, Rewards, Actions, param_grid, rbf_bw, qmodel='polynomial', gamma=gamma,
                              model=model, max_iter=200, tol=1e-4, nfold = 5, num_threads = num_threads,
                              sampled_time_points = sampled_time_points, kernel_regression=True, metric=metric)
        model = out['best_model']
        print(model)

        # obtain (St, At, Rt)
        # Q learning after change point
        q_all = stat.q_learning(States[:,sampled_time_points,:], Rewards[:,sampled_time_points[:-1]], Actions[:,sampled_time_points[:-1]],
                                qmodel, degree, gamma)

        # obtain (St+1)
        q_all.States1 = q_all.create_design_matrix(States = States[:,sampled_time_points+1,:], Actions= np.zeros((N, num_sampled_times), dtype='int32'), type='current')
        q_all_fit = q_all.fit(model, max_iter=300, tol = 1e-6)

        _, Rewards_new, _ = simulate(N_new, optimal_policy_model=q_all, T0=time_change_pt_true)
        estimated_value = 0.0
        for t in range(T_new-time_change_pt_true):
            estimated_value += Rewards_new[:,t] * gamma**t
        estimated_value = np.mean(estimated_value)
        print("bandwidth =", bandwidth)
        print("Kernel estimated reward:", estimated_value, "\n")
        pickle.dump(estimated_value, open("value_" + method + "_gamma" + re.sub("\\.", "", str(gamma)) + "_bandwidth" + re.sub("\\.", "", str(bandwidth)) + ".dat", "wb"))
        return 0

    # run kernel method
    num_sampled_times = 150
    if bandwidth > 0:
        FQI_kernel(bandwidth=round(bandwidth,1))
        sys.stdout.flush()


print('Finished. Time: ', datetime.now() - startTime)
sys.stdout.close()
sys.stdout = stdoutOrigin