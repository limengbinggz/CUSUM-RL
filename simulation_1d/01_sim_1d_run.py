'''
Simulate stationary 1-dimensional time series data in 4 scenarios:
transition: homogeneous; reward:
transition: homogeneous; reward: smooth
transition: piece-wise constant ; reward: homogeneous
transition: smooth; reward: homogeneous
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")
import functions.compute_test_statistics as stat
import functions.simulate_data_1d as sim

'''
Arguments passed:
- seed: int. random seed to generate data
- kappa: int. test whether data on [T-kappa, T] is stationary
- num_threads: int. number of threads to compute the test statistic over many u's in parallel
- gamma: float. discount factor for the cumulative discounted reward. between 0 and 1
- trans_setting: string. scenario of the transition function. 
    Takes value from 'homo' (homogeneous), 'pwconst2' (piece-wise constant), or 'smooth' (smooth)
- reward_setting: string. scenario of the reward function. 
    Takes value from 'homo' (homogeneous), 'pwconst2' (piece-wise constant), or 'smooth' (smooth)
- N: int. number of individuals
- RBFSampler_random_state: int. random seed to supply to RBFSampler function

Example:
seed = 29
kappa = 55
num_threads = 5
gamma = 0.9
trans_setting = 'smooth'
reward_setting = 'homo'
N = int(25)
RBFSampler_random_state = 1
'''
# Arguments passed
seed = int(sys.argv[1])
kappa = int(sys.argv[2])
num_threads = int(sys.argv[3])
gamma = float(sys.argv[4])
trans_setting = sys.argv[5]
reward_setting = sys.argv[6]
N = int(sys.argv[7])
RBFSampler_random_state = int(sys.argv[8])

np.random.seed(seed)

# select_basis: whether to execute cross validation during the test
select_basis = True
# if select_basis == True, we need to supply the number of basis to be selected from.
# num_basis_list: a list containing the number of basis functions to select from in cross validation
# 0 means that only the linear term in included
num_basis_list = [0,1,2,3,5]
# criterion: criterion in cross validation. Takes a string from 'ls' (least squares) or
# 'kerneldist' (kernel distance, with bandwidth selected by the median heuristic)
criterion = 'ls'
# select_basis_interval: int. Run cross validation to select basis every select_basis_interval number of u's
select_basis_interval = 5
# if select_basis == False, we need to specify an integer as the number of basis functions to use
# degree: integer
degree=2 # whatever number here because select_basis=True

# qmodel: basis function. Can take a string from 'rbf' (RBFSampler) or 'polynomial' (polynomial basis)
qmodel = 'rbf'
# the number of grid points to search over the state space for max_u S_u
J = 1000
# boundry removal parameter
epsilon = 0.1
# number of bootstrap samples B_u
nB = 2000
# u_list: a list of integers u between [epsilon * T, T - epsilon * T], to search for max_u S_u
# example: [5,8,10,13,25]
# u_list can also be None, so that a list of regularly spaced time points will be used,
# but we need to specify the number of u's to be used as num_changept (int)
u_list = None
num_changept = 25


# %% simulate data
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

# width of smooth transition function
w = 0.01
delta = 1/10
sim_dat = sim.simulate_data(N, T, delta)
if trans_setting == 'homo' and reward_setting == 'pwconst2':
    def mytransition_function(t):
        return sim_dat.transition_homo(mean, cov)
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, sim_dat.reward_pwconstant2, seed)
elif trans_setting == 'homo' and reward_setting == 'smooth':
    def mytransition_function(t):
        return sim_dat.transition_homo(mean, cov)
    def myreward_function(t):
        return sim_dat.reward_smooth2(t, w)
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed)
elif trans_setting == 'pwconst2' and reward_setting == 'homo':
    def mytransition_function(t):
        return sim_dat.transition_pwconstant2(t, mean, cov)
    def myreward_function(t):
        return sim_dat.reward_homo()
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed)
elif trans_setting == 'smooth' and reward_setting == 'homo':
    def mytransition_function(t):
        return sim_dat.transition_smooth2(t, mean, cov, w)
    def myreward_function(t):
        return sim_dat.reward_homo()
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed)

# normalize state variables
def transform(x):
    return (x - np.mean(x)) / np.std(x)
for i in range(1):
    States[:,:,i] = transform(States[:,:,i])


# %% environment setup
# create folder under seed if not existing
mydate = '_' + criterion + '_altseed' + str(RBFSampler_random_state)
if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
path_name = 'data/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
            '_kappa' + str(kappa) + '_N' + str(N) + '_1d' + mydate
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
             '_kappa' + str(kappa) + '_N' + str(N) + '_1d_' + str(seed)
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
os.chdir(path_name)

# direct the screen output to a file
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("\nName of Python script:", sys.argv[0])
sys.stdout.flush()

# %% run Q learning with rbf basis function approximation
time_start = T - kappa
time_terminal = T
print('Running Q learning algorithm on [', time_start, ',', time_terminal, ']\n')
sys.stdout.flush()

# calculate bandwidth for RBFSampler using the median heuristic
# stack the states by person, and time: S11, ..., S1T, S21, ..., S2T
States_stack = States[:,time_start:(time_terminal+1),:].transpose(2, 0, 1).reshape(p, -1).T
from sklearn.metrics import pairwise_distances
# compute pairwise distance between states
pw_dist = pairwise_distances(States_stack, metric='euclidean')
# use the median of the minimum of distances as bandwidth
rbf_bw = 1.0 / np.nanmedian(np.where(pw_dist > 0, pw_dist, np.nan))
# rbf_bw = 0.1
print("Bandwidth chosen: {:.5f}".format(rbf_bw))
del pw_dist

# Create multiple threads as follows
# num_threads = 3
np.random.seed(seed)
T_total = T
theta = 0.5
# rbf_bw = 0.5
startTime = datetime.now()
out = stat.pvalue(States[:,time_start:(time_terminal+1),:],
                  Rewards[:,time_start:time_terminal],
                  Actions[:,time_start:time_terminal],
                  T_total, qmodel, degree, rbf_bw,
                  gamma, u_list, num_changept, num_threads, theta, J, epsilon, nB,
                  select_basis = select_basis, select_basis_interval = select_basis_interval, num_basis_list=num_basis_list,
                  criterion=criterion, RBFSampler_random_state=RBFSampler_random_state)
print('Finished. Time: ', datetime.now() - startTime)
sys.stdout.flush()
ST = out.ST
BT = out.BT
ST_normalized = out.ST_normalized
BT_normalized = out.BT_normalized
ST_int = out.ST_int
BT_int = out.BT_int
ST_int_emp = out.ST_int_emp
BT_int_emp = out.BT_int_emp



#%% print out test statistics of unnormalized version
# lower 0.05 quantile
critical_value = np.quantile(BT, 0.95)
# do we reject H0?
reject = ST > critical_value
print('ST =', ST, '\n')
print('critical_value =', critical_value, '\n')
print('Unnormalized: We reject H0:', reject, '\n')
# save objects to files
with open("reject.dat", "wb") as f:
    pickle.dump(reject, f)
with open("ST.dat", "wb") as f:
    pickle.dump(ST, f)
with open("BT.dat", "wb") as f:
    pickle.dump(BT, f)
sys.stdout.flush()

# %% print out test statistics of normalized version
# lower 0.05 quantile
critical_value_normalized = np.quantile(BT_normalized, 0.95)
# do we reject H0?
reject = ST_normalized > critical_value_normalized
print('ST_normalized =', ST_normalized, '\n')
print('critical_value_normalized =', critical_value_normalized, '\n')
print('Normalized: We reject H0:', reject, '\n')
# save objects to files
with open("reject_normalized.dat", "wb") as f:
    pickle.dump(reject, f)
with open("ST_normalized.dat", "wb") as f:
    pickle.dump(ST_normalized, f)
with open("BT_normalized.dat", "wb") as f:
    pickle.dump(BT_normalized, f)
sys.stdout.flush()

# #%% print out test statistics of integral version
# # lower 0.05 quantile
# critical_value_int = np.quantile(BT_int, 0.95)
# # do we reject H0?
# reject = ST_int > critical_value_int
# print('ST_int =', ST_int, '\n')
# print('critical_value_int =', critical_value_int, '\n')
# print('Integral type: We reject H0:', reject, '\n')
# # save objects to files
# with open("reject_int.dat", "wb") as f:
#     pickle.dump(reject, f)
# with open("ST_int.dat", "wb") as f:
#     pickle.dump(ST_int, f)
# with open("BT_int.dat", "wb") as f:
#     pickle.dump(BT_int, f)
# sys.stdout.flush()


#%% print out test statistics of integral version wrt empirical distribution
# lower 0.05 quantile
critical_value_int_emp = np.quantile(BT_int_emp, 0.95)
# do we reject H0?
reject = ST_int_emp > critical_value_int_emp
print('ST_int_emp =', ST_int_emp, '\n')
print('critical_value_int_emp =', critical_value_int_emp, '\n')
print('Integral type wrt empirical distribution: We reject H0:', reject, '\n')
# save objects to files
with open("reject_int_emp.dat", "wb") as f:
    pickle.dump(reject, f)
with open("ST_int_emp.dat", "wb") as f:
    pickle.dump(ST_int_emp, f)
with open("BT_int_emp.dat", "wb") as f:
    pickle.dump(BT_int_emp, f)
sys.stdout.flush()

if kappa == 25 and seed < 5:
    # histogram of simulated states
    print('Plot density of states.\n')
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(np.concatenate(States), bins='auto')
    plt.xlabel("State")
    plt.ylabel('Count')
    plt.title("Distribution of simulated states")
    # plt.show()
    fig.savefig('density_states.png')

sys.stdout.close()
sys.stdout=stdoutOrigin