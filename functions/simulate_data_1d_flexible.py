'''
Simulate stationary time series data and apply Q-learning.
Generate 1 dimensional states
'''


import numpy as np
import math
from copy import deepcopy
from copy import copy



def psi(t, w):
    if t <= 0:
        # print("NULL")
        return 0
    else:
        return math.exp(-w / t)

def smooth_transform(x, f1, f2, x0, x1, w=0.01):
    '''
    Define a smooth transformation function on interval [x0, x1], connecting smoothly function f1 on x < x0 and f2 on x > x1
    :param x: the transformed function value to be evaluated at
    :return: a scalar of transformed function value
    '''

    if x > x0:
        ratio = (x - x0) / (x1 - x0)
        return f1(x) + (f2(x) - f1(x)) / (psi(ratio, w) + psi(1 - ratio, w)) * psi(ratio, w)
    elif x <= x0:
        return f1(x)
    elif x >= x1:
        return f2(x)



# #%%
# def simulate(system_settings, seed=0, S0 = None, T0=0, T_total=26, burnin=0,
#              optimal_policy_model=None, epsilon_greedy = 0.05, mean0 = 0.0, cov0 = 0.5, normalized=[0.0, 1.0]):
#     '''
#     simulate states, rewards, and action data
#     :param mean0: mean vector for the initial state S_0
#     :param cov0: covariance matrix for the initial state S_0
#     :param seed: numpy random seed
#     :param S0: initial states of N subjects
#     :param A0: initial actions of N subjects
#     :param optimal_policy_model: input actions and simulate state and rewards following the input actions. Can be used for
#     Monte Carlo policy evaluation
#     :return: a list [ states, rewards, actions]
#     '''
#     # number of total time points
#     T = system_settings['T']
#     # number of individuals
#     N = system_settings['N']
#     change_points = copy.deepcopy(system_settings['changepoints'])
#
#     # set seed
#     np.random.seed(seed)
#     States = np.zeros([N, T + 1, 1])  # S[i][t]
#     Rewards = np.zeros([N, T])  # R[i,t]
#     Actions = np.zeros([N, T])  # Actions[i,t]
#
#     if system_settings['state_change_type'] == 'smooth':
#         delta = system_settings['delta']
#     change_points.append(T)
#
#     #%% random actions
#     if optimal_policy_model is None:
#         t = 0
#         for i in range(N):  # for each individual
#             St = np.random.normal(mean0, cov0, 1)
#             # print(self.St)
#             States[i, 0, :] = St
#             At = np.random.binomial(1, 0.5, 1)[0]
#             Actions[i, t] = At
#             Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
#             # generate S_i,t+1
#             St = system_settings['state_functions'][0](St, At, t)
#             States[i, t + 1, :] = St
#
#         # for the subsequent time points
#         for i in range(N):  # each individual i
#             St = States[i, 1, :]
#             for t in range(1, T):  # each time point t
#                 # generate policy
#                 At = np.random.binomial(1, 0.5, 1)
#                 Actions[i, t] = At
#                 # generate immediate response R_i,t
#                 Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
#                 # generate S_i,t+1
#                 St = system_settings['state_functions'][0](St, At, t)
#                 States[i, t + 1, :] = St
#
#     # convert Actions to integers
#     Actions = Actions.astype(int)
#     return States, Rewards, Actions


#%%
def simulate(system_settings, seed=0, S0 = None, T0=0, T_total=26, burnin=0,
             optimal_policy_model=None, epsilon_greedy = 0.05,
             mean0 = 0.0, cov0 = 0.5, mean = 0.0, cov = 0.25, normalized=[0.0, 1.0]):
    '''
    simulate states, rewards, and action data
    :param mean0: mean vector for the initial state S_0
    :param cov0: covariance matrix for the initial state S_0
    :param seed: numpy random seed
    :param S0: initial states of N subjects
    :param A0: initial actions of N subjects
    :param optimal_policy_model: input actions and simulate state and rewards following the input actions. Can be used for
    Monte Carlo policy evaluation
    :return: a list [ states, rewards, actions]
    '''
    # number of total time points
    T = system_settings['T']
    # number of individuals
    N = system_settings['N']
    change_points = deepcopy(system_settings['changepoints'])

    # set seed
    np.random.seed(seed)
    States = np.zeros([N, T + 1, 1])  # S[i][t]
    # print(States.shape)
    Rewards = np.zeros([N, T])  # R[i,t]
    Actions = np.zeros([N, T])  # Actions[i,t]

    if system_settings['state_change_type'] == 'smooth' or system_settings['reward_change_type'] == 'smooth':
        deltaT = system_settings['delta'] * T
    change_points.append(T)

    # print(np.random.normal(mean0, cov0, 1))

    #%% random actions
    if optimal_policy_model is None:

        # generate initial state S_0 and action A_0
        # States[:, 0, 0] = np.random.normal(mean0, cov0, N)
        # Actions[:, 0] = np.random.binomial(1, 0.5, N)

        t = 0
        for i in range(N):  # for each individual
            # generate initial state S_0 and action A_0
            # generate initial state S_0 and action A_0
            if S0 is None:
                St = np.random.normal(mean0, cov0, 1)
            else:
                St = S0[i, 0]

            # St = np.random.normal(mean0, cov0, 1)
            # print("St =", St)
            States[i, 0, :] = St
            At = np.random.binomial(1, 0.5, 1)[0]
            Actions[i, t] = At
            Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
            # generate S_i,t+1
            St = system_settings['state_functions'][0](St, At, t) + np.random.normal(mean, cov, 1)[0]
            # print("St =", St)
            States[i, t + 1, :] = St
            # print("States[i, t + 1, :] =", States[i, t + 1, :])

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            for segment in range(len(change_points)):  # for each change point
                for t in range(previous_change_point, change_points[segment]):

                    ## generate action
                    At = np.random.binomial(1, 0.5, 1)[0]
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    if system_settings['reward_change_type'] == 'piecewise_constant':
                        Rt = system_settings['reward_functions'][segment](St, At, t)
                        # print("t =", t_current, " Reward: pwconst, segment", segment)
                    elif system_settings['reward_change_type'] == 'homogeneous':
                        Rt = system_settings['reward_functions'][0](St, At, t)
                    elif system_settings['reward_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT - 1:
                                Rt = system_settings['reward_functions'][segment](St, At, t)
                                # print("t =", t_current, ": smooth; before cp")
                            else: # during the smooth change
                                def f1(tt):
                                    return system_settings['reward_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['reward_functions'][segment + 1](St, At, tt)
                                Rt = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment])
                                # print("t =", t_current, ": smooth; during change")
                        elif segment == len(change_points) - 1:  # at the last segment
                            Rt = system_settings['reward_functions'][segment](St, At, t)
                            # print("t =", t_current, ": smooth; last segment")
                    Rewards[i, t] = Rt

                    ## compute the next state
                    if system_settings['state_change_type'] == 'piecewise_constant':
                        St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                        # print("St =", St)
                    elif system_settings['state_change_type'] == 'homogeneous':
                        St = system_settings['state_functions'][0](St, At, t) + np.random.normal(mean, cov, 1)[0]
                        # print("t =", t_current+1, ": homo, segment", segment)
                    elif system_settings['state_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT:
                                # print("t =", t, "smooth")
                                St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                                # print("t =", t_current+1, ": smooth; before cp")
                            else: # during the smooth change

                                def f1(tt):
                                    return system_settings['state_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['state_functions'][segment + 1](St, At, tt)
                                # print("t =", t, ", f1 =", f1(t), ", f2 =", f2(t), ", St =", St, ", At =", At)
                                St = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment]) + np.random.normal(mean, cov, 1)[0]
                                # print("t =", t+1, ": smooth =", smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment]))
                        elif segment == len(change_points) - 1:  # at the last segment
                            St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                            # print("t =", t_current+1, ": smooth; last segment")
                    States[i, t + 1, 0] = St
                    # print("States[i, t + 1, 0] =", States[i, t + 1, 0])

                previous_change_point = change_points[segment]

    #%% with known policy
    else:
        myState = np.zeros([2, 2, 1])
        t = 0
        for i in range(N):  # for each individual
            # print("i =", i, ", t = 0")
            # generate initial state S_0 and action A_0
            if S0 is None:
                St = np.random.normal(mean0, cov0, 1)
            else:
                St = S0[i, 0]
            # print("St =", St)
            # print("St =", St)
            States[i, 0, :] = St
            # compute the current action
            myState[0, 0, 0] = St
            # generate policy
            if np.random.rand() < epsilon_greedy:
                At = np.random.binomial(1, 0.5, 1)[0]
                # print("epsilon At =", At)
            else:
                At = optimal_policy_model.predict(myState).opt_action[0]
                # print("optimal At =", At)
            Actions[i, t] = At
            Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
            # generate S_i,t+1
            St = system_settings['state_functions'][0](St, At, t) + np.random.normal(mean, cov, 1)[0]
            # print("St =", St)
            States[i, t + 1, :] = St
            # print("States[i, t + 1, :] =", States[i, t + 1, :])

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            for segment in range(len(change_points)):  # for each change point
                for t in range(previous_change_point, change_points[segment]):

                    ## generate action
                    # compute the current action
                    myState[0, 0, 0] = St
                    # generate policy
                    if np.random.rand() < epsilon_greedy:
                        At = np.random.binomial(1, 0.5, 1)[0]
                        # print("i =", i, ", t =", t, "At =", At)
                    else:
                        # print("myState =", myState[:,:,0])
                        # print("action =", optimal_policy_model.predict(myState).opt_action[0])
                        At = optimal_policy_model.predict(myState).opt_action[0]
                    # print("i =", i, ", t =", t, "At =", At)
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    if system_settings['reward_change_type'] == 'piecewise_constant':
                        Rt = system_settings['reward_functions'][segment](St, At, t)
                        # print("t =", t_current, " Reward: pwconst, segment", segment)
                    elif system_settings['reward_change_type'] == 'homogeneous':
                        Rt = system_settings['reward_functions'][0](St, At, t)
                    elif system_settings['reward_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT - 1:
                                Rt = system_settings['reward_functions'][segment](St, At, t)
                                # print("t =", t_current, ": smooth; before cp")
                            else: # during the smooth change
                                def f1(tt):
                                    return system_settings['reward_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['reward_functions'][segment + 1](St, At, tt)
                                Rt = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment])
                                # print("t =", t_current, ": smooth; during change")
                        elif segment == len(change_points) - 1:  # at the last segment
                            Rt = system_settings['reward_functions'][segment](St, At, t)
                            # print("t =", t_current, ": smooth; last segment")
                    Rewards[i, t] = Rt

                    ## compute the next state
                    if system_settings['state_change_type'] == 'piecewise_constant':
                        St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                        # print("St =", St)
                    elif system_settings['state_change_type'] == 'homogeneous':
                        St = system_settings['state_functions'][0](St, At, t) + np.random.normal(mean, cov, 1)[0]
                        # print("t =", t_current+1, ": homo, segment", segment)
                    elif system_settings['state_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT:
                                # print("t =", t, "smooth")
                                St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                                # print("t =", t_current+1, ": smooth; before cp")
                            else: # during the smooth change

                                def f1(tt):
                                    return system_settings['state_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['state_functions'][segment + 1](St, At, tt)
                                # print("t =", t, ", f1 =", f1(t), ", f2 =", f2(t), ", St =", St, ", At =", At)
                                St = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment]) + np.random.normal(mean, cov, 1)[0]
                                # print("t =", t+1, ": smooth =", smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment]))
                        elif segment == len(change_points) - 1:  # at the last segment
                            St = system_settings['state_functions'][segment](St, At, t) + np.random.normal(mean, cov, 1)[0]
                            # print("t =", t_current+1, ": smooth; last segment")
                    States[i, t + 1, 0] = St
                    # print("States[i, t + 1, 0] =", States[i, t + 1, 0])

                previous_change_point = change_points[segment]

    # convert Actions to integers
    Actions = Actions.astype(int)
    return States, Rewards, Actions



# #%%
# def simulate(system_settings, seed=0, S0 = None, T0=0, T_total=26, burnin=0,
#              optimal_policy_model=None, epsilon_greedy = 0.05, mean0 = 0.0, cov0 = 0.5, normalized=[0.0, 1.0]):
#     '''
#     simulate states, rewards, and action data
#     :param mean0: mean vector for the initial state S_0
#     :param cov0: covariance matrix for the initial state S_0
#     :param seed: numpy random seed
#     :param S0: initial states of N subjects
#     :param A0: initial actions of N subjects
#     :param optimal_policy_model: input actions and simulate state and rewards following the input actions. Can be used for
#     Monte Carlo policy evaluation
#     :return: a list [ states, rewards, actions]
#     '''
#     # number of total time points
#     T = system_settings['T']
#     # number of individuals
#     N = system_settings['N']
#     change_points = copy.deepcopy(system_settings['changepoints'])
#
#     # set seed
#     np.random.seed(seed)
#     States = np.zeros([N, T + 1, 1])  # S[i][t]
#     Rewards = np.zeros([N, T])  # R[i,t]
#     Actions = np.zeros([N, T])  # Actions[i,t]
#
#     if system_settings['state_change_type'] == 'smooth':
#         delta = system_settings['delta']
#     change_points.append(T)
#
#     #%% random actions
#     if optimal_policy_model is None:
#
#         # generate initial state S_0 and action A_0
#         States[:, 0, 0] = np.random.normal(mean0, cov0, N)
#         # Actions[:, 0] = np.random.binomial(1, 0.5, N)
#
#         # extract the current state St and action At
#         St = copy.deepcopy(States[:, 0, 0])
#         # At = copy.deepcopy(Actions[:, 0])
#
#         # actual time stamp
#         t_current = 0
#         previous_chaneg_point = 0
#         for segment in range(len(change_points)): # for each change point
#             for t in range(previous_chaneg_point, change_points[segment]):
#                 # compute the current action
#                 At = np.random.binomial(1, 0.5, N)
#                 Actions[:, t_current] = At
#                 ## compute the current reward
#                 if system_settings['reward_change_type'] == 'piecewise_constant':
#                     Rt = system_settings['reward_functions'][segment](St, At, t)
#                     # print("t =", t_current, " Reward: pwconst, segment", segment)
#                 elif system_settings['reward_change_type'] == 'homogeneous':
#                     Rt = system_settings['reward_functions'][0](St, At, t)
#                 elif system_settings['reward_change_type'] == 'smooth':
#                     if segment < len(change_points)-1: # before reaching the last segment
#                         # before the smooth change point: simply use the constant part
#                         if t_current <= change_points[segment] - delta - 1:
#                             Rt = system_settings['reward_functions'][segment](St, At, t)
#                             # print("t =", t_current, ": smooth; before cp")
#                         else: # during the smooth change
#                             def f1(tt):
#                                 return system_settings['reward_functions'][segment](St, At, tt)
#                             def f2(tt):
#                                 return system_settings['reward_functions'][segment + 1](St, At, tt)
#                             Rt = smooth_transform(t, f1, f2, change_points[segment] - delta, change_points[segment])
#                             # print("t =", t_current, ": smooth; during change")
#                     elif segment == len(change_points) - 1:  # at the last segment
#                         Rt = system_settings['reward_functions'][segment](St, At, t)
#                         # print("t =", t_current, ": smooth; last segment")
#                 Rewards[:, t_current] = Rt
#
#                 ## compute the next state
#                 if system_settings['state_change_type'] == 'piecewise_constant':
#                     St = system_settings['state_functions'][segment](St, At, t)
#                     # print("t =", t_current+1, " State: pwconst, segment", segment)
#                 elif system_settings['state_change_type'] == 'homogeneous':
#                     St = system_settings['state_functions'][0](St, At, t)
#                     # print("t =", t_current+1, ": homo, segment", segment)
#                 elif system_settings['state_change_type'] == 'smooth':
#                     if segment < len(change_points)-1: # before reaching the last segment
#                         # before the smooth change point: simply use the constant part
#                         if t_current <= change_points[segment] - delta - 1:
#                             St = system_settings['state_functions'][segment](St, At, t)
#                             # print("t =", t_current+1, ": smooth; before cp")
#                         else: # during the smooth change
#                             def f1(tt):
#                                 return system_settings['state_functions'][segment](St, At, tt)
#                             def f2(tt):
#                                 return system_settings['state_functions'][segment + 1](St, At, tt)
#                             St = smooth_transform(t, f1, f2, change_points[segment] - delta, change_points[segment])
#                             # print("t =", t_current+1, ": smooth; during change")
#                     elif segment == len(change_points) - 1:  # at the last segment
#                         St = system_settings['state_functions'][segment](St, At, t)
#                         # print("t =", t_current+1, ": smooth; last segment")
#                 States[:, t_current + 1, 0] = St
#                 t_current += 1
#                 previous_chaneg_point = change_points[segment]
#
#     #%% generate actions based on input policy
#     else:
#         # generate initial state S_0 and action A_0
#         States[:, 0, 0] = S0
#         # extract the current state St
#         St = copy.deepcopy(States[:, 0, 0])
#
#         myState = np.zeros([N, 2, 1])
#
#         # actual time stamp
#         t_current = 0
#         previous_chaneg_point = 0
#         for segment in range(len(change_points)): # for each change point
#             for t in range(previous_chaneg_point, change_points[segment]):
#                 # compute the current action
#                 myState[:, 0, 0] = St
#                 # generate policy
#                 At = optimal_policy_model.predict(myState).opt_action
#                 greedy_index = np.random.rand(N) < epsilon_greedy
#                 At[greedy_index] = 1 - At[greedy_index]
#                 # for i in range(N):
#                     # At[i] = np.random.binomial(At[i], epsilon_greedy, 1)[0]
#                 Actions[:, t_current] = At
#                 ## compute the current reward
#                 if system_settings['reward_change_type'] == 'piecewise_constant':
#                     Rt = system_settings['reward_functions'][segment](St, At, t)
#                 elif system_settings['reward_change_type'] == 'homogeneous':
#                     Rt = system_settings['reward_functions'][0](St, At, t)
#                 elif system_settings['reward_change_type'] == 'smooth':
#                     if segment < len(change_points)-1: # before reaching the last segment
#                         # before the smooth change point: simply use the constant part
#                         if t_current <= change_points[segment] - delta - 1:
#                             Rt = system_settings['reward_functions'][segment](St, At, t)
#                             # print("t =", t_current, ": smooth; before cp")
#                         else: # during the smooth change
#                             def f1(tt):
#                                 return system_settings['reward_functions'][segment](St, At, tt)
#                             def f2(tt):
#                                 return system_settings['reward_functions'][segment + 1](St, At, tt)
#                             Rt = smooth_transform(t, f1, f2, change_points[segment] - delta, change_points[segment])
#                             # print("t =", t_current, ": smooth; during change")
#                     elif segment == len(change_points) - 1:  # at the last segment
#                         Rt = system_settings['reward_functions'][segment](St, At, t)
#                         # print("t =", t_current, ": smooth; last segment")
#                 Rewards[:, t_current] = Rt
#
#                 ## compute the next state
#                 if system_settings['state_change_type'] == 'piecewise_constant':
#                     St = system_settings['state_functions'][segment](St, At, t)
#                     # print("t =", t_current+1, ": pwconst, segment", segment)
#                 elif system_settings['state_change_type'] == 'homogeneous':
#                     St = system_settings['state_functions'][0](St, At, t)
#                     # print("t =", t_current+1, ": homo, segment", segment)
#                 elif system_settings['state_change_type'] == 'smooth':
#                     if segment < len(change_points)-1: # before reaching the last segment
#                         # before the smooth change point: simply use the constant part
#                         if t_current <= change_points[segment] - delta - 1:
#                             St = system_settings['state_functions'][segment](St, At, t)
#                             # print("t =", t_current+1, ": smooth; before cp")
#                         else: # during the smooth change
#                             def f1(tt):
#                                 return system_settings['state_functions'][segment](St, At, tt)
#                             def f2(tt):
#                                 return system_settings['state_functions'][segment + 1](St, At, tt)
#                             St = smooth_transform(t, f1, f2, change_points[segment] - delta, change_points[segment])
#                             # print("t =", t_current+1, ": smooth; during change")
#                     elif segment == len(change_points) - 1:  # at the last segment
#                         St = system_settings['state_functions'][segment](St, At, t)
#                         # print("t =", t_current+1, ": smooth; last segment")
#                 States[:, t_current + 1, 0] = St
#                 t_current += 1
#                 previous_chaneg_point = change_points[segment]
#
#     # convert Actions to integers
#     Actions = Actions.astype(int)
#     return States, Rewards, Actions



