'''
Simulate stationary time series data and apply Q-learning.
Simulate real data with 4-dimensional states
'''

import numpy as np
from scipy.stats import multivariate_normal
from copy import copy

# %%
class simulate_data():
    def __init__(self, N, T, change_pt=16):
        '''

        :param N: number of trajectories / episodes
        :param T: number of time points
        :param delta: the proportion of time line for a smooth transition
        # :param mean0: mean vector for the initial state S_0
        # :param cov0: covariance matrix for the initial state S_0
        '''
        self.N = N
        self.T = T
        # initialize A_t
        self.At = 1
        # initialize S_t
        self.St = 1
        # initialize S_t-1
        self.Stm1 = 1
        self.change_pt = int(change_pt)
        self.err_mean = np.zeros(3)
        self.err_cov = np.diag([1, 1, 0.2])



    # def transition_pwconstant2(self, t):
    #     '''
    #     Generate time-homogeneous transition at time t
    #     # :param t: the time at which we generate a transition
    #     :return: a scalar of state
    #     '''
    #     At1 = 2 * self.At - 1
    #     err_cov = np.diag([2.5, 1.5, 0.4])
    #     if t <= self.change_pt:
    #         # transition_matrix = np.array([[10 - 4.0 * At1, 0.2 * At1 + 0.5, 0.08 * At1, 0],
    #         #                               [12 - 0.7 * At1, 0, 0.4, 0],
    #         #                               [2 + 0.3 * At1, 0.01, 0, 0.7]])
    #         # print('t =', t, ", first")
    #         transition_matrix = np.array([[10 + 0.6 * At1, 0.4 + 0.25 * At1, 0.1 - 0.1*At1, -0.04, 0.1],
    #                                       [11 - 0.4 * At1, 0.05, 0, 0.4, 0],
    #                                       [1.2 - 0.5 * At1, -0.02, 0, 0.03 + 0.03*At1, 0.8]])#.03 + 0.01*At1
    #     elif t in range(self.change_pt, self.T+1):
    #         transition_matrix = np.array([[10 - 0.6 * At1, 0.4 - 0.2 * At1, 0.1 + 0.2*At1, 0.04, -0.1],
    #                                       [11 - 0.4 * At1, 0.05, 0, 0.4, 0],
    #                                       [1.2 + 0.5 * At1, -0.02, 0, 0.03 - 0.03*At1, 0.8]])
    #         # transition_matrix = np.array([[5.6 + 4.0 * At1, 0.54 + 0.04 * At1, 0.15 - 0.02*At1, 0.02, -0.03-0.05*At1],
    #         #                               [12.4 + 0.57 * At1, 0.04+0.02*At1, -0.03 - 0.02*At1, 0.37 - 0.03*At1, 0.01*At1],
    #         #                               [2.4 + 0.3 * At1, -0.02, 0.02+0.01*At1, -0.02-0.02*At1, 0.74]])
    #         # print('t =', t, ", second")
    #
    #     # else:
    #     #     print("Time t out of range")
    #     #     return 0
    #     St_full = np.insert(self.St, 0, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # St_full = np.insert(self.St, 0, 1, axis=0)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, err_cov)


    def transition1(self, t):
        '''
        Generate transition before change point
        '''
        At1 = 2 * self.At - 1
        # print('t =', t, ", first")
        transition_matrix = np.array([[10 + 0.6 * At1, 0.4 + 0.3 * At1, -0.04, 0.1, 0.1 - 0.1*At1],
                                      [11 - 0.4 * At1, 0.05, 0.4, 0, 0],
                                      [1.2 - 0.5 * At1, -0.02, 0.03 + 0.03*At1, 0.8, 0]])#.03 + 0.01*At1
        St_full = np.insert(self.St, 3, self.Stm1, axis=0)
        St_full = np.insert(St_full, 0, 1, axis=0)
        # print("St_full =", St_full)
        return transition_matrix @ St_full + \
               multivariate_normal.rvs(self.err_mean, self.err_cov)

    # def transition2(self, t):
    #     '''
    #     Generate transition before change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     # print('t =', t, ", first")
    #     transition_matrix = np.array([[10 + 0.6 * At1, 0.4 + 0.3 * At1, -0.04, 0.1, 0.1 - 0.1*At1],
    #                                   [11 - 0.4 * At1, 0.05, 0.4, 0, 0],
    #                                   [1.2 - 0.5 * At1, -0.02, 0.03 + 0.03*At1, 0.8, 0]])#.03 + 0.01*At1
    #     St_full = np.insert(self.St, 3, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # print("St_full =", St_full)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)

    def transition2(self, t):
        '''
        Generate transition after change point
        '''
        At1 = 2 * self.At - 1
        transition_matrix = np.array([[10 - 0.6 * At1, 0.4 - 0.3 * At1, -0.04, -0.1, 0.1 + 0.1 * At1],
                                      [11 + 0.4 * At1, 0.05, 0.4, 0, 0],
                                      [1.2 + 0.5 * At1, -0.02, 0.03 - 0.03 * At1, 0.8, 0]])
        St_full = np.insert(self.St, 3, self.Stm1, axis=0)
        St_full = np.insert(St_full, 0, 1, axis=0)
        # print("St_full =", St_full)
        return transition_matrix @ St_full + \
               multivariate_normal.rvs(self.err_mean, self.err_cov)

    # def transition2(self, t):
    #     '''
    #     Generate transition after change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     transition_matrix = np.array([[1000 - 0.6 * At1, 0.4 - 0.3 * At1, -0.04, -0.1, 0.1 + 0.1 * At1],
    #                                   [1100 + 0.4 * At1, 0.05, 0.4, 0, 0],
    #                                   [1200 + 0.5 * At1, -0.02, 0.03 - 0.03 * At1, 0.8, 0]])
    #     St_full = np.insert(self.St, 3, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # print("St_full =", St_full)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)

    # def transition1(self, t):
    #     '''
    #     Generate transition before change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     # print('t =', t, ", first")
    #     transition_matrix = np.array([[10 + 0.6 * At1, 0.4 + 0.3 * At1, 0.1 - 0.2*At1, -0.04, 0.1],
    #                                   [11 - 0.4 * At1, 0.05, 0, 0.4, 0],
    #                                   [1.2 - 0.5 * At1, -0.02, 0, 0.03 + 0.03*At1, 0.8]])#.03 + 0.01*At1
    #     St_full = np.insert(self.St, 0, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # St_full = np.insert(self.St, 0, 1, axis=0)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)
    #
    #
    # def transition2(self, t):
    #     '''
    #     Generate transition after change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     transition_matrix = np.array([[10 - 0.6 * At1, 0.4 - 0.3 * At1, 0.1 + 0.2 * At1, 0.04, -0.1],
    #                                   [11 - 0.4 * At1, 0.05, 0, 0.4, 0],
    #                                   [1.2 + 0.5 * At1, -0.02, 0, 0.03 - 0.03 * At1, 0.8]])
    #     # transition_matrix = np.array([[10 - 0.6 * At1, 0.4 - 0.3 * At1, 0.1 + 0.2 * At1, 0.04, -0.1],
    #     #                               [11 - 0.4 * At1, 0.05, 0, 0.4, 0],
    #     #                               [1.2 + 0.5 * At1, -0.02, 0, 0.03 - 0.03 * At1, 0.8]])
    #     St_full = np.insert(self.St, 0, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # St_full = np.insert(self.St, 0, 1, axis=0)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)


    def simulate(self, seed=0, T0=0, T1=26, burnin = 0,
                 optimal_policy_model=None, normalized = [0.0, 1.0]):
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
        # if T1 > self.T:
        #     print("error: terminal time T1 of the simulated trajectory should be no more than the total number of time points T")
        #     return 0

        # set seed
        np.random.seed(seed)
        States = np.zeros([self.N, T1-T0 + 2, 4])  # S[i][t]
        # Rewards = np.zeros([self.N, T1-T0])  # R[i,t]
        Actions = np.zeros([self.N, T1-T0+1])  # Actions[i,t]

        if optimal_policy_model is None:
            for i in range(self.N):  # for each individual

                ## initial
                # generate initial state S_i0
                self.St = np.ones(3)
                self.St[0] = np.random.normal(20, 3) # step count of week t
                self.St[1] = np.random.normal(20, 2) # sleep of week t
                self.St[2] = np.random.normal(7, 1) # mood score of week t
                if burnin == 0:
                    States[i, 0, :3] = self.St
                # generate policy
                self.At = np.random.binomial(1, 0.25)
                # record S_t-1
                St_minus_1 = copy(self.St)
                self.Stm1 = copy(St_minus_1[0])

                ## for the subsequent time points
                t_recorded = 0
                for t in range(-burnin, self.change_pt-T0):  # each time point t
                    if t >= 0:
                        States[i, t_recorded, :3] = self.St
                        Actions[i, t_recorded] = self.At
                        # print("t_recorded = ", t_recorded, "ONE")
                        # print("St = ", self.St)
                        t_recorded += 1
                    if t >= 1:
                        self.Stm1 = copy(St_minus_1[0])
                        St_minus_1 = copy(self.St)
                    # generate S_i,t+1
                    self.St = self.transition1(t)
                    # generate policy
                    self.At = np.random.binomial(1, 0.25)
                    # self.At = 1
                # print("burned")
                # self.St = copy(States[i, 1, :3])
                for t in range(0, T1 - self.change_pt+2):  # each time point t
                    if t >= 0:
                        if t_recorded <= T1-T0:
                            Actions[i, t_recorded] = self.At
                        States[i, t_recorded, :3] = self.St
                        # print("t_recorded = ", t_recorded, "TWO")
                        # print("St = ", self.St)
                        t_recorded += 1
                    self.Stm1 = copy(St_minus_1[0])
                    St_minus_1 = copy(self.St)
                    # generate S_i,t+1
                    self.St = self.transition2(t)
                    # generate policy
                    self.At = np.random.binomial(1, 0.25)


        #%% if optimal policy is input
        else:
            myState = np.zeros([1, 2, 4])
            for i in range(self.N):  # for each individual
                ## initial t=0
                # generate initial state S_i0
                self.St = np.ones(3)
                self.St[0] = np.random.normal(20, 3) # step count of week t
                self.St[1] = np.random.normal(20, 2) # sleep of week t
                self.St[2] = np.random.normal(7, 1) # mood score of week t
                if burnin == 0:
                    States[i, 0, :3] = self.St
                # record S_t-1
                St_minus_1 = copy(self.St)
                self.Stm1 = copy(St_minus_1[0])
                myState[0, 0, :3] = self.St
                myState[0, 0, 3] = self.Stm1
                myState[0, 0, :] -= normalized[0]
                myState[0, 0, :] /= normalized[1]
                # generate policy
                self.At = optimal_policy_model.predict(myState).opt_action[0]
                # self.At = 1

                ## for the subsequent time points
                t_recorded = 0
                if T0 < self.change_pt:

                    ## for the subsequent time points
                    t_recorded = 0
                    # if T0 <= self.change_pt:
                    for t in range(-burnin, self.change_pt - T0):  # each time point t
                        # print("xxx")
                        if t == 0:
                            # print('t =', t_recorded, ", first")
                            States[i, t_recorded, :3] = self.St
                        elif t >= 1:
                            Actions[i, t_recorded] = self.At
                            States[i, t_recorded + 1, :3] = self.St
                            self.Stm1 = copy(States[i, t_recorded, 0])
                            # print('t =', t_recorded+1, ", first")
                            t_recorded += 1
                        # self.Stm1 = copy(self.St[0])
                        # generate S_i,t+1
                        self.St = self.transition1(t)
                        # generate action according to policy
                        myState[0, 0, :3] = self.St
                        myState[0, 0, 3] = self.Stm1
                        myState[0, 0, :] -= normalized[0]
                        myState[0, 0, :] /= normalized[1]
                        self.At = optimal_policy_model.predict(myState).opt_action[0]

                # after change point
                for t in range(0, T1 - self.change_pt+2):  # each time point t
                    if t >= 0:
                        if t_recorded <= T1 - T0:
                            Actions[i, t_recorded] = self.At
                        States[i, t_recorded, :3] = self.St
                        # self.Stm1 = copy(States[i, t_recorded, 0])
                        # print('t =', t_recorded+1, ", second")
                        t_recorded += 1
                    self.Stm1 = copy(St_minus_1[0])
                    St_minus_1 = copy(self.St)
                    # print("self.At", self.At)
                    # generate S_i,t+1
                    self.St = self.transition2(t)
                    # generate policy
                    myState[0, 0, :3] = self.St
                    myState[0, 0, 3] = copy(St_minus_1[0])
                    myState[0, 0, :] -= normalized[0]
                    myState[0, 0, :] /= normalized[1]
                    self.At = optimal_policy_model.predict(myState).opt_action[0]
                    # self.At = 1
                    # print(myState)
                    # print(self.At)
                States[i, t_recorded, :3] = self.St

        # add weekly step of week t-1 to S4
        States[:,1:,3] = States[:,:(T1-T0+1),0]
        States = np.delete(States, 0, axis=1)
        # make rewards
        Rewards = copy(States[:,:(T1-T0),0])
        # Rewards = copy(States[:,1:,0])
        # convert Actions to integers
        Actions = np.delete(Actions, 0, axis=1)
        Actions = Actions.astype(int)
        return States, Rewards, Actions
