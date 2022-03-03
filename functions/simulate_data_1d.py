'''
Simulate stationary time series data and apply Q-learning.
Generate 1 dimensional states
'''


import numpy as np
import math

# %%
class simulate_data():
    def __init__(self, N, T, delta=1.0):
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
        # initialize R_t
        # self.Rt = 0.0
        self.Td2 = int(T / 2)

        if delta != None:
            self.Td2_minus_delta = int(T / 2 - delta*T) #int(T / 2)
            self.Td2_plus_delta = int(T / 2 + delta*T)


    def transition_homo(self, mean, cov):
        '''
        Generate time-homogeneous transition at time t
        # :param t: the time at which we generate a transition
        :return: a scalar of state
        '''
        return 0.5 * (2.0 * self.At - 1.0) * self.St + np.random.normal(mean, cov, 1)


    def transition_pwconstant2(self, t, mean, cov):
        '''
        Generate time-homogeneous transition at time t
        # :param t: the time at which we generate a transition
        :return: a scalar of state
        '''
        if t < self.Td2:
            return -0.5 * self.St * (2.0 * self.At - 1.0) + np.random.normal(mean, cov, 1)
        elif t >= self.Td2:
            return 0.5 * self.St * (2.0 * self.At - 1.0) + np.random.normal(mean, cov, 1)
        # else:
        #     print("Time t out of range")
        #     return 0



    def transition_smooth2(self, t, mean, cov, w=1.0):
        '''
        Generate smooth transition function in time with 1 smooth change point
        t = the time at which we generate a reward
        :return: a scalar of reward at time t
        '''
        def R1(t):
            return -0.5 * self.St * (2.0 * self.At - 1.0)

        def R2(t):
            return 0.5 * self.St * (2.0 * self.At - 1.0)

        if t < self.Td2_minus_delta+1: # the first piece
            # print('first piece')
            return R1(t) + np.random.normal(mean, cov, 1)
        elif t >= self.Td2: # the second piece
            # print('second piece')
        # elif t in range(self.Td2_plus_delta-1, self.T): # the second piece
            return R2(t) + np.random.normal(mean, cov, 1)
        # elif t > self.T: # invalid time
        #     print("Time t out of range")
        #     return 0
        elif t in range(self.Td2_minus_delta+1, self.Td2): # the transition piece
            # print('transition piece')
            return self.smooth_transform(t, R1, R2, self.Td2_minus_delta, self.Td2, w) + np.random.normal(mean, cov, 1)


    def reward_homo(self):
        '''
        Generate time-homogeneous reward function in time consisting of 2 pieces
        t = the time at which we generate a reward
        :return: a scalar of reward at time t
        '''

        return 0.25*self.St[0]**2 * (2.0 * self.At - 1.0) + 4*self.St[0]
        # return 0.5 * self.St[0]


    def reward_pwconstant2(self, t):
        '''
        Generate piecewise constant reward function in time consisting of 2 pieces
        t = the time at which we generate a reward
        :return: a scalar of reward at time t
        '''

        if t < self.Td2:
            # return -3 * self.St[0] #* (2.0 * self.At - 1.0)
            return -1.5 * (2.0 * self.At - 1.0) * self.St[0] #* (2.0 * self.At - 1.0)
        elif t >= self.Td2:
            return (2.0 * self.At - 1.0) * self.St[0]#-1.0 * self.St[0] + 0.5 * (2.0 * self.At - 1.0)


    def reward_smooth2(self, t, w=1.0):
        '''
        Generate smooth reward function in time with 1 smooth change point
        t = the time at which we generate a reward
        :return: a scalar of reward at time t
        '''
        def R1(t):
            # return - 3 * self.St[0] #* (2.0 * self.At - 1.0)
            return -1.5 * (2.0 * self.At - 1.0) * self.St[0] #* (2.0 * self.At - 1.0)

        def R2(t):
            return (2.0 * self.At - 1.0) * self.St[0] #+ 1.0 * (2.0 * self.At - 1.0)

        if t < self.Td2_minus_delta+1:
            return R1(t)
        if t >= self.Td2: # the second piece
            return R2(t)
        # elif t > self.T: # invalid time
        #     print("Time t out of range")
        #     return 0
        else: # the transition piece
            return self.smooth_transform(t, R1, R2, self.Td2_minus_delta, self.Td2, w)


    def smooth_transform(self, x, f1, f2, x0, x1, w=1.0):
        '''
        Define a smooth transformation function on interval [x0, x1], connecting smoothly function f1 on x < x0 and f2 on x > x1
        :param x: the transformed function value to be evaluated at
        :return: a scalar of transformed function value
        '''
        def psi(t):
            if t <= 0:
                # print("NULL")
                return 0
            else:
                return math.exp(-w / t)

        if x > x0:
            ratio = (x - x0) / (x1 - x0)
            return f1(x) + (f2(x) - f1(x)) / (psi(ratio) + psi(1-ratio)) * psi(ratio)
        elif x <= x0:
            return f1(x)
        elif x >= x1:
            return f2(x)
        # return 0 * x

    def simulate(self, mean0, cov0, transition_function, reward_function, seed=0,
                 S0=None, A0=None, T0=0, T1=100, optimal_policy_model=None):
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
        States = np.zeros([self.N, T1-T0 + 1, 1])  # S[i][t]
        Rewards = np.zeros([self.N, T1-T0])  # R[i,t]
        Actions = np.zeros([self.N, T1-T0])  # Actions[i,t]

        if not S0 is None:
            States[:, 0, :] = S0

        if not A0 is None:
            Actions[:, 0] = A0

        # def sigmoid(x):
        #     return 1.0 / (1.0 + np.exp(-x))

        if optimal_policy_model is None:
            t = 0  # time 0
            for i in range(self.N): # for each individual

                # generate initial state S_i0
                if S0 is None:
                    self.St = np.random.normal(mean0, cov0, 1)
                    # print(self.St)
                    States[i, 0, :] = self.St
                else:
                    self.St = States[i, 0, :]

                if A0 is None:
                    # generate policy
                    self.At = np.random.binomial(1, 0.5, 1)
                    Actions[i, t] = self.At
                else:
                    self.At = Actions[i, t]

                # generate immediate response R_i,t
                Rewards[i, t] = reward_function(t+T0)
                # print("St = ", self.St)
                # print("St = ", self.St)
                # print("Rewards[i, t] = ", Rewards[i, t])
                # generate S_i,t+1
                self.St = transition_function(t+T0)
                States[i, t + 1, :] = self.St

            # for the subsequent time points
            for i in range(self.N):  # each individual i
                self.St = States[i, 1, :]
                for t in range(1, T1-T0):  # each time point t
                    # generate policy
                    self.At = np.random.binomial(1, 0.5, 1)
                    Actions[i, t] = self.At
                    # generate immediate response R_i,t
                    Rewards[i, t] = reward_function(t+T0)
                    # generate S_i,t+1
                    self.St = transition_function(t+T0)
                    States[i, t + 1, :] = self.St


        else:
            myState = np.zeros([1, 2, 1])

            t = 0  # time 0
            for i in range(self.N):  # for each individual

                # generate initial state S_i0
                if S0 is None:
                    self.St = np.random.normal(mean0, cov0, 1)
                    # print(self.St)
                    States[i, 0, :] = self.St
                else:
                    self.St = States[i, 0, :]

                # generate action
                myState[0, 0, :] = self.St
                self.At = optimal_policy_model.predict(myState).opt_action
                Actions[i, t] = self.At

                # generate immediate response R_i,t
                Rewards[i, t] = reward_function(t + T0)
                self.St = transition_function(t + T0)
                States[i, t + 1, :] = self.St

            # other times
            for i in range(self.N):  # each episode i
                self.St = States[i, 1, :]
                for t in range(1, T1-T0):  # each time point t
                    # print(t)
                    myState[0, 0, :] = self.St
                    # generate policy
                    self.At = optimal_policy_model.predict(myState).opt_action
                    Actions[i, t] = self.At

                    # generate immediate response R_i,t
                    Rewards[i, t] = reward_function(t+T0)

                    # generate S_i,t+1
                    self.St = transition_function(t+T0)
                    States[i, t+1, :] = self.St

        # convert Actions to integers
        Actions = Actions.astype(int)
        return States, Rewards, Actions
