#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with epsilon-greedy. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various resampling procedures for epsilon-greedy
       in the setting of non-stationarity testing in
       a contextual bandit
    4. resampling weighting calculations for the above resampling procedures

NB: set epsilon = 0 to simply get regular LinUCB
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


class EpsilonGreedy:
    def __init__(self, T, epsilon, d, null):
        self.T = T
        self.epsilon = epsilon
        self.d = d
        self.null = null

    def get_dataset(self):
        data = []
        regr0 = LinearRegression()
        regr1 = LinearRegression()

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []

        Xs, ys = [X0, X1], [y0, y1]

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = np.random.multivariate_normal(np.array([1,-1]), np.eye(self.d))
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                action = np.random.choice(2)
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(2)]

                U = np.random.uniform()
                action = np.argmax(predictions) if U < 1-self.epsilon else np.random.choice(2)

            # sample reward according to null or alternative correspondingly
            if self.null:
                r = np.sum(x) + np.random.normal()
            else:
                r = action + np.sum(x) + np.random.normal()

            ys[action].append(r)
            Xs[action].append(np.append([1], x))

            regrs[action].fit(Xs[action], ys[action])

            action_counters[action] += 1
            data.append([[action,x],r])
        return data
           
    
    def get_data_weight(self, data):
        # if epsilon = 1, then all have the same weight
        if self.epsilon == 1.:
            return 1.
        
        prod = 1.

        regr0 = LinearRegression()
        regr1 = LinearRegression()

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []

        Xs = [X0, X1]
        ys = [y0, y1]

        # iterate through data to calculate weight
        for i in range(self.T):
            x = data[i][0][1]
            action = data[i][0][0]
            r = data[i][1]

            if not np.all(action_counters > 0):
                prod *= 0.5 # it was uniform in this case

            else:
            # get predictions and calculate max action
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(2)]
                max_action = np.argmax(predictions)

                if action == max_action:
                    prod *= 1-self.epsilon/2
                else:
                    prod *= self.epsilon/2

            ys[action].append(r)
            Xs[action].append(np.append([1], x))
            regrs[action].fit(Xs[action],ys[action])

            action_counters[action] += 1
        return prod


    def uniform_permute(self, data, propose_or_weight):
        '''This sampling scheme samples uniformly over permutations'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            # sample a permutation of the data uniformly at random
            perm = list(np.random.choice(self.T, self.T, replace=False))

            # permute the data
            shuffled_data = [[[data[perm[i]][0][0], data[perm[i]][0][1]], data[perm[i]][1]] for i in range(self.T)]

            # return the data and its weight
            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.

    def uniform_X(self, data, propose_or_weight):
        '''This sampling scheme samples X's uniformly'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = [[[np.random.choice(2), data[i][0][1]], data[i][1]] for i in range(self.T)]
            return sampled_data, 1.
        else:
            return 1.


    def imitation_X(self, data, propose_or_weight, b_ci):
        probability = 1.

        if propose_or_weight:
            sampled_data = []

        regr0 = LinearRegression()
        regr1 = LinearRegression()

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []

        Xs, ys = [X0, X1], [y0, y1]

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = data[i][0][1]
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                if propose_or_weight:
                    action = np.random.choice(2)
                else:
                    action = data[i][0][0]
                probability *= 0.5
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(2)]

                U = np.random.uniform()
                
                if propose_or_weight:
                    action = np.argmax(predictions) if U < 1-self.epsilon else np.random.choice(2)
                else:
                    action = data[i][0][0]

                if action == np.argmax(predictions):
                    probability *= (1-self.epsilon/2)
                else:
                    probability *= self.epsilon/2

            # set reward according to already seen
            r = data[i][1]
            if propose_or_weight:
                sampled_data.append([[action, x], r])

            ys[action].append(r)
            Xs[action].append(np.append([1], x))

            regrs[action].fit(Xs[action], ys[action])

            action_counters[action] += 1

        if propose_or_weight:
            return sampled_data, probability
        else:
            return probability


    def get_proposal(self, data, style, b_ci=0):
        if style == 'i_X':
            return self.imitation_X(data, True, b_ci)
        if style == 'u':
            return self.uniform_X(data, True)
        if style == 'ui_X':
            intermediary, prob = self.uniform_permute(data, True)
            return self.imitation_X(intermediary, True, b_ci)
        if style == 'uu_X':
            intermediary, prob = self.uniform_permute(data, True)
            return self.uniform_X(intermediary, True)
        

    def get_proposal_weight(self, proposal, starting, style, b_ci=0):
        if style == 'i_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'u':
            return self.uniform_X(proposal, False)
        if style == 'ui_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'uu_X':
            return self.uniform_X(proposal, False)