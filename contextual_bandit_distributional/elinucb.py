#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for conditional independence testing with epsilon-LinUCB. 
This file contains:
    1. data generation process for epsilon-LinUCB
    2. data weight calculation for epsilon-LinUCB
    3. various proposal sampling processes for epsilon-LinUCB
       in the setting of conditional indepenence testing in
       a contextual bandit
    4. proposal sampling weighting calculations for the above proposals

NB: set epsilon = 0 to simply get regular LinUCB
"""
import numpy as np

class ELinUCB:
    def __init__(self, T, epsilon, d, null, b_0=None):
        self.T = T
        self.epsilon = epsilon
        self.d = d
        self.null = null
        self.b_0 = b_0 # this variable is used for confidence interval construction, only

    def get_dataset(self):
        data = []

        # tracking parameters in LinUCB
        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        for i in range(self.T):
            # generate context
            x = np.random.multivariate_normal(np.array([1,-1]), np.eye(self.d))
            for a in range(2):
                A_a_inv = np.linalg.inv(A[a])
                p[a] = np.dot(np.dot(A_a_inv,b[a]), x) + np.sqrt(np.dot(np.dot(x,A_a_inv),x))

            U = np.random.uniform()
            # take first two actions at first two timesteps (epsilon-greedily)
            if i == 0 or i == 1:
                action = i if U < 1-self.epsilon else np.random.choice(2)
            else:
                # otherwise regular epsilon-greedy action selection
                action = np.argmax([p[0], p[1]]) if U < 1-self.epsilon else np.random.choice(2)

            # if we're doing testing, then sample accordingly from null or alternative
            if self.b_0 == None:
                # sample reward according to null or alternative correspondingly
                if self.null:
                    r = np.sum(x) + np.random.normal()
                else:
                    r = action + np.sum(x) + np.random.normal()
            # otherwise, draw the reward according to b_0
            else:
                r = self.b_0*action + np.sum(x) + np.random.normal()

            # update action-reward trackers
            A[action] = A[action] + np.outer(x, x)
            b[action] = b[action] + r*x

            data.append([[action,x],r])
        return data
           
    
    def get_data_weight(self, data, b_ci=0):
        '''This function is used both for hypothesis testing 
            and confidence interval construction.
            The b_ci in the input corresponds to confidence interval. As default, b is set
            to 0, in which case it is just regular testing'''
        # if epsilon = 1, then all have the same weight
        if self.epsilon == 1.:
            return 1.
        prod = 1.

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        # iterate through data to calculate weight
        for i in range(self.T):
            x = data[i][0][1]
            action = data[i][0][0]
            r = data[i][1] + b_ci*action

            for a in range(2):
                A_a_inv = np.linalg.inv(A[a])
                p[a] = np.dot(np.dot(A_a_inv,b[a]), x) + np.sqrt(np.dot(np.dot(x,A_a_inv),x))
            
            # if in first two time steps, argmax action is i
            if i == 0 or i == 1:
                if action == i:
                    prod *= 1-self.epsilon/2
                else:
                    prod *= self.epsilon/2
            # otherwise, it's regular eLinUCB action selection prob
            else:
                if action == np.argmax([p[0], p[1]]):
                    prod *= 1-self.epsilon/2
                else:
                    prod *= self.epsilon/2

            A[action] = A[action] + np.outer(x, x)
            b[action] = b[action] + r*x
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
    
    def simulation_X(self, data, propose_or_weight, b_ci):
        probability = 1.

        if propose_or_weight:
            sampled_data = []

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        # iterate through timesteps
        for i in range(self.T):
            # x remains the current x at index
            x = data[i][0][1]
            for a in range(2):
                A_a_inv = np.linalg.inv(A[a])
                p[a] = np.dot(np.dot(A_a_inv,b[a]), x) + np.sqrt(np.dot(np.dot(x,A_a_inv),x))
            
            # linucb argmax action selection
            if i == 0 or i == 1:
                action = i
            else:
                action = np.argmax([p[0], p[1]])

            # elinucb probabilities
            prob = [self.epsilon/2, self.epsilon/2]
            prob[action] += 1-self.epsilon

            # select action accordingly if proposing 
            # otherwise set to current action

            if propose_or_weight:
                rand_action = np.random.choice(2, p=prob)
            else:
                rand_action = data[i][0][0]

            probability *= prob[rand_action]

            r = data[i][1] + b_ci*rand_action
            if propose_or_weight:
                sampled_data.append([[rand_action, x], r])

            # update action trackers
            A[rand_action] = A[rand_action] + np.outer(x, x)
            b[rand_action] = b[rand_action] + r*x

        if propose_or_weight:
            return sampled_data, probability
        else:
            return probability

    
    
    def get_proposal(self, data, style, b_ci=0):
        if style == 's':
            return self.simulation_X(data, True, b_ci)
        if style == 'u':
            return self.uniform_X(data, True)
        if style == 'us':
            intermediary, prob = self.uniform_permute(data, True)
            return self.simulation_X(intermediary, True, b_ci)
        if style == 'uu':
            intermediary, prob = self.uniform_permute(data, True)
            return self.uniform_X(intermediary, True)
        

    def get_proposal_weight(self, proposal, starting, style, b_ci=0):
        if style == 's':
            return self.simulation_X(proposal, False, b_ci)
        if style == 'u':
            return self.uniform_X(proposal, False)
        if style == 'us':
            return self.simulation_X(proposal, False, b_ci)
        if style == 'uu':
            return self.uniform_X(proposal, False)