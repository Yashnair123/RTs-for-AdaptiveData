#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with epsilon-greedy with non-
constant g function. 
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
        regr2 = LinearRegression()

        regrs = [regr0, regr1, regr2]

        action_counters = np.zeros(3)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []
        X2, y2 = [], []

        Xs, ys = [X0, X1, X2], [y0, y1, y2]

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = np.random.multivariate_normal((-np.ones(self.d))**2, np.eye(self.d))/np.sqrt(self.d)
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                action = np.random.choice(3)
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(3)]

                U = np.random.uniform()
                action = np.argmax(predictions) if U < 1-self.epsilon else np.random.choice(3)

            # sample reward according to null or alternative correspondingly
            if self.null:
                r = np.sum(x) + 2*int(action == 0 or action == 1) + np.random.normal()
            else:
                r = 2*action + np.sum(x) + np.random.normal()

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
        regr2 = LinearRegression()

        regrs = [regr0, regr1, regr2]

        action_counters = np.zeros(3)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []
        X2, y2 = [], []

        Xs = [X0, X1, X2]
        ys = [y0, y1, y2]

        # iterate through data to calculate weight
        for i in range(self.T):
            x = data[i][0][1]
            action = data[i][0][0]
            r = data[i][1]

            if not np.all(action_counters > 0):
                prod *= 1./3 # it was uniform in this case

            else:
            # get predictions and calculate max action
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(3)]
                max_action = np.argmax(predictions)

                if action == max_action:
                    prod *= 1-self.epsilon + self.epsilon/3.
                else:
                    prod *= self.epsilon/3

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



    def restricted_uniform_permute(self, data, propose_or_weight):
        '''This sampling scheme samples uniformly over permutations with the same list
        of Z-factors'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:# sample a permutation of the data uniformly at random
            zero_inds = []
            one_inds = []

            # get the indices with zero as Z-factor and 1
            for i in range(len(data)):
                if data[i][0][0] == 0 or data[i][0][0] == 1:
                    zero_inds.append(i)
                else:
                    one_inds.append(i)

            # permute them
            perm0 = list(np.random.choice(len(zero_inds), len(zero_inds), replace=False))
            perm1 = list(np.random.choice(len(one_inds), len(one_inds), replace=False))

            zero_inds = [zero_inds[i] for i in perm0]
            one_inds = [one_inds[i] for i in perm1]
            

            shuffled_data = []
            zero_ind = 0
            one_ind = 0
            for i in range(len(data)):
                if data[i][0][0] == 0 or data[i][0][0] == 1:
                    shuffled_data.append(data[zero_inds[zero_ind]])
                    zero_ind += 1
                else:
                    shuffled_data.append(data[one_inds[one_ind]])
                    one_ind += 1

            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.
        

    def uniform_X(self, data, propose_or_weight):
        '''This sampling scheme samples X's uniformly'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = [[[np.random.choice(2) if data[i][0][0] == 0 or data[i][0][0] == 1 else 2\
                              , data[i][0][1]], data[i][1]] for i in range(self.T)]
            return sampled_data, 1.
        else:
            return 1.


    def imitation_X(self, data, propose_or_weight, b_ci):
        probability = 1.

        if propose_or_weight:
            sampled_data = []

        regr0 = LinearRegression()
        regr1 = LinearRegression()
        regr2 = LinearRegression()

        regrs = [regr0, regr1, regr2]

        action_counters = np.zeros(3)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []
        X2, y2 = [], []

        Xs, ys = [X0, X1, X2], [y0, y1, y2]

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = data[i][0][1]
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                prob = np.zeros(3) + self.epsilon/3
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(3)]
                argmax = np.argmax(predictions)

                prob = np.zeros(3) + self.epsilon/3
                prob[argmax] += 1-self.epsilon

            if data[i][0][0] == 0 or data[i][0][0] == 1:
                new_prob = np.zeros(3)
                new_prob[0], new_prob[1] = prob[0], prob[1]
                new_prob = new_prob/np.sum(new_prob)
            else:
                new_prob = np.zeros(3)
                new_prob[-1] = 1.

            if propose_or_weight:
                action = np.random.choice([0,1,2], p=new_prob)
            else:
                action = data[i][0][0]

            probability *= new_prob[action]
                # if action == np.argmax(predictions):
                #     probability *= (1-self.epsilon + self.epsilon/3)
                # else:
                #     probability *= self.epsilon/3

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
        
    def combined(self, data, propose_or_weight, b_ci=0):
        '''This sampling scheme samples X's from the combined 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        prod = 1.
        if propose_or_weight:
            sampled_data = []

        regr0 = LinearRegression()
        regr1 = LinearRegression()
        regr2 = LinearRegression()

        regrs = [regr0, regr1, regr2]

        action_counters = np.zeros(3)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [], []
        X1, y1 = [], []
        X2, y2 = [], []

        Xs, ys = [X0, X1, X2], [y0, y1, y2]
        
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            x = data[i][0][1]
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                argmax = 'undecided'
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(3)]
                argmax = np.argmax(predictions)

            # set corresponding p-vector
            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                p /= np.sum(p)

            # p-vector induces distribution over Z's:
            p_z = np.zeros(2)
            #index 0 corresponds to [0,1], and index 1 correponds to 2
            p_z[0] = p[0] + p[1]
            p_z[1] = p[2]
            
            # sample from remaining timesteps according to this vector
            prob = np.zeros(self.T)
            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    if data[i_][0] == 0 or data[i_][0] == 1:
                        prob[i_] = p_z[0]
                    else:
                        prob[i_] = p_z[1]

            # normalize and sample if proposing, otherwise set to current
            prob = prob/np.sum(prob)
            if propose_or_weight:
                sample = np.random.choice(self.T, p=prob)
            else:
                sample = i
            prod *= prob[sample]

            # mark that we selected this index
            curr_selected[sample] = 1

            # now transform p-vector into distribution over X's given this Z
            select_a = data[sample][0][0]
            select_x = data[sample][0][1]
            select_r = data[sample][1]



            # select argmax for the new select_x to get new p vector
            if not np.all(action_counters > 0):
                new_argmax = 'undecided'
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], select_x)]) for a in range(3)]
                new_argmax = np.argmax(predictions)

            # set corresponding p-vector
            update_p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                update_p[new_argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                update_p /= np.sum(update_p)

            if select_a == 0 or select_a == 1:
                new_p = np.zeros(2)
                new_p[0], new_p[1] = update_p[0], update_p[1]
                new_p = new_p/np.sum(new_p)

                if propose_or_weight:
                    new_a = np.random.choice([0,1], p=new_p)
                else:
                    new_a = select_a # the action is the one already there if computing weight
                
                prod *= new_p[[0,1].index(new_a)]
            else:
                if propose_or_weight:
                    new_a = 2
                else:
                    new_a = select_a
                prod *= 1.

            if propose_or_weight:
                sampled_data.append(([new_a,select_x], select_r)) # only append independent version to sampled data, so subtract offset
            
            # update epsilon-greedy action-reward tracker
            ys[new_a].append(select_r)
            Xs[new_a].append(np.append([1], select_x))

            regrs[new_a].fit(Xs[new_a], ys[new_a])

            action_counters[new_a] += 1

        if propose_or_weight:
            return sampled_data, prod
        else:
            return prod


    def get_proposal(self, data, style, b_ci=0):
        if style == 'i_X':
            return self.imitation_X(data, True, b_ci)
        if style == 'u':
            return self.uniform_X(data, True)
        if style == 'ui_X':
            intermediary, prob = self.uniform_permute(data, True)
            return self.imitation_X(intermediary, True, b_ci)
        if style == 'rui_X':
            intermediary, prob = self.restricted_uniform_permute(data, True)
            return self.imitation_X(intermediary, True, b_ci)
        if style == 'uu_X':
            intermediary, prob = self.uniform_permute(data, True)
            return self.uniform_X(intermediary, True)
        if style == 'comb':
            return self.combined(data, True, b_ci)
        

    def get_proposal_weight(self, proposal, starting, style, b_ci=0):
        if style == 'i_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'u':
            return self.uniform_X(proposal, False)
        if style == 'ui_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'rui_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'uu_X':
            return self.uniform_X(proposal, False)
        if style == 'comb':
            return self.combined(proposal, False, b_ci)