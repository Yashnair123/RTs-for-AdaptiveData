#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with epsilon-greedy. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various resampling procedures for epsilon-greedy
       in the setting of non-stationarity testing in
       a contextual bandit
    4. resampling weighting calculations for the above resampling procedures
"""
import numpy as np
from sklearn.linear_model import Lasso

class EpsilonGreedy:
    def __init__(self, T, epsilon, d, null, conditional):
        self.T = T
        self.epsilon = epsilon
        self.d = d
        self.null = null
        self.coin_flips = []
        self.conditional = conditional

    def get_dataset(self):
        data = []
        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs, ys = [X0, X1], [y0, y1]

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = np.random.multivariate_normal(np.ones(self.d), np.eye(self.d))
            
            # if haven't seen all, then select action uniformly at random
            if not np.all(action_counters > 0):
                action = np.random.choice(2)
                
                if self.conditional:
                    if action == 1: # 1 is considered the greedy action, when undecided
                        self.coin_flips.append(1)
                    else:
                        self.coin_flips.append(0)
            
            # otherwise, select epsilon-greedily
            else:
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(2)]

                U = np.random.uniform()
    
                action = np.argmax(predictions) if U < 1-self.epsilon else np.random.choice(2)

                # collect coin flips if conditional
                if self.conditional:
                    if action == np.argmax(predictions):
                        self.coin_flips.append(1)
                    else:
                        self.coin_flips.append(0)

            # sample reward according to null or alternative correspondingly
            if i == self.T-1 and not self.null:
                r = 5*(2*action-1) + np.sum(x[:10]) + np.random.normal()
            else:
                r = -5*(2*action-1) + np.sum(x[:10]) + np.random.normal()

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

        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs = [X0, X1]
        ys = [y0, y1]

        # iterate through data to calculate weight
        for i in range(self.T):
            x = data[i][0][1]
            action = data[i][0][0]
            r = data[i][1]

            if not np.all(action_counters > 0):
                if self.conditional:
                    coin_flip = self.coin_flips[i]
                    action_to_take = 1 if coin_flip == 1 else 0
                    if action != action_to_take:
                        return 0.
                else:
                    prod *= 0.5 # it was uniform in this case

            else:
            # get predictions and calculate max action
                predictions = [regrs[a].predict([np.append([1], x)]) for a in range(2)]
                max_action = np.argmax(predictions)

                if self.conditional:
                    coin_flip = self.coin_flips[i]
                    if coin_flip == 1:
                        if action != max_action:
                            return 0.
                    else:
                        if action == max_action:
                            return 0.
                else:
                    if action == max_action:
                        prod *= 1-self.epsilon/2
                    else:
                        prod *= self.epsilon/2

            ys[action].append(r)
            Xs[action].append(np.append([1], x))
            regrs[action].fit(Xs[action],ys[action])

            action_counters[action] += 1
        return prod

    
    def uniform(self, data, propose_or_weight):
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

    def imitation(self, data, propose_or_weight):
        ''''The imitation distribution samples without replacement, 
        proportional to the policy probabilities'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        prob = 1.

        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs = [X0, X1]
        ys = [y0, y1]

        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if not np.all(action_counters > 0):
                        max_action = 'undecided'
                    else:
                        predictions = [regrs[a].predict([np.append([1], x_curr)]) for a in range(2)]
                        max_action = np.argmax(predictions)

                    if max_action == 'undecided':
                        p_vec[i_] = 0.5
                    else:
                        if a_curr == max_action:
                            p_vec[i_] = 1-self.epsilon/2
                        else:
                            p_vec[i_] = self.epsilon/2

            if np.all(p_vec == 0):
                for i_ in range(self.T):
                    if curr_selected[i_] == 0:
                        p_vec[i_] = 1.
            # normalize p-vector
            p_vec = p_vec/np.sum(p_vec)
            
            # if proposing, then sample according to p, 
            # if calculating the weight, then calculate the 
            # probability of having selected ith index, using p
            if propose_or_weight:
                sample = np.random.choice(self.T, p=p_vec)
            else:
                sample = i

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data if proposing
            # and update probability correspondingly, in either case
            ((sampled_a, sampled_x), sampled_r) = ((data[sample][0][0], data[sample][0][1]), data[sample][1])
            if propose_or_weight:
                shuffled_data.append(((sampled_a, sampled_x), sampled_r))
            prob *= p_vec[sample] if self.epsilon != 0 else 1. # if deterministic, then uniform

            prob *= 100 if self.T == 100 else 1. # this line is to make sure the weights 
            # don't get too small

            ys[sampled_a].append(sampled_r)
            Xs[sampled_a].append(np.append([1], sampled_x))
            regrs[sampled_a].fit(Xs[sampled_a],ys[sampled_a])

            action_counters[sampled_a] += 1
        

        if propose_or_weight:
            return shuffled_data, prob 
        else:
            return prob


    def re_imitation(self, data, propose_or_weight):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''

        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            return self.re_imitation_propose(data)
        else:
            return self.re_imitation_weight(data)


    def re_imitation_propose(self, data):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function just samples from the proposal'''

        prob = 1.

        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs = [X0, X1]
        ys = [y0, y1]
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            # determine if going to be greedy or not
            if np.all(action_counters > 0):
                greedy_or_not = np.random.choice(2, p=[self.epsilon/2, 1-self.epsilon/2])
            else:
                greedy_or_not = np.random.choice(2)

            # track to see if it was forced to take a certain choice of greedy_or_not
            forced = True
            
            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if not np.all(action_counters > 0):
                        max_action = 'undecided'
                    else:
                        predictions = [regrs[a].predict([np.append([1], x_curr)]) for a in range(2)]
                        max_action = np.argmax(predictions)    

                    # max action is 1 if undecided
                    if max_action == 'undecided':
                        max_action = 1


                    # set p vector probs based on max action (if decided)
                    if greedy_or_not == 1:
                        if a_curr == max_action:
                            p_vec[i_] = 1.
                        else:
                            forced = False
                    else:
                        if a_curr != max_action:
                            p_vec[i_] = 1.
                        else:
                            forced = False

            
            # if there are no remaining indices, then we must select the other action
            # flip, and the selectable indices are precisely those not already selected
            # otherwise, multiply by the probability of selecting action a
            if np.all(p_vec == 0) or forced:
                for i_ in range(self.T):
                    if curr_selected[i_] != 1:
                        p_vec[i_] = 1.
            else:
                if not np.all(action_counters > 0):
                    prob *= 0.5
                else:
                    prob *= 1-self.epsilon/2 if greedy_or_not == 1 else self.epsilon/2

            

            # normalize p-vector
            p_vec = p_vec/np.sum(p_vec)
            
            # sample according to p, 
            sample = np.random.choice(self.T, p=p_vec)

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data
            # and update probability correspondingly, in either case
            ((sampled_a, sampled_x), sampled_r) = ((data[sample][0][0], data[sample][0][1]), data[sample][1])
            shuffled_data.append(((sampled_a, sampled_x), sampled_r))
            prob *= p_vec[sample]

            prob *= 100 if self.T == 100 else 1. # this line is to make sure the weights 
            # don't get too small

            ys[sampled_a].append(sampled_r)
            Xs[sampled_a].append(np.append([1], sampled_x))
            regrs[sampled_a].fit(Xs[sampled_a],ys[sampled_a])

            action_counters[sampled_a] += 1

        return shuffled_data, prob

    
    def re_imitation_weight(self, data):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function calculates weights'''

        prob = 1.

        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs = [X0, X1]
        ys = [y0, y1]

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            (a_i, x_i) = data[i][0]
            # calculate if current action is greedy_or_not
            
            if not np.all(action_counters > 0):
                max_action = 1
            else:
                predictions = [regrs[a].predict([np.append([1], x_i)]) for a in range(2)]
                max_action = np.argmax(predictions)

            if max_action == a_i:
                sampled_greedy_or_not = 1
            else:
                sampled_greedy_or_not = 0
            

            forced = True

            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]

                    if not np.all(action_counters > 0):
                        max_action = 1
                    else:
                        predictions = [regrs[a].predict([np.append([1], x_curr)]) for a in range(2)]
                        max_action = np.argmax(predictions)

                    # act based on sampled_greedy_or_not and determine if forced
                    if sampled_greedy_or_not == 1:
                        if max_action == a_curr:
                            p_vec[i_] = 1.
                        else:
                            forced = False
                    if sampled_greedy_or_not == 0:
                        if max_action != a_curr:
                            p_vec[i_] = 1.
                        else:
                            forced = False
            

            # if not forced, then take into account the probability of the coin flip
            if not forced:
                if not np.all(action_counters > 0):
                    prob *= 0.5
                else:
                    prob *= [self.epsilon/2, 1-self.epsilon/2][sampled_greedy_or_not]

            # normalize and then sample accordingly if proposing,
            # otherwise set to current index
            p_vec = p_vec/np.sum(p_vec)
            sample = i
            curr_selected[sample] = 1.
            prob *= p_vec[sample]

            prob *= 100 if self.T == 100 else 1. # this line is to make sure the weights 
            # don't get too small

            # update action-context-reward trackers
            ((sampled_a, sampled_x), sampled_r) = ((data[sample][0][0], data[sample][0][1]), data[sample][1])
            
            ys[sampled_a].append(sampled_r)
            Xs[sampled_a].append(np.append([1], sampled_x))
            regrs[sampled_a].fit(Xs[sampled_a],ys[sampled_a])

            action_counters[sampled_a] += 1
        return prob


    def cond_imitation(self, data, propose_or_weight):
        ''''The cond_imitation distribution samples without replacement, 
        conditioning on the coin flips made by elinucb'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        regr0 = Lasso(alpha=10.)
        regr1 = Lasso(alpha=10.)

        regrs = [regr0, regr1]

        action_counters = np.zeros(2)
        
        # tracking parameters for epsilon-greedy
        X0, y0 = [np.zeros(101)], [0]
        X1, y1 = [np.zeros(101)], [0]

        Xs = [X0, X1]
        ys = [y0, y1]

        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)
            coin_flip = self.coin_flips[i]


            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if not np.all(action_counters > 0):
                        max_action = 1
                    else:
                        predictions = [regrs[a].predict([np.append([1], x_curr)]) for a in range(2)]
                        max_action = np.argmax(predictions)

                    action_to_take = max_action if coin_flip == 1 else 1-max_action
                    if a_curr == action_to_take:
                        p_vec[i_] = 1.

            if np.all(p_vec == 0):
                for i_ in range(self.T):
                    if curr_selected[i_] == 0:
                        p_vec[i_] = 1.
            # normalize p-vector
            p_vec = p_vec/np.sum(p_vec)
            
            # if proposing, then sample according to p, 
            # if calculating the weight, then calculate the 
            # probability of having selected ith index, using p
            if propose_or_weight:
                sample = np.random.choice(self.T, p=p_vec)
            else:
                sample = i

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data if proposing
            # and update probability correspondingly, in either case
            ((sampled_a, sampled_x), sampled_r) = ((data[sample][0][0], data[sample][0][1]), data[sample][1])
            if propose_or_weight:
                shuffled_data.append(((sampled_a, sampled_x), sampled_r))

            ys[sampled_a].append(sampled_r)
            Xs[sampled_a].append(np.append([1], sampled_x))
            regrs[sampled_a].fit(Xs[sampled_a],ys[sampled_a])

            action_counters[sampled_a] += 1

        if propose_or_weight:
            return shuffled_data, 1. 
        else:
            return 1.


    def get_proposal(self, data, style):
        if style == 'u':
            return self.uniform(data, True)
        if style == 'i':
            return self.imitation(data, True)
        if style == 'r':
            return self.re_imitation(data, True)
        if style == 'c':
            return self.cond_imitation(data, True)
        


    def get_proposal_weight(self, proposal, starting, style):
        if style == 'u':
            return self.uniform(proposal, False)
        if style == 'i':
            return self.imitation(proposal, False)
        if style == 'r':
            return self.re_imitation(proposal, False)
        if style == 'c':
            return self.cond_imitation(proposal, False)
                
                