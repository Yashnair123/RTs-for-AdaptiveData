#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with epsilon-greedy. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various resampling procedures for epsilon-greedy
       in the setting of non-stationarity test in a 2-armed bandit
    4. resampling weighting calculations for the above resampling procedures
"""
import numpy as np
import copy

class EpsilonGreedy:
    
    def __init__(self, T, epsilon, null, conditional):
        self.T = T
        self.epsilon = epsilon
        self.null = null
        self.coin_flips = []
        self.conditional = conditional


    def get_dataset(self):
        data = []
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)

        # generate epsilon-greedy data for the first T-1 timesteps:
        for _ in range(self.T-1):
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            # pick a epsilon-greedily if there is a greedy action (i.e., argmax is not 'undecided')
            # otherwise choose a uniformly at random
            a = np.random.choice(2, p=[self.epsilon/2 + (1-argmax)*(1-self.epsilon), \
                self.epsilon/2 + argmax*(1-self.epsilon)]) if argmax != 'undecided' else np.random.choice(2)

            # if doing cond_imitation, keep track of coin flips (if argmax is 'undecided', the argmax is default 1 but coin flip is 
            # with 1/2 probability)
            if self.conditional:
                if argmax != 'undecided':
                    self.coin_flips.append(1 if a == argmax else 0)
                else:
                    self.coin_flips.append(a)

            # null reward distribution is N(2a-1, 1)
            r = np.random.normal(2*a-1)
            
            data.append((a,r))

            action_counters[a] += 1
            action_sums[a] += r
        
        # sample action and reward at last timestep: sampling distribution depends on self.null
        argmax = np.argmax(action_sums/action_counters) if \
            np.all(action_counters > np.zeros(2)) else 'undecided'
        
        a = np.random.choice(2, p=[self.epsilon/2 + (1-argmax)*(1-self.epsilon), \
            self.epsilon/2 + argmax*(1-self.epsilon)]) if argmax != 'undecided' else np.random.choice(2)

        # if doing cond_imitation, keep track of the last coin flip as well
        if self.conditional:
            if argmax != 'undecided':
                self.coin_flips.append(1 if a == argmax else 0)
            else:
                self.coin_flips.append(a)

        if self.null:
            r = np.random.normal(2*a-1)
        else:
            # under the alternative, reward distribution is N(4(2a-1), 1)
            r = np.random.normal(4*(2*a-1))
        data.append((a,r))

        return data
           
    
    def get_data_weight(self, data):
        # for fast computation, if epsilon = 1, then the weight of all datasets 
        # will be the same, so may return 1
        if self.epsilon == 1.:
            return 1.
        
        # otherwise, compute the weight, which is simply the product of action
        # selection probabilities
        prod = 1.
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        for i in range(self.T):
            a = data[i][0]
            r = data[i][1]

            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            # if doing cond_imitation, then need to follow the coin flips
            if self.conditional:
                coin_flip = self.coin_flips[i]
                if argmax != 'undecided':
                    action_to_take = argmax if coin_flip == 1 else 1-argmax
                else:
                    action_to_take = 1 if coin_flip == 1 else 0
                
                if a != action_to_take:
                    return 0.

            if argmax == 'undecided':
                prod *= 0.5
            else:
                prod *= 1-(self.epsilon/2) if argmax == a else self.epsilon/2
            
            action_counters[a] += 1
            action_sums[a] += r
        
        # return slightly differently if doing cond_imitation
        if self.conditional:
            return 1.
        else:
            return prod


    def get_shared_data_weight(self, data, b_vals):
        # for shared computation, in construction of conformal band
        if self.epsilon == 1.:
            return [1.]*len(b_vals)
        
        # otherwise, compute the weight, which is simply the product of action
        # selection probabilities, but share computation amongst the different b's
        
        shared_prod = 1.
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)

        prods = []

        for i in range(self.T):
            a = data[i][0]
            r = data[i][1]
            
            if r in b_vals and i < self.T-1: # only matters if not final timestep
                i_start = i
                # compute remainder of product for each b value
                for b in b_vals:
                    b_prod = shared_prod
                    b_action_counters = copy.deepcopy(action_counters)
                    b_action_sums = copy.deepcopy(action_sums)
                    for j in range(i_start, self.T):
                        a = data[j][0]
                        r = data[j][1] if j != i_start else b # make sure to set to b if we're at the index that's changed
                        argmax = np.argmax(b_action_sums/b_action_counters) if \
                            np.all(b_action_counters > np.zeros(2)) else 'undecided'

                        # if doing cond_imitation, then need to follow the coin flips
                        if self.conditional:
                            coin_flip = self.coin_flips[j]
                            if argmax != 'undecided':
                                action_to_take = argmax if coin_flip == 1 else 1-argmax
                            else:
                                action_to_take = 1 if coin_flip == 1 else 0
                            
                            if a != action_to_take:
                                b_prod *= 0.
                                break # can break out of this for loop once it's zero
                        else:
                            if argmax == 'undecided':
                                b_prod *= 0.5
                            else:
                                b_prod *= 1-(self.epsilon/2) if argmax == a else self.epsilon/2
                        
                        b_action_counters[a] += 1
                        b_action_sums[a] += r

                    prods.append(b_prod)

                break #break out of the main for loop, because computation is done
            else: # in this case, haven't seen b_value yet, or have but i == T-1
                argmax = np.argmax(action_sums/action_counters) if \
                    np.all(action_counters > np.zeros(2)) else 'undecided'

                # if doing cond_imitation, then need to follow the coin flips
                if self.conditional:
                    coin_flip = self.coin_flips[i]
                    if argmax != 'undecided':
                        action_to_take = argmax if coin_flip == 1 else 1-argmax
                    else:
                        action_to_take = 1 if coin_flip == 1 else 0
                    
                    if a != action_to_take:
                        shared_prod *= 0.
                        return [0.] * len(b_vals) # in this case, all will be 0
                else:
                    if argmax == 'undecided':
                        shared_prod *= 0.5
                    else:
                        shared_prod *= 1-(self.epsilon/2) if argmax == a else self.epsilon/2
                
                action_counters[a] += 1
                action_sums[a] += r

                if i == self.T-1:
                    prods = [shared_prod]*len(b_vals) # in this case, all are the same since change is just at last timestep
        
        return prods


    def uniform(self, data, propose_or_weight):
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            # sample a permutation of the data uniformly at random
            perm = list(np.random.choice(self.T, self.T, replace=False))

            # permute the data
            shuffled_data = [(data[perm[i]][0], data[perm[i]][1]) for i in range(self.T)]

            # return the data and its weight
            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.


    def imitation(self, data, propose_or_weight):
        ''''The imitation distribution samples without replacement, 
        proportional to the action-selection probabilities'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        prob = 1.

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            # populate p-vector with weights corresponding to epsilon-greedy policy
            # among those remaining indices
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if argmax == 'undecided':
                        p[i_] = 1.
                    else:
                        p[i_] = (1 - self.epsilon/2.) \
                        if data[i_][0] == argmax else self.epsilon/2.

            # otherwise select unif at random
            if np.all(p==0):
                for i_ in range(self.T):
                    if curr_selected[i_] == 0:
                        p[i_] = 1.

            # normalize p-vector
            p = p/np.sum(p)
            
            # if proposing, then sample according to p, 
            # if calculating the weight, then calculate the 
            # probability of having selected ith index, using p
            if propose_or_weight:
                sample = np.random.choice(self.T, p=p)
            else:
                sample = i

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data if proposing
            # and update probability correspondingly, in either case
            if propose_or_weight:
                shuffled_data.append((data[sample][0], data[sample][1]))
            prob *= p[sample]

            # update reward-action trackers
            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]

        if propose_or_weight:
            return shuffled_data, prob 
        else:
            return prob


    def re_imitation(self, data, propose_or_weight):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily and then samples 
        correspondingly from the remaining timesteps.'''

        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        if propose_or_weight:
            return self.re_imitation_propose(data)
        else:
            return self.re_imitation_weight(data)


    def re_imitation_propose(self, data):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function just samples from the distribution'''

        prob = 1.

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            # sample action epsilon-greedily
            a = np.random.choice(2, p=[self.epsilon/2 + (1-argmax)*(1-self.epsilon), \
                self.epsilon/2 + argmax*(1-self.epsilon)]) if argmax != 'undecided' else np.random.choice(2)

            # populate p-vector so that we only selected indices with action a 
            # among those remaining
            
            forced = True
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if data[i_][0] == a:
                        p[i_] = 1.
                    if data[i_][0] != a:
                        forced = False
            
            # if there are no remaining indices, then we must select the other action
            # flip, and the selectable indices are precisely those not already selected
            # otherwise, multiply by the probability of selecting action a
            if np.all(p == 0) or forced:
                for i_ in range(self.T):
                    if curr_selected[i_] != 1:
                        p[i_] = 1.
            else:
                if argmax == 'undecided':
                    prob *= 0.5
                else:
                    prob *= 1-self.epsilon/2 if a == argmax else self.epsilon/2

            # normalize p-vector
            p = p/np.sum(p)
            
            # sample according to p, 
            sample = np.random.choice(self.T, p=p)

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data
            # and update probability correspondingly, in either case
            shuffled_data.append((data[sample][0], data[sample][1]))
            prob *= p[sample]

            # update reward-action trackers
            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]

        return shuffled_data, prob


    def re_imitation_weight(self, data):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function calculates weights'''

        prob = 1.

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
    

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            # sample action epsilon-greedy made
            a = data[i][0]

            forced = True
            # check to see if we were forced to select action a
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if data[i_][0] != a:
                        forced = False
            
            if not forced:
                if argmax == 'undecided':
                    prob *= 0.5
                else:
                    prob *= 1-self.epsilon/2 if a == argmax else self.epsilon/2 

            # populate p-vector so that we only selected indices with action a 
            # among those remaining
            
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if data[i_][0] == a:
                        p[i_] = 1.

            # normalize p-vector
            p = p/np.sum(p)
            
            # the true data sampled index i
            sample = i

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # update probability correspondingly, in either case
            prob *= p[sample]

            # update reward-action trackers
            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]

        return prob


    def cond_imitation(self, data, propose_or_weight):
        ''''The cond_imitation distribution samples without replacement, 
        using the cond_imitation distribution'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            coin_flip = self.coin_flips[i]
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(2)) else 'undecided'

            if argmax != 'undecided':
                action_to_select = argmax if coin_flip == 1 else 1-argmax
            else:
                action_to_select = 1 if coin_flip == 1 else 0
            # populate p-vector with weights corresponding to epsilon-greedy policy
            # among those remaining indices
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if data[i_][0] == action_to_select:
                        p[i_] = 1.

            # if none left, then do uniformly
            if np.all(p == 0):
                for i_ in range(self.T):
                    if curr_selected[i_] == 0:
                        p[i_] = 1.
            # normalize p-vector
            p = p/np.sum(p)
            
            # if proposing, then sample according to p, 
            # if calculating the weight, then calculate the 
            # probability of having selected ith index, using p
            if propose_or_weight:
                sample = np.random.choice(self.T, p=p)
            else:
                sample = i

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data if proposing
            # and update probability correspondingly, in either case
            if propose_or_weight:
                shuffled_data.append((data[sample][0], data[sample][1]))

            # update reward-action trackers
            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]

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
