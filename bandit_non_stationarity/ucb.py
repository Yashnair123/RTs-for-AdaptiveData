#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with UCB. 
This file contains:
    1. data generation process for UCB
    2. data weight calculation for UCB
    3. various proposal sampling processes for UCB
       in the setting of non-stationarity test in a 2-armed bandit
    4. proposal sampling weighting calculations for the above proposals
"""
import numpy as np
import copy

class UCB:
    
    def __init__(self, T, null, conditional):
        self.T = T
        self.null = null
        self.epsilon = 0.
        self.conditional = conditional
        self.coin_flips = []
           
    def get_dataset(self):
        data = []

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        # first T-1 time steps
        for i in range(self.T-1):
            if i < 2:
                curr_argmax = i
            else:
                # compute the upper confidence bound
                if np.all(action_counters > 0):
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                else:
                    us = np.array([1,1])
                curr_argmax = np.argmax(us)

            U = np.random.uniform()
            a = curr_argmax if U < 1-self.epsilon else np.random.choice(2) # select argmax action

            if self.conditional:
                if a == curr_argmax:
                    self.coin_flips.append(1)
                else:
                    self.coin_flips.append(0)
            r = np.random.normal(2*a-1) # receive null distribution reward

            action_counters[a] += 1
            action_sums[a] += r
            data.append((a,r))

        # last timestep separately
        if np.all(action_counters > 0):
            us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
        else:
            us = np.array([1,1])
        curr_argmax = np.argmax(us)
        
        U = np.random.uniform()
        a = curr_argmax if U < 1-self.epsilon else np.random.choice(2)

        if self.conditional:
            if a == curr_argmax:
                self.coin_flips.append(1)
            else:
                self.coin_flips.append(0)

        if self.null:
            r = np.random.normal(2*a-1) # receive null distribution reward
        else:
            r = np.random.normal(4*(2*a-1)) # receive alternative distribution reward

        data.append((a,r))

        return data
    
    
    def get_data_weight(self, data):
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)

        prod = 1.
        
        for i in range(self.T):
            if i < 2:
                curr_argmax = i
            else:
                # compute the upper confidence bound
                if np.all(action_counters > 0):
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                else:
                    us = np.array([1,1])
                curr_argmax = np.argmax(us)

            a = data[i][0]
            r = data[i][1]

            if self.conditional:
                if self.coin_flips[i] == 1 and a != curr_argmax:
                    return 0.
                if self.coin_flips[i] == 0 and a == curr_argmax:
                    return 0.
            
            if a == curr_argmax:
                prod *= 1-self.epsilon/2
            else:
                prod *= self.epsilon/2
            
            # # if the action is not the UCB argmax, then the data has weight 0
            # if curr_argmax != a:
            #     return 0
            
            action_counters[a] += 1
            action_sums[a] += r
        
        if self.conditional:
            return 1.
        else:
            return prod



    def get_shared_data_weight(self, data, b_vals):
        action_sums = np.zeros(2)
        action_counters = np.zeros(2)

        shared_prod = 1.
        prods = []
        
        for i in range(self.T):
            a = data[i][0]
            r = data[i][1]

            if r in b_vals and i < self.T-1: # only matters if not final timestep
                i_start = i
                for b in b_vals:
                    b_prod = shared_prod
                    b_action_counters = copy.deepcopy(action_counters)
                    b_action_sums = copy.deepcopy(action_sums)
                    for j in range(i_start, self.T):
                        a = data[j][0]
                        r = data[j][1] if j != i_start else b # make sure to set to b if we're at the index that's changed

                        if j < 2:
                            curr_argmax = j
                        else:
                            # compute the upper confidence bound
                            if np.all(b_action_counters > 0):
                                us = b_action_sums/b_action_counters + np.sqrt(2. * np.log(np.sum(b_action_counters))/b_action_counters)
                            else:
                                us = np.array([1,1])
                            curr_argmax = np.argmax(us)

                        if self.conditional:
                            if self.coin_flips[j] == 1 and a != curr_argmax:
                                b_prod = 0.
                                break
                            if self.coin_flips[j] == 0 and a == curr_argmax:
                                b_prod = 0.
                                break

                        if a == curr_argmax:
                            b_prod *= 1-self.epsilon/2
                        else:
                            b_prod *= self.epsilon/2
                        
                        # # if the action is not the UCB argmax, then the data has weight 0
                        # if curr_argmax != a:
                        #     return 0
                        
                        b_action_counters[a] += 1
                        b_action_sums[a] += r
                    prods.append(b_prod)
                break
            else:
                if i < 2:
                    curr_argmax = i
                else:
                    # compute the upper confidence bound
                    if np.all(action_counters > 0):
                        us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                    else:
                        us = np.array([1,1])
                    curr_argmax = np.argmax(us)


                if self.conditional:
                    if self.coin_flips[i] == 1 and a != curr_argmax:
                        return [0.]*len(b_vals)
                    if self.coin_flips[i] == 0 and a == curr_argmax:
                        return [0.]*len(b_vals)
                
                if a == curr_argmax:
                    shared_prod *= 1-self.epsilon/2
                else:
                    shared_prod *= self.epsilon/2
                
                # # if the action is not the UCB argmax, then the data has weight 0
                # if curr_argmax != a:
                #     return 0
                
                action_counters[a] += 1
                action_sums[a] += r

                if i == self.T-1:
                    prods = [shared_prod]*len(b_vals)
        
        return prods


    def uniform(self, data, propose_or_weight):
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:# sample a permutation of the data uniformly at random
            perm = list(np.random.choice(self.T, self.T, replace=False))

            # permute the data
            shuffled_data = [(data[perm[i]][0], data[perm[i]][1]) for i in range(self.T)]

            # return the data and its weight
            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.

    
    def simulation1(self, data, propose_or_weight):
        ''''The simulation1 distribution samples without replacement, 
        proportional to the policy probabilities'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        prod = 1.
        if propose_or_weight:
            shuffled_data = []

        # pointer list to keep track of what's selected
        curr_selected = np.zeros(self.T)
        
        for i in range(self.T):   
            # p-vector will contain timestep selection probabilities  
            p = np.zeros(self.T)
            if i < 2:
                curr_argmax = i
            else:
                if np.all(action_counters > 0):
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                else:
                    us = np.array([1,1])
                curr_argmax = np.argmax(us)
            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    p[i_] = 1-self.epsilon/2 if data[i_][0] == curr_argmax else self.epsilon/2
        
            # if no feasible possibilities, sample uniformly
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

            prod *= p[sample]

            # mark sampled index in curr_selected so that we no longer sample it
            curr_selected[sample] = 1

            # add timestep to shuffled data if proposing
            # and update probability correspondingly, in either case
            if propose_or_weight:
                shuffled_data.append((data[sample][0], data[sample][1]))

            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]
        
        # note that there is no reason to calculate probabilities as simulation1 is
        # truncated uniform sampling
        if propose_or_weight:
            return shuffled_data, prod
        else:
            return prod


    def simulation3(self, data, propose_or_weight):
        ''''The simulation3 distribution samples without replacement, 
        proportional to the policy probabilities, conditionally'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        action_sums = np.zeros(2)
        action_counters = np.zeros(2)
        
        if propose_or_weight:
            shuffled_data = []

        # pointer list to keep track of what's selected
        curr_selected = np.zeros(self.T)
        
        for i in range(self.T):   
            coin_flip = self.coin_flips[i]
            # p-vector will contain timestep selection probabilities  
            p = np.zeros(self.T)
            if i < 2:
                curr_argmax = i
            else:
                if np.all(action_counters > 0):
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                else:
                    us = np.array([1,1])
                curr_argmax = np.argmax(us)
            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    if coin_flip == 1:
                        p[i_] = 1 if data[i_][0] == curr_argmax else 0
                    else:
                        p[i_] = 0 if data[i_][0] == curr_argmax else 1
        
            # if no feasible possibilities, sample uniformly
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

            action_counters[data[sample][0]] += 1
            action_sums[data[sample][0]] += data[sample][1]
        
        # note that there is no reason to calculate probabilities as simulation1 is
        # truncated uniform sampling
        if propose_or_weight:
            return shuffled_data, 1.
        else:
            return 1.

    

    def get_proposal(self, data, style):
        if style == 'u':
            return self.uniform(data, True)
        if style == 's1':
            return self.simulation1(data, True)
        if style == 's3':
            return self.simulation3(data, True)
        


    def get_proposal_weight(self, proposal, starting, style):
        if style == 'u':
            return self.uniform(proposal, False)
        if style == 's1':
            return self.simulation1(proposal, False)
        if style == 's3':
            return self.simulation3(proposal, False)