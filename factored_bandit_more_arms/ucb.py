#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with UCB in 
a bandit with many (5) arms. 
This file contains:
    1. data generation process for UCB
    2. data weight calculation for UCB
    3. various resampling procedures for UCB
       in the setting of a distributional in a 3-armed bandit
       viewed as a factored bandit
    4. resampling weighting calculations for the above resampling procedures
"""
import numpy as np
import copy


class UCB:
    
    def __init__(self, T, null, b_0=None):
        self.T = T
        self.null = null
        self.b_0 = b_0 # this variable is used for confidence interval construction, only

    def get_dataset(self):
        '''generate UCB data from 5 armed bandit'''
        data = []
        action_sums = np.zeros(5)
        action_counters = np.zeros(5)
        for i in range(self.T):
            if i < 5:
                argmax = i
            else:
                # calculate upper confidence bound
                us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                argmax = np.argmax(us)

            # select UCB argmax as action
            a = argmax

            # if we're doing testing, then sample accordingly from null or alternative
            if self.b_0 == None:
                # convert to factored bandit data, and sample reward
                # depending if under null or alternative
                if self.null:
                    # actions 1 and 3 give same
                    # and actions 2 and 4 give same
                    if a == 1 or a == 3:
                        r = np.random.normal(4)
                    elif a == 2 or a == 4:
                        r = np.random.normal(2)
                    else:
                        r = np.random.normal()
                else:
                    r = np.random.normal(2*a)
            # otherwise, draw the reward according to b_0 (i.e., factor 1 = factor_0 + b_0)
            else:
                if a == 0:
                    r = np.random.normal()
                elif a == 1:
                    r = np.random.normal(self.b_0)
                else:
                    r = np.random.normal(2)
            
            data.append((a,r)) # true rewards are placed in only the true data for confidence interval

            action_counters[a] += 1
            action_sums[a] += r
        
        return data

           
    
    def get_data_weight(self, orig_data, b_ci=0):
        if orig_data == 'flag':
            return 0.
        action_sums = np.zeros(5)
        action_counters = np.zeros(5)
        for i in range(self.T):
            if i < 5:
                argmax = i
            else:
                us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                argmax = np.argmax(us)

            a = orig_data[i][0]
            r = orig_data[i][1] + b_ci*int(a==1)

            # if action is not argmax, UCB would never have selected it
            # so the weight would be 0
            if a != argmax:
                return 0.

            action_counters[a] += 1
            action_sums[a] += r
        
        # if haven't yet returned zero, UCB would have drawn this dataset (with equal probability)
        return 1.



    def uniform_permute(self, data, propose_or_weight):
        '''This sampling scheme samples uniformly over permutations'''
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


    def restricted_uniform_permute(self, data, propose_or_weight):
        '''This sampling scheme samples uniformly over permutations with the same list
        of Z-factors'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:# sample a permutation of the data uniformly at random
            odd_inds = []
            even_inds = []
            zero_inds = []

            # get the indices with zero as Z-factor and 1
            for i in range(len(data)):
                if data[i][0] == 1 or data[i][0] == 3:
                    odd_inds.append(i)
                elif data[i][0] == 2 or data[i][0] == 4:
                    even_inds.append(i)
                else:
                    zero_inds.append(i)

            # permute them
            permeven = list(np.random.choice(len(even_inds), len(even_inds), replace=False))
            permodd = list(np.random.choice(len(odd_inds), len(odd_inds), replace=False))
            permzero = list(np.random.choice(len(zero_inds), len(zero_inds), replace=False))


            even_inds_permute = [even_inds[i] for i in permeven]
            odd_inds_permute = [odd_inds[i] for i in permodd]
            zero_inds_permute = [zero_inds[i] for i in permzero]
            

            shuffled_data = copy.deepcopy(data)
            even_ind = 0
            odd_ind = 0
            zero_ind = 0
            for i in range(len(data)):
                if data[i][0] == 2 or data[i][0] == 4:
                    shuffled_data[i] = data[even_inds_permute[even_ind]]
                    even_ind += 1
                elif data[i][0] == 1 or data[i][0] == 3:
                    shuffled_data[i] = data[odd_inds_permute[odd_ind]]
                    odd_ind += 1
                else:
                    shuffled_data[i] = data[zero_inds_permute[zero_ind]]
                    zero_ind += 1

            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.


    def imitation(self, data, propose_or_weight, b_ci=0):
        ''''The imitation distribution samples without replacement, 
        proportional to the action-selection probabilities'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        
        prod = 1.
        if propose_or_weight:
            shuffled_data = []

        # pointer list to keep track of what's selected
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p = np.zeros(self.T)

            if i < 3:
                argmax = i
            else:
                # make sure all actions selected by this point, otherwise flag it as unusable.
                if not np.all(action_counters):
                    return 'flag', 1.
                us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                argmax = np.argmax(us)

            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    p[i_] = 1. if self.xz_a_mapping(data[i_][0][0],data[i_][0][1]) == argmax else 0.
        
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

            action_counters[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] += 1
            action_sums[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] \
                += (data[sample][1] + b_ci*int(self.xz_a_mapping(data[sample][0][0], data[sample][0][1])==1))
            # reward depends on b_ci, since data[sample][1] has subtracted it off
                     
        # note that there is no reason to calculate probabilities as imitation is
        # truncated uniform sampling
        if propose_or_weight:
            return shuffled_data, prod
        else:
            return prod


    def uniform_X(self, data, propose_or_weight):
        '''This sampling scheme samples X's uniformly, except other than the first
        two timesteps, whence it must select i, the timestep, as X'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = []
            for i in range(self.T):
                if i <= 4:
                    newA = data[i][0]
                else:
                    if data[i][0] == 2 or data[i][0] == 4:
                        newA = np.random.choice([2,4])
                    elif data[i][0] == 1 or data[i][0] == 3:
                        newA = np.random.choice([1,3])
                    else:
                        newA = 0
            sampled_data.append((newA, data[i][1]))
            return sampled_data, 1.
        else:
            return 1.


    def imitation_X(self, data, propose_or_weight, b_ci=0):
        '''This sampling scheme samples X's from he simulation 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = []

            action_sums = np.zeros(5)
            action_counters = np.zeros(5)
            
            for i in range(self.T):
                if i < 5:
                    argmax = i
                else:
                    # make sure all actions selected by this point, otherwise return 0.
                    if not np.all(action_counters):
                        return 'flag', 1.
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                    argmax = np.argmax(us)

                p = np.zeros(5)
                p[argmax] = 1.
                # UCB would select the argmax action, but now
                # we turn it into which actions should be selected in 
                # the factored bandit
                actual = data[i][0]

                if actual == 2 or actual == 4:
                    if argmax == 2 or argmax == 4:
                        a = argmax
                    else:
                        a = np.random.choice([2,4])
                elif actual == 1 or actual == 3:
                    if argmax == 1 or argmax == 3:
                        a = argmax
                    else:
                        a = np.random.choice([1,3])
                else:
                    a = 0
                
                r = data[i][1] + b_ci*int(a==1)

                sampled_data.append((a,data[i][1])) # only append independent version to sampled data, so subtract offset
                

                # update UCB action trackers
                action_counters[a] += 1
                action_sums[a] += r

            # note that the probability is always constant because any non-zero
            # weight (under the null data distribution) sampled dataset will
            # always have the same weight under the proposal, since only 
            # depends on rewards and Z-factor, which is sample-invariant
            return sampled_data, 1.
        else:
            return 1.
    

    def combined(self, data, propose_or_weight, b_ci=0):
        if propose_or_weight:
            sampled_data = []
            curr_selected = np.zeros(self.T)

            action_sums = np.zeros(5)
            action_counters = np.zeros(5)

            for i in range(self.T):
                if i < 5:
                    argmax = i
                else:
                    # calculate UCB argmax
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                    argmax = np.argmax(us)
                
                
                prob = np.zeros(self.T)

                # sample uniformly over actions with the same z component as the argmax
                for i_ in range(self.T):
                    if curr_selected[i_] == 0:
                        if argmax == 2 or argmax == 4:
                            prob[i_] = 1. if data[i_][0] == 2 or data[i_][0] == 4 else 0.
                        elif argmax == 1 or argmax == 3:
                            prob[i_] = 1. if data[i_][0] == 1 or data[i_][0] == 3 else 0.
                        else:
                            prob[i_] = 1. if data[i_][0] == 0 else 0.

                # if none remaining, then uniform over remainder
                if np.all(prob==0):
                    for i_ in range(self.T):
                        if curr_selected[i_] == 0:
                            prob[i_] = 1
                
                prob = prob/np.sum(prob)
                sample = np.random.choice(self.T, p=prob)

                # mark that we selected this index
                curr_selected[sample] = 1

                # convert to factored bandit data and append
                selected_action = data[sample][0]
                if argmax == 2 or argmax == 4:
                    action_to_take = argmax if selected_action == 2 or selected_action == 4 else np.random.choice([0,1,3])
                elif argmax == 1 or argmax == 3:
                    action_to_take = argmax if selected_action == 1 or selected_action == 3 else np.random.choice([0,2,4])
                else:
                    action_to_take = argmax if selected_action == 0 else np.random.choice([1,2,3,4])
                r = data[sample][1]
                sampled_data.append((action_to_take, r))  # only append independent version to sampled data, so subtract offset
                
                # update UCB action-reward tracker
                action_counters[action_to_take] += 1
                action_sums[action_to_take] += r

            # note that the probability is always constant because any non-zero
            # weight (under the null data distribution) sampled dataset will
            # always have the same weight under the proposal, since only 
            # depends on rewards and Z-factor, which is sample-invariant
            return sampled_data, 1.
        else:
            return 1.


    def get_proposal(self, data, style, b_ci=0):
        if style == 'u':
            return self.uniform_X(data, True)
        if style == 'i_X':
            return self.imitation_X(data, True, b_ci)
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
        if style == 'ii_X':
            intermediary, prob = self.imitation(data, True, b_ci)
            return self.imitation_X(intermediary, True, b_ci)
        


    def get_proposal_weight(self, proposal, starting, style, b_ci=0):
        if style == 'u':
            return self.uniform_X(proposal, False)
        if style == 'i_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'ui_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'rui_X':
            return self.imitation_X(proposal, False, b_ci)
        if style == 'uu_X':
            return self.uniform_X(proposal, False)
        if style == 'comb':
            return self.combined(proposal, False, b_ci)
        if style == 'ii_X':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.imitation(intermediary, False, b_ci)
            action_prob = self.imitation_X(proposal, False, b_ci)
            return permute_prob*action_prob
        

    def finite_sample(self, data):
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)

        for i in range(len(data)):
            a = self.xz_a_mapping(data[i][0][0], data[i][0][1])
            r = data[i][1]

            action_counters[a] += 1
            action_sums[a] += r

        return action_counters, action_sums