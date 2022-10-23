#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with UCB. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various proposal sampling processes for epsilon-greedy
       in the setting of a distributional in a 3-armed bandit
       viewed as a factored bandit
    4. proposal sampling weighting calculations for the above proposals
"""
import numpy as np
import copy


class UCB:
    
    def __init__(self, T, null, b_0=None):
        self.T = T
        self.null = null
        self.b_0 = b_0 # this variable is used for confidence interval construction, only

    def get_dataset(self):
        '''generate UCB data from 3 armed bandit'''
        data = []
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        for i in range(self.T):
            if i < 3:
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
                    if a != 2:
                        r = np.random.normal()
                    else:
                        r = np.random.normal(2)
                else:
                    if a == 0:
                        r = np.random.normal()
                    elif a == 1:
                        r = np.random.normal(3)
                    else:
                        r = np.random.normal(2)
            # otherwise, draw the reward according to b_0 (i.e., factor 1 = factor_0 + b_0)
            else:
                if a == 0:
                    r = np.random.normal()
                elif a == 1:
                    r = np.random.normal(self.b_0)
                else:
                    r = np.random.normal(2)
            
            X,Z = self.a_xz_mapping(a)
            data.append(((X,Z),r)) # true rewards are placed in only the true data for confidence interval

            action_counters[a] += 1
            action_sums[a] += r
        
        return data

    def a_xz_mapping(self, a):
        '''randomized mapping taking a to (X,Z) in 
        factored bandit'''
        if a != 2:
            Z = 1
            X = a
        else:
            X = np.random.choice(2)
            Z = 0
        return X,Z

    def xz_a_mapping(self, X,Z):
        '''mapping taking factored bandit (X,Z) to a'''
        if Z == 0:
            return 2
        elif X == 0 and Z == 1:
            return 0
        elif X == 1 and Z == 1:
            return 1
           
    
    def get_data_weight(self, orig_data, b_ci=0):
        if orig_data == 'flag':
            return 0.
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        for i in range(self.T):
            if i < 3:
                argmax = i
            else:
                us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                argmax = np.argmax(us)

            X,Z = orig_data[i][0][0], orig_data[i][0][1]
            a = self.xz_a_mapping(X,Z)
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
            shuffled_data = [((data[perm[i]][0][0], data[perm[i]][0][1]), data[perm[i]][1]) for i in range(self.T)]

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
                if data[i][0][1] == 0:
                    zero_inds.append(i)
                else:
                    one_inds.append(i)

            # permute them
            perm0 = list(np.random.choice(len(zero_inds), len(zero_inds), replace=False))
            perm1 = list(np.random.choice(len(one_inds), len(one_inds), replace=False))

            zero_inds = [zero_inds[i] for i in perm0]
            one_inds = [one_inds[i] for i in perm1]
            

            shuffled_data = copy.deepcopy(data)
            zero_ind = 0
            one_ind = 0
            for i in range(len(data)):
                if data[i][0][1] == 0:
                    shuffled_data[i] = data[zero_inds[zero_ind]]
                    zero_ind += 1
                else:
                    shuffled_data[i] = data[one_inds[one_ind]]
                    one_ind += 1

            return shuffled_data, 1.
        else:
            # uniform sampling always has weight 1
            return 1.


    def simulation1(self, data, propose_or_weight, b_ci=0):
        ''''The simulation1 distribution samples without replacement, 
        proportional to the policy probabilities'''
        
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
                     
        # note that there is no reason to calculate probabilities as simulation1 is
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
            sampled_data = [((i if i <=1 else np.random.choice(2), data[i][0][1]), data[i][1]) for i in range(self.T)]
            return sampled_data, 1.
        else:
            return 1.


    def simulation_X(self, data, propose_or_weight, b_ci=0):
        '''This sampling scheme samples X's from he simulation 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = []

            action_sums = np.zeros(3)
            action_counters = np.zeros(3)
            
            for i in range(self.T):
                if i < 3:
                    argmax = i
                else:
                    # make sure all actions selected by this point, otherwise return 0.
                    if not np.all(action_counters):
                        return 'flag', 1.
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                    argmax = np.argmax(us)

                p = np.zeros(3)
                p[argmax] = 1.
                # UCB would select the argmax action, but now
                # we turn it into which actions should be selected in 
                # the factored bandit
                z = data[i][0][1]
                new_p = np.zeros(2)
                
                # induced distribution on x:
                for x in range(2):
                    new_p[x] = p[self.xz_a_mapping(x,z)]
                
                # if there are no action to select for this
                # value of z, then just sample uniformly
                if np.all(new_p==0):
                    new_p=np.ones(2)
                # normalize
                new_p = new_p/np.sum(new_p)

                x = np.random.choice(2, p=new_p)
                a = self.xz_a_mapping(x,z)
                r = data[i][1] + b_ci*int(a==1)

                sampled_data.append(((x,z),data[i][1])) # only append independent version to sampled data, so subtract offset
                

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

            action_sums = np.zeros(3)
            action_counters = np.zeros(3)

            for i in range(self.T):
                if i < 3:
                    argmax = i
                else:
                    # calculate UCB argmax
                    us = action_sums/action_counters + np.sqrt(2. * np.log(np.sum(action_counters))/action_counters)
                    argmax = np.argmax(us)
                
                # convert to factored bandit
                max_x,max_z = self.a_xz_mapping(argmax)
                
                prob = np.zeros(self.T)

                # sample uniformly over actions with the same z component as the argmax
                for i_ in range(self.T):
                    if curr_selected[i_] == 0 and data[i_][0][1] == max_z:
                        prob[i_] = 1.

                # if none remaining, then uniform over remainder
                if np.all(prob==0):
                    for i_ in range(self.T):
                        if curr_selected[i_] == 0:
                            prob[i_] = 1
                    max_x = np.random.choice(2)
                
                prob = prob/np.sum(prob)
                sample = np.random.choice(self.T, p=prob)

                # mark that we selected this index
                curr_selected[sample] = 1

                # convert to factored bandit data and append
                z, r = data[sample][0][1], data[sample][1]
                selected_action = self.xz_a_mapping(max_x,z)
                r += b_ci*int(selected_action==1) # add offset to get true reward
                sampled_data.append(((max_x, z), r-b_ci*int(selected_action==1)))  # only append independent version to sampled data, so subtract offset
                
                # update UCB action-reward tracker
                action_counters[selected_action] += 1
                action_sums[selected_action] += r

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
        if style == 's':
            return self.simulation_X(data, True, b_ci)
        if style == 'us':
            intermediary, prob = self.uniform_permute(data, True)
            return self.simulation_X(intermediary, True, b_ci)
        if style == 'rus':
            intermediary, prob = self.restricted_uniform_permute(data, True)
            return self.simulation_X(intermediary, True, b_ci)
        if style == 'uu':
            intermediary, prob = self.uniform_permute(data, True)
            return self.uniform_X(intermediary, True)
        if style == 'c':
            return self.combined(data, True, b_ci)
        if style == 's1s':
            intermediary, prob = self.simulation1(data, True, b_ci)
            return self.simulation_X(intermediary, True, b_ci)
        


    def get_proposal_weight(self, proposal, starting, style, b_ci=0):
        if style == 'u':
            return self.uniform_X(proposal, False)
        if style == 's':
            return self.simulation_X(proposal, False, b_ci)
        if style == 'us':
            return self.simulation_X(proposal, False, b_ci)
        if style == 'rus':
            return self.simulation_X(proposal, False, b_ci)
        if style == 'uu':
            return self.uniform_X(proposal, False)
        if style == 'c':
            return self.combined(proposal, False, b_ci)
        if style == 's1s':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.simulation1(intermediary, False, b_ci)
            action_prob = self.simulation_X(proposal, False, b_ci)
            return permute_prob*action_prob