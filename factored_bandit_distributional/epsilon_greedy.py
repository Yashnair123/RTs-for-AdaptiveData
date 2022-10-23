#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with epsilon-greedy. 
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

class EpsilonGreedy:
    
    def __init__(self, T, epsilon, null, conditional, b_0=None):
        self.T = T
        self.epsilon = epsilon
        self.null = null
        self.conditional = conditional
        self.coin_flips = []
        self.b_0 = b_0 # this variable is used for confidence interval construction, only

    def get_dataset(self):
        data = []

        action_sums = np.zeros(3)
        action_counters = np.zeros(3)

        # collect epsilon-greedy data
        for i in range(self.T):
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                p /= np.sum(p)
            
            a = np.random.choice(3, p=p) # epsilon-greedy action selection

            if self.conditional:
                if argmax == 'undecided':
                    self.coin_flips.append(a)
                else:
                    other_not_selected = [i not in [a,argmax] for i in range(3)].index(True)
                    if a == argmax:
                        self.coin_flips.append(2) # 2 denotes greedy action
                    elif a > other_not_selected:
                        self.coin_flips.append(1) # 1 denotes lexicographically largest non-greedy
                    else:
                        self.coin_flips.append(0)
                    

            # if we're doing testing, then sample accordingly from null or alternative
            if self.b_0 == None:
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
            else:
                if a == 0:
                    r = np.random.normal()
                elif a == 1:
                    r = np.random.normal(self.b_0)
                else:
                    r = np.random.normal(2)
            
            X,Z = self.a_xz_mapping(a)
            data.append(((X,Z),r))

            action_counters[a] += 1
            action_sums[a] += r
        
        return data

    def a_xz_mapping(self, a):
        if a != 2:
            Z = 1
            X = a
        else:
            X = np.random.choice(2)
            Z = 0
        return X,Z

    def xz_a_mapping(self, X,Z):
        if Z == 0:
            return 2
        elif X == 0 and Z == 1:
            return 0
        elif X == 1 and Z == 1:
            return 1
           
    
    def get_data_weight(self, orig_data, b_ci=0):
        # if epsilon, then all datasets equally likely (dividing by reward probabilities)
        if self.epsilon == 1:
            return 1.

        prod = 1.
        
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        
        for i in range(self.T):
            X,Z = orig_data[i][0][0], orig_data[i][0][1]
            a = self.xz_a_mapping(X,Z)
            r = orig_data[i][1] + b_ci*int(a==1)

            # calculate greedy action
            curr_arg_max = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'


            # if doing simulation3, then need to follow the coin flips
            if self.conditional:
                coin_flip = self.coin_flips[i]
                if curr_arg_max == 'undecided':
                    action_to_take = coin_flip
                else:
                    if coin_flip == 2:
                        action_to_take = curr_arg_max
                    elif coin_flip == 1:
                        actions = [0,1,2]
                        actions.remove(curr_arg_max)
                        action_to_take = max(actions)
                    else:
                        actions = [0,1,2]
                        actions.remove(curr_arg_max)
                        action_to_take = min(actions)
                
                if a != action_to_take:
                    return 0.

            if curr_arg_max == 'undecided':
                prod *= 1./3. # uniform
            else:
                prod *= 1-self.epsilon + self.epsilon/3 if curr_arg_max == a else self.epsilon/3 #epsilon greedy
            
            action_counters[a] += 1
            action_sums[a] += r
        
        # return depends on if conditional or not
        if self.conditional:
            return 1.
        else:
            return prod


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
        prob = 1.
        
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            # populate p-vector with weights corresponding to epsilon-greedy policy
            # among those remaining indices
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if argmax == 'undecided':
                        p[i_] = 1.
                    else:
                        p[i_] = (1 - self.epsilon + self.epsilon/3.) \
                        if self.xz_a_mapping(data[i_][0][0], data[i_][0][1]) \
                            == argmax else self.epsilon/3.

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
            action_counters[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] += 1
            action_sums[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] \
                += (data[sample][1] + b_ci*int(self.xz_a_mapping(data[sample][0][0], data[sample][0][1])==1))
            # reward depends on b_ci, since data[sample][1] has subtracted it off

        if propose_or_weight:
            return shuffled_data, prob 
        else:
            return prob



    def simulation2(self, data, propose_or_weight, b_ci=0):
        ''''The simulation2 distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily and then samples 
        correspondingly from the remaining timesteps.'''

        prob = 1.
        
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            # obtain probabilities for three actions before looking at remaining actions
            prob_vec = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                prob_vec[argmax] += 1-self.epsilon

            actions_remaining = np.zeros(3)

            # populate p-vector with weights corresponding to epsilon-greedy policy
            # among those remaining indices
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    actions_remaining[self.xz_a_mapping(data[i_][0][0], data[i_][0][1])] = 1.
            
            # condition on the remaining actions and normalize
            prob_vec = prob_vec * actions_remaining
            prob_vec = prob_vec / np.sum(prob_vec)

            # sample the action
            if propose_or_weight:
                action_to_take = np.random.choice(3, p=prob_vec)
            else:
                # in this case, the action_to_take was the action selected
                action_to_take = self.xz_a_mapping(data[i][0][0], data[i][0][1])
            
            prob *= prob_vec[action_to_take]


            # select uniformly over that action_to_take
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if self.xz_a_mapping(data[i_][0][0], data[i_][0][1]) == action_to_take:
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
            action_counters[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] += 1
            action_sums[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] \
                += (data[sample][1] + b_ci*int(self.xz_a_mapping(data[sample][0][0], data[sample][0][1])==1))
            # reward depends on b_ci, since data[sample][1] has subtracted it off

        if propose_or_weight:
            return shuffled_data, prob 
        else:
            return prob


    def simulation3(self, data, propose_or_weight, b_ci=0):
        ''''The simulation3 distribution samples without replacement, 
        using the simulation3 distribution'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)
        
        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        
        for i in range(self.T):
            coin_flip = self.coin_flips[i]
            p = np.zeros(self.T)
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            if argmax == 'undecided':
                action_to_select = coin_flip
            else:
                if coin_flip == 2:
                    action_to_select = argmax
                elif coin_flip == 1:
                    actions = [0,1,2]
                    actions.remove(argmax)
                    action_to_take = max(actions)
                else:
                    actions = [0,1,2]
                    actions.remove(argmax)
                    action_to_take = min(actions)
            # populate p-vector with weights corresponding to epsilon-greedy policy
            # among those remaining indices
            for i_ in range(self.T):
                if curr_selected[i_] != 1:
                    if self.xz_a_mapping(data[i_][0][0], data[i_][0][1]) == action_to_select:
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
            action_counters[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] += 1
            action_sums[self.xz_a_mapping(data[sample][0][0], data[sample][0][1])] \
                += (data[sample][1] + b_ci*int(self.xz_a_mapping(data[sample][0][0], data[sample][0][1])==1))
            # reward depends on b_ci, since data[sample][1] has subtracted it off

        if propose_or_weight:
            return shuffled_data, 1. 
        else:
            return 1.


    def uniform_X(self, data, propose_or_weight):
        '''This sampling scheme samples X's uniformly, except other than the first
        two timesteps, whence it must select i, the timestep, as X'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = [((np.random.choice(2), data[i][0][1]), data[i][1]) for i in range(self.T)]
            return sampled_data, 1.
        else:
            return 1.


    def simulation_X(self, data, propose_or_weight, b_ci=0):
        '''This sampling scheme samples X's from the simulation 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        
        prob = 1.
        if propose_or_weight:
            sampled_data = []

        action_sums = np.zeros(3)
        action_counters = np.zeros(3)

        for i in range(self.T):
            # compute epsilon-greedy argmax
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            # set corresponding p-vector
            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                p /= np.sum(p)
            
            # transform p-vector into distribution over X's given Z
            z = data[i][0][1]
            new_p = np.zeros(2)

            # induced distribution on x:
            for x in range(2):
                new_p[x] = p[self.xz_a_mapping(x,z)]
            # normalize
            new_p = new_p/np.sum(new_p)

            # if proposing, then sample from new_p's distribution
            # otherwise set x to be the current x value and calculate
            # its weight using new_p
            if propose_or_weight:
                x = np.random.choice(2, p=new_p)
            else:
                x = data[i][0][0]
            a = self.xz_a_mapping(x,z)
            r = data[i][1] + b_ci*int(a==1)

            prob *= new_p[x]
            
            if propose_or_weight:
                sampled_data.append(((x,z),data[i][1])) # only append independent version to sampled data, so subtract offset
            
            
            action_counters[a] += 1
            action_sums[a] += r
    
        if propose_or_weight:
            return sampled_data, prob
        else:
            return prob


    def combined(self, data, propose_or_weight, b_ci=0):
        '''This sampling scheme samples X's from the combined 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        prod = 1.
        sampled_data = []
        curr_selected = np.zeros(self.T)

        action_sums = np.zeros(3)
        action_counters = np.zeros(3)

        for i in range(self.T):
            # calculate epsilon-greedy argmax
            argmax = np.argmax(action_sums/action_counters) if \
            np.all(action_counters > np.zeros(3)) else 'undecided'

            # set corresponding p-vector
            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                p /= np.sum(p)

            # p-vector induces distribution over Z's:
            p_z = np.zeros(2)
            p_z[1], p_z[0] = np.sum(p[:2]), p[2]
            
            # sample from remaining timesteps according to this vector
            prob = np.zeros(self.T)
            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    prob[i_] = p_z[data[i_][0][1]]

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
            z = data[sample][0][1]
            new_p = np.zeros(2)

            # induced distribution on x:
            for x in range(2):
                new_p[x] = p[self.xz_a_mapping(x,z)]
            
            # normalize
            new_p = new_p/np.sum(new_p)

            # sample x according to new_p if proposing, otherwise set to current
            if propose_or_weight:
                x = np.random.choice(2, p=new_p)
            else:
                x = data[i][0][0]
            selected_action = self.xz_a_mapping(x,z)
            prod *= new_p[x]

            # convert to factored bandit data and append
            r = data[sample][1] + b_ci*int(selected_action==1)
            sampled_data.append(((x, z), r-b_ci*int(selected_action==1))) # only append independent version to sampled data, so subtract offset
            
            # update epsilon-greedy action-reward tracker
            action_counters[selected_action] += 1
            action_sums[selected_action] += r

        if propose_or_weight:
            return sampled_data, prod
        else:
            return prod
    
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
        if style == 's2s':
            intermediary, prob = self.simulation2(data, True, b_ci)
            return self.simulation_X(intermediary, True, b_ci)
        if style == 's3s':
            intermediary, prob = self.simulation3(data, True, b_ci)
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
        if style == 's2s':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.simulation2(intermediary, False, b_ci)
            action_prob = self.simulation_X(proposal, False, b_ci)
            return permute_prob*action_prob
        if style == 's3s':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.simulation3(intermediary, False, b_ci)
            action_prob = self.simulation_X(proposal, False, b_ci)
            return permute_prob*action_prob