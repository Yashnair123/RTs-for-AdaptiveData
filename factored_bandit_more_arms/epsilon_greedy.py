#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for distributional testing with epsilon-greedy in a 
bandit with many (5) arms. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various resampling procedures for epsilon-greedy
       in the setting of a distributional in a 3-armed bandit
       viewed as a factored bandit
    4. resampling weighting calculations for the above resampling procedures
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

        action_sums = np.zeros(5)
        action_counters = np.zeros(5)
        

        # collect epsilon-greedy data
        for i in range(self.T):
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(5)) else 'undecided'

            p = np.zeros(5) + self.epsilon/5
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                p /= np.sum(p)
            
            a = np.random.choice(5, p=p) # epsilon-greedy action selection

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
            else:
                if a == 0:
                    r = np.random.normal()
                elif a == 1:
                    r = np.random.normal(self.b_0)
                else:
                    r = np.random.normal(2)
            
            data.append((a,r))

            action_counters[a] += 1
            action_sums[a] += r
        
        return data
           
    
    def get_data_weight(self, orig_data, b_ci=0):
        # if epsilon, then all datasets equally likely (dividing by reward probabilities)
        if self.epsilon == 1:
            return 1.

        prod = 1.
        
        action_sums = np.zeros(5)
        action_counters = np.zeros(5)
        
        for i in range(self.T):
            a,r = orig_data[i][0], orig_data[i][1]

            # calculate greedy action
            curr_arg_max = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(5)) else 'undecided'


            # if doing cond_imitation, then need to follow the coin flips
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
                prod *= 1./5. # uniform
            else:
                prod *= 1-self.epsilon + self.epsilon/5 if curr_arg_max == a else self.epsilon/5 #epsilon greedy
            
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
        prob = 1.
        
        action_sums = np.zeros(5)
        action_counters = np.zeros(5)
        
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



    def re_imitation(self, data, propose_or_weight, b_ci=0):
        ''''The re_imitation distribution samples, at each timestep, an action
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


    def cond_imitation(self, data, propose_or_weight, b_ci=0):
        ''''The cond_imitation distribution samples without replacement, 
        using the cond_imitation distribution'''
        
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
        '''This sampling scheme samples X's uniformly'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            sampled_data = []
            for i in range(self.T):
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
        '''This sampling scheme samples X's from the simulation 
        distribution over X's'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        
        prob = 1.
        if propose_or_weight:
            sampled_data = []

        action_sums = np.zeros(5)
        action_counters = np.zeros(5)

        for i in range(self.T):
            # compute epsilon-greedy argmax
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(5)) else 'undecided'

            # set corresponding p-vector
            p = np.zeros(5) + self.epsilon/5
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                p /= np.sum(p)
            
            # transform p-vector into distribution over X's given Z
            a = data[i][0]
            r = data[i][1]
            if a == 2 or a == 4:
                new_p = np.zeros(2)
                new_p[0], new_p[1] = p[2], p[4]
                new_p = new_p/np.sum(new_p)

                if propose_or_weight:
                    a = np.random.choice([2,4], p=new_p)
                
                prob *= new_p[[2,4].index(a)]
            elif a == 1 or a == 3:
                new_p = np.zeros(2)
                new_p[0], new_p[1] = p[1], p[3]
                new_p = new_p/np.sum(new_p)
                if propose_or_weight:
                    a = np.random.choice([1,3], p=new_p)
                prob *= new_p[[1,3].index(a)]
            else:
                if propose_or_weight:
                    a = 0
                prob *= 1.
            
            if propose_or_weight:
                sampled_data.append((a,r))
            
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

        action_sums = np.zeros(5)
        action_counters = np.zeros(5)

        for i in range(self.T):
            # calculate epsilon-greedy argmax
            argmax = np.argmax(action_sums/action_counters) if \
            np.all(action_counters > np.zeros(5)) else 'undecided'

            # set corresponding p-vector
            p = np.zeros(5) + self.epsilon/5
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                # p-vector is uniform
                p /= np.sum(p)

            # p-vector induces distribution over Z's:
            p_z = np.zeros(3)
            #index 0, 1, 2, correspond to evens, odds, and zero (summed)
            p_z[0] = p[2] + p[4]
            p_z[1] = p[1] + p[3]
            p_z[2] = p[0]
            
            # sample from remaining timesteps according to this vector
            prob = np.zeros(self.T)
            for i_ in range(self.T):
                if curr_selected[i_] == 0:
                    if data[i_][0] == 2 or data[i_][0] == 4:
                        prob[i_] = p_z[0]
                    elif data[i_][0] == 1 or data[i_][0] == 3:
                        prob[i_] = p_z[1]
                    else:
                        prob[i_] = p_z[2]

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
            a = data[sample][0]
            r = data[sample][1]


            if a == 2 or a == 4:
                new_p = np.zeros(2)
                new_p[0], new_p[1] = p[2], p[4]
                new_p = new_p/np.sum(new_p)

                if propose_or_weight:
                    a = np.random.choice([2,4], p=new_p)
                
                prob *= new_p[[2,4].index(a)]
            elif a == 1 or a == 3:
                new_p = np.zeros(2)
                new_p[0], new_p[1] = p[1], p[3]
                new_p = new_p/np.sum(new_p)
                if propose_or_weight:
                    a = np.random.choice([1,3], p=new_p)
                prob *= new_p[[1,3].index(a)]
            else:
                if propose_or_weight:
                    a = 0
                prob *= 1.

            sampled_data.append((a, r)) # only append independent version to sampled data, so subtract offset
            
            # update epsilon-greedy action-reward tracker
            action_counters[a] += 1
            action_sums[a] += r

        if propose_or_weight:
            return sampled_data, prod
        else:
            return prod
    
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
        if style == 'ri_X':
            intermediary, prob = self.re_imitation(data, True, b_ci)
            return self.imitation_X(intermediary, True, b_ci)
        if style == 'ci_X':
            intermediary, prob = self.cond_imitation(data, True, b_ci)
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
        if style == 'ri_X':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.re_imitation(intermediary, False, b_ci)
            action_prob = self.imitation_X(proposal, False, b_ci)
            return permute_prob*action_prob
        if style == 'ci_X':
            intermediary = []
            starting_reward_seq = [starting[i][1] for i in range(len(starting))]
            for i in range(len(proposal)):
                starting_index = starting_reward_seq.index(proposal[i][1])
                intermediary.append((starting[starting_index][0], proposal[i][1]))
            
            # calculate the probability of drawing the permutation
            permute_prob = self.cond_imitation(intermediary, False, b_ci)
            action_prob = self.imitation_X(proposal, False, b_ci)
            return permute_prob*action_prob
        

    def asymptotic_ci(self, data):
        action_sums = np.zeros(3)
        estimator_action_sums = np.zeros(3)
        action_counters = np.zeros(3)


        h0 = np.zeros(len(data))
        h1 = np.zeros(len(data))

        m_hat_0 = np.zeros(len(data))
        m_hat_1 = np.zeros(len(data))

        Gamma_hat_0 = np.zeros(len(data))
        Gamma_hat_1 = np.zeros(len(data))
        for i in range(len(data)):
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                p /= np.sum(p)

            a = self.xz_a_mapping(data[i][0][0], data[i][0][1])
            r = data[i][1]

            h0[i] = np.sqrt(p[0]/self.T)
            h1[i] = np.sqrt(p[1]/self.T)

            # estimate conditional means
            m_hat_0[i] = estimator_action_sums[0]/(i+1)
            m_hat_1[i] = estimator_action_sums[1]/(i+1)

            # calculate unbiased score rule (AIPW estimator)
            Gamma_hat_0[i] = (r/p[0]) + (1. - (1./p[0])) * m_hat_0[i] if a == 0 else m_hat_0[i]
            Gamma_hat_1[i] = (r/p[1]) + (1. - (1./p[1])) * m_hat_1[i] if a == 1 else m_hat_1[i]

            action_counters[a] += 1
            action_sums[a] += r
            estimator_action_sums[a] = estimator_action_sums[a] + r/p[a]
    
        normalized_h0 = h0/np.sum(h0)
        normalized_h1 = h1/np.sum(h1)

        Q_hat_0 = np.dot(normalized_h0, Gamma_hat_0)
        Q_hat_1 = np.dot(normalized_h1, Gamma_hat_1)
        unstudentized_test_stat =  Q_hat_1 - Q_hat_0

        V_hat_0 = np.dot(normalized_h0**2, (Gamma_hat_0 - np.mean(Gamma_hat_0))**2)
        V_hat_1 = np.dot(normalized_h1**2, (Gamma_hat_1 - np.mean(Gamma_hat_1))**2)
        denom = np.sqrt(V_hat_0 + V_hat_1)

        return unstudentized_test_stat, denom


    def finite_sample(self, data):
        action_sums = np.zeros(3)
        action_counters = np.zeros(3)

        for i in range(len(data)):
            argmax = np.argmax(action_sums/action_counters) if \
                np.all(action_counters > np.zeros(3)) else 'undecided'

            p = np.zeros(3) + self.epsilon/3
            if argmax != 'undecided':
                p[argmax] += 1-self.epsilon
            else:
                p /= np.sum(p)

            a = self.xz_a_mapping(data[i][0][0], data[i][0][1])
            r = data[i][1]

            action_counters[a] += 1
            action_sums[a] += r


        return action_counters, action_sums


        
