#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with Q learning
with epsilon-greedy selection in an MDP. 
This file contains:
    1. data generation process for epsilon-greedy
    2. data weight calculation for epsilon-greedy
    3. various resampling procedures for epsilon-greedy
       in the setting of an MDP with 3 states and 2 actions
    4. resampling weighting calculations for the above resampling distributions
"""
import numpy as np


class EpsilonGreedy:
    
    def __init__(self, T, epsilon, null, conditional):
        self.T = T
        self.null = null
        self.epsilon = epsilon
        self.coin_flips = []
        self.conditional = conditional
           
    def get_dataset(self):
        data = []
        # initialize Q function
        Q = np.zeros((3,2))
        
        # start at state 0 if not uniform iid baseline, else uniform
        if self.epsilon == 1.:
            state = np.random.choice([0,1,2])
        else:
            state = 0
        for i in range(self.T):
            U = np.random.uniform()
            # epsilon-greedy action selection
            action = np.argmax(Q[state]) if U < 1-self.epsilon else np.random.choice(2)
            
            # if conditional, keep track of coin flips
            if self.conditional:
                if action == np.argmax(Q[state]):
                    self.coin_flips.append(1)
                else:
                    self.coin_flips.append(0)
            r = state
            V = np.random.uniform()

            # transition accordingly if at last timestep and under null or alternative
            if i != self.T-1 or self.null:
                next_state = (state + (2*action-1)) % 3 if V < 0.95 else (state - (2*action-1)) % 3
            else:
                next_state = (state - (2*action-1)) % 3 if V < 0.95 else (state + (2*action-1)) % 3
            
            # update Q function
            Q[state][action] = Q[state][action] + 0.05*(r + np.max(Q[next_state])-Q[state][action])

            # append to data
            data.append(((state,action), next_state))

            # transition
            # if uniform iid baseline, just make next state unif
            if self.epsilon == 1.:
                state = np.random.choice([0,1,2])
            else:
                state = next_state
        # get last action and append to data. Note that the form of the dataset is somewhat
        # different here in that there is a single action at the last timestep, rather than
        # (state, action) (i.e., (X,Y))
        U = np.random.uniform()
        action = np.argmax(Q[state]) if U < 1-self.epsilon else np.random.choice(2)
        
        # if conditional, keep track of coin flips
        if self.conditional:
            if action == np.argmax(Q[state]):
                self.coin_flips.append(1)
            else:
                self.coin_flips.append(0)
    
        data.append(action)
        return data
    
    
    def get_data_weight(self, orig_data):
        # if uniform, just return 1
        if self.epsilon == 1.:
            return 1.
        # if the data is flagged, it must be assigned weight 0
        if orig_data == 'flag':
            return 0.
        Q = np.zeros((3,2))
        prod = 1.
        state = 0

        # iterate across timesteps
        for i in range(self.T):

            ((state,action), next_state) = orig_data[i]
            max_action = np.argmax(Q[state])

            # calculate coin action to take if conditional
            if self.conditional:
                coin_flip = self.coin_flips[i]
                action_to_take = max_action if coin_flip == 1 else 1-max_action

                if action != action_to_take:
                    return 0.

            # calculate epsilon-greedy action selection probabilities based on the max 
            # action
            if action == max_action:
                prod *= (1-self.epsilon/2)
            else:
                prod *= (self.epsilon/2)
            r = state

            # update Q function and transition accordingly
            Q[state][action] = Q[state][action] + 0.05*(r + np.max(Q[next_state])-Q[state][action])
            state = next_state

        # account for what to do if conditional
        if self.conditional:
            return 1.
        else:
            return prod

    def uniform(self, data, propose_or_weight):
        '''This sampling scheme samples permutations from the uniform 
        distribution over permutations'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''


        # if uniform baseline, any permutation is OK
        if self.epsilon == 1.:
            if propose_or_weight:
                perm = list(np.random.choice(self.T, self.T, replace=False))
                shuffled_data = [data[perm[i]] for i in range(self.T)]
                shuffled_data.append(data[-1])
                
                # return the data and its weight
                return shuffled_data, 1.
            else:
                # uniform sampling always has weight 1
                return 1.
        else:
            if propose_or_weight:
                # will convert the data into the usual "string" of the
                # form (state, action), (next state, next action), ...
                # we will then permute the string and then convert back 
                # to the original data format
                string = []
                shuffled_string = []
                
                for i in range(len(data)-1):
                    string.append(data[i][0])
                # last action timestep
                string.append((data[i][1], data[-1]))

                # index_proceeding will be a dict whose keys
                # are (s,a) pairs and the values at the key 
                # (s,a) is simply the list of indices which proceed
                # any index at which string is equal to (s,a)
                index_proceeding = dict()

                # start at 1 because the first index proceeds nothing
                for i in range(1, len(string)):
                    (s,a) = string[i-1]
                    if (s,a) in index_proceeding:
                        index_proceeding[(s,a)].append(i)
                    else:
                        index_proceeding[(s,a)] = [i]
                

                # we are now ready to "shuffle" string
                # in a uniform fashion
                # marker array for those indices which have already been selected
                curr_selected = np.zeros(len(string))

                for i in range(len(string)):
                    # if at first timestep, then we just remain the same
                    if i == 0:
                        shuffled_string.append(string[i])
                    # otherwise uniformly select over all allowable indices
                    # (i.e., those in index_proceeding)
                    else:
                        (s,a) = shuffled_string[i-1]
                        indices = []
                        # if we run out, then just return a flag to indicate
                        # to the weight function that this "sample" must be assigned
                        # weight 0
                        if (s,a) not in index_proceeding:
                            return 'flag', 1.
                        
                        # otherwise, add each index, not already selected,
                        # in the indicies proceeding (s,a) to the list
                        for index in index_proceeding[(s,a)]:
                            if curr_selected[index] == 0:
                                indices.append(index)

                        # once again, if run out of indices, then return the flag
                        if len(indices) == 0:
                            return 'flag', 1.
                        
                        # sample uniformly from these remaining indices
                        sample = np.random.choice(indices)

                        # append to the sample
                        shuffled_string.append(string[sample])
                        curr_selected[sample] = 1.

                # convert the shuffled string back into the original data format
                # to get shuffled_data
                shuffled_data = []
                for i in range(len(shuffled_string)-1):
                    shuffled_data.append((shuffled_string[i], shuffled_string[i+1][0]))
                shuffled_data.append(shuffled_string[-1][1])

                # since we permuted uniformly, the proposal weight is 1
                return shuffled_data, 1.
            else:
                return 1.

        
    def imitation(self, data, propose_or_weight):
        '''This sampling scheme samples permutations from the imitation distributions 
        distribution over permutations of MDP data'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        # will convert the data into the usual "string" of the
        # form (state, action), (next state, next action), ...
        # we will then permute the string and then convert back 
        # to the original data format
        string = []
        shuffled_string = []
        prod = 1.
        
        for i in range(len(data)-1):
            string.append(data[i][0])
        # last action timestep
        string.append((data[i][1], data[-1]))

        # index_proceeding will be a dict whose keys
        # are (s,a) pairs and the values at the key 
        # (s,a) is simply the list of indices which proceed
        # any index at which string is equal to (s,a)
        index_proceeding = dict()

        # start at 1 because the first index proceeds nothing
        for i in range(1, len(string)):
            (s,a) = string[i-1]
            r = s
            if (s,a) in index_proceeding:
                index_proceeding[(s,a)].append(i)
            else:
                index_proceeding[(s,a)] = [i]
        
        Q = np.zeros((3,2))
        # we are now ready to "shuffle" string
        # in a uniform fashion
        # marker array for those indices which have already been selected
        curr_selected = np.zeros(len(string))

        for i in range(len(string)):
            probs = np.zeros(len(string))
            # if at first timestep, then we just remain the same
            if i == 0:
                shuffled_string.append(string[i])
                curr_selected[i] = 1
            # otherwise select over all allowable indices
            # (i.e., those in index_proceeding)
            else:
                (s,a) = shuffled_string[i-1]
                r = s
                indices = []
                # if we run out, then just return a flag to indicate
                # to the weight function that this "sample" must be assigned
                # weight 0
                if (s,a) not in index_proceeding:
                    return 'flag', 1.
                
                # otherwise, add each index, not already selected,
                # in the indicies proceeding (s,a) to the list
                for index in index_proceeding[(s,a)]:
                    if curr_selected[index] == 0:
                        indices.append(index)

                # once again, if run out of indices, then return the flag
                if len(indices) == 0:
                    return 'flag', 1.
                
                # sample according to the imitation distribution
                # first set the probs p-vector over these selectable indices
                for i_ in indices:
                    # calculate the Q function for each of these indices
                    Q_pot = np.zeros((3,2))
                    for s_ in range(3):
                        for a_ in range(2):
                            Q_pot[s_][a_] = Q[s_][a_]

                    (curr_state, curr_action) = string[i_]
                    
                    # update the potential Q function
                    Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[curr_state])-Q[s][a])
                    max_action = np.argmax(Q_pot[curr_state])

                    # weight indices epsilon-greedily
                    if curr_action == max_action:
                        probs[i_] = 1-self.epsilon/2
                    else:
                        probs[i_] = self.epsilon/2

                # if none left, then sample uniformly among potential indices (only occurs for epsilon=0)
                if np.all(probs == 0):
                    for i_ in indices:
                        probs[i_] = 1

                # normalize and then sample accordingly if proposing,
                # otherwise set to current index
                probs = probs/np.sum(probs)
                if propose_or_weight:
                    sample = np.random.choice(len(string), p=probs)
                else:
                    sample = i

                (sampled_next_state, sampled_next_action) = string[sample]

                # update Q function
                Q[s][a] = Q[s][a] + 0.05*(r + np.max(Q[sampled_next_state])-Q[s][a])
                prod *= probs[sample] if self.epsilon != 0 else 1. # we only need multiply by 1 in the 
                # epsilon = 0 (i.e., deterministic) case, since it is truncated uniform


                shuffled_string.append(string[sample])

                curr_selected[sample] = 1.
        
        # convert the shuffled string back into the original data format
        # to get shuffled_data
        shuffled_data = []
        for i in range(len(shuffled_string)-1):
            shuffled_data.append((shuffled_string[i], shuffled_string[i+1][0]))
        shuffled_data.append(shuffled_string[-1][1])

        if propose_or_weight:
            # since we permuted uniformly, the proposal weight is 1
            return shuffled_data, prod
        else:
            return prod


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
        
        '''This function just samples from the proposal'''

        # will convert the data into the usual "string" of the
        # form (state, action), (next state, next action), ...
        # we will then permute the string and then convert back 
        # to the original data format
        string = []
        shuffled_string = []
        prod = 1.
        
        for i in range(len(data)-1):
            string.append(data[i][0])
        # last action timestep
        string.append((data[i][1], data[-1]))

        # index_proceeding will be a dict whose keys
        # are (s,a) pairs and the values at the key 
        # (s,a) is simply the list of indices which proceed
        # any index at which string is equal to (s,a)
        index_proceeding = dict()

        # start at 1 because the first index proceeds nothing
        for i in range(1, len(string)):
            (s,a) = string[i-1]
            if (s,a) in index_proceeding:
                index_proceeding[(s,a)].append(i)
            else:
                index_proceeding[(s,a)] = [i]
        
        Q = np.zeros((3,2))
        # we are now ready to "shuffle" string
        # in a uniform fashion
        # marker array for those indices which have already been selected
        curr_selected = np.zeros(len(string))

        for i in range(len(string)):
            probs = np.zeros(len(string))
            # if at first timestep, then we just remain the same
            if i == 0:
                shuffled_string.append(string[i])
                curr_selected[i] = 1
            # otherwise uniformly select over all allowable indices
            # (i.e., those in index_proceeding)
            else:
                (s,a) = shuffled_string[i-1]
                r = s
                indices = []
                # if we run out, then just return a flag to indicate
                # to the weight function that this "sample" must be assigned
                # weight 0
                if (s,a) not in index_proceeding:
                    return 'flag', 1.
                
                # otherwise, add each index, not already selected,
                # in the indicies proceeding (s,a) to the list
                for index in index_proceeding[(s,a)]:
                    if curr_selected[index] == 0:
                        indices.append(index)

                # once again, if run out of indices, then return the flag
                if len(indices) == 0:
                    return 'flag', 1.
                
                # sample whether or not epsilon-greedy behaves greedily or not in this step
                greedy_or_not = np.random.choice(2, p=[self.epsilon/2, 1-self.epsilon/2])
                
                # also determine if we were forced to pick a certain action by using "forced"
                forced = True

                # sample according to the re_imitation distribution
                # first set the probs p-vector over these selectable indices
                for i_ in indices:
                    # calculate the Q function for each of these indices
                    Q_pot = np.zeros((3,2))
                    for s_ in range(3):
                        for a_ in range(2):
                            Q_pot[s_][a_] = Q[s_][a_]

                    (curr_state, curr_action) = string[i_]
                    
                    # update the potential Q function
                    Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[curr_state])-Q[s][a])
                    
                    max_action = np.argmax(Q_pot[curr_state])

                    # weight indices epsilon-greedily
                    if greedy_or_not == 1:
                        if curr_action == max_action:
                            probs[i_] = 1.
                        else:
                            forced = False # in this case, we saw an action not equal to what the coin flip said
                            # and so were not forced
                    else:
                        if curr_action != max_action:
                            probs[i_] = 1.
                        else:
                            forced = False # in this case, we saw an action not equal to what the coin flip said
                            # and so were not forced
                        

                # if none left, then we are forced to select the other coin flip, which characterizes
                # all the other timesteps in indices
                if np.all(probs == 0) or forced:
                    for i_ in indices:
                        probs[i_] = 1
                else:
                    prod *= [self.epsilon/2, 1-self.epsilon/2][greedy_or_not]

                # normalize and then sample accordingly if proposing,
                # otherwise set to current index
                probs = probs/np.sum(probs)
                sample = np.random.choice(len(string), p=probs)
                prod *= probs[sample]

                (sampled_next_state, sampled_next_action) = string[sample]

                # update Q function
                Q[s][a] = Q[s][a] + 0.05*(r + np.max(Q[sampled_next_state])-Q[s][a])

                # append to the sample
                shuffled_string.append(string[sample])

                curr_selected[sample] = 1.

        # convert the shuffled string back into the original data format
        # to get shuffled_data
        shuffled_data = []
        for i in range(len(shuffled_string)-1):
            shuffled_data.append((shuffled_string[i], shuffled_string[i+1][0]))
        shuffled_data.append(shuffled_string[-1][1])

        return shuffled_data, prod


    def re_imitation_weight(self, data):
        ''''The re_imitation distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function just calculates weights'''

        # will convert the data into the usual "string" of the
        # form (state, action), (next state, next action), ...
        # we will then permute the string and then convert back 
        # to the original data format
        string = []
        shuffled_string = []
        prod = 1.
        
        for i in range(len(data)-1):
            string.append(data[i][0])
        # last action timestep
        string.append((data[i][1], data[-1]))

        # index_proceeding will be a dict whose keys
        # are (s,a) pairs and the values at the key 
        # (s,a) is simply the list of indices which proceed
        # any index at which string is equal to (s,a)
        index_proceeding = dict()

        # start at 1 because the first index proceeds nothing
        for i in range(1, len(string)):
            (s,a) = string[i-1]
            if (s,a) in index_proceeding:
                index_proceeding[(s,a)].append(i)
            else:
                index_proceeding[(s,a)] = [i]
        
        Q = np.zeros((3,2))
        # we are now ready to "shuffle" string
        # in a uniform fashion
        # marker array for those indices which have already been selected
        curr_selected = np.zeros(len(string))

        for i in range(len(string)):
            probs = np.zeros(len(string))
            # if at first timestep, then we just remain the same
            if i == 0:
                shuffled_string.append(string[i])
                curr_selected[i] = 1
            # otherwise uniformly select over all allowable indices
            # (i.e., those in index_proceeding)
            else:
                (s,a) = shuffled_string[i-1]
                r = s
                
                (sampled_s, sampled_a) = string[i] # this is what was actually sampled

                # check to see if this is a greedy selection or a nongreedy one

                Q_pot = np.zeros((3,2))
                for s_ in range(3):
                    for a_ in range(2):
                        Q_pot[s_][a_] = Q[s_][a_]

                # update the potential Q function
                Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[sampled_s])-Q[s][a])
                max_action = np.argmax(Q_pot[sampled_s])

                # determine whether the sampled thing was greedy or not
                if max_action == sampled_a:
                    sampled_greedy_or_not = 1
                else:
                    sampled_greedy_or_not = 0

                indices = []
                # if we run out, then just return a flag to indicate
                # to the weight function that this "sample" must be assigned
                # weight 0
                if (s,a) not in index_proceeding:
                    return 'flag', 1.
                
                # otherwise, add each index, not already selected,
                # in the indicies proceeding (s,a) to the list
                for index in index_proceeding[(s,a)]:
                    if curr_selected[index] == 0:
                        indices.append(index)

                # once again, if run out of indices, then return the flag
                if len(indices) == 0:
                    return 'flag', 1.
                
                # determine if we were forced to pick a certain action by using "forced"
                forced = True

                for i_ in indices:
                    # calculate the Q function for each of these indices
                    Q_pot = np.zeros((3,2))
                    for s_ in range(3):
                        for a_ in range(2):
                            Q_pot[s_][a_] = Q[s_][a_]

                    (curr_state, curr_action) = string[i_]
                    
                    # update the potential Q function
                    Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[curr_state])-Q[s][a])
                    
                    max_action = np.argmax(Q_pot[curr_state])

                    curr_greedy_or_not = 1. if curr_action == max_action else 0.
                    if curr_greedy_or_not != sampled_greedy_or_not:
                        forced = False
                
                # now calculate weights according to the re_imitation distribution
                # first set the probs p-vector over these selectable indices
                for i_ in indices:
                    # calculate the Q function for each of these indices
                    Q_pot = np.zeros((3,2))
                    for s_ in range(3):
                        for a_ in range(2):
                            Q_pot[s_][a_] = Q[s_][a_]

                    (curr_state, curr_action) = string[i_]
                    
                    # update the potential Q function
                    Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[curr_state])-Q[s][a])
                    max_action = np.argmax(Q_pot[curr_state])

                    # weight indices epsilon-greedily
                    if sampled_greedy_or_not == 1:
                        if curr_action == max_action:
                            probs[i_] = 1.
                    else:
                        if curr_action != max_action:
                            probs[i_] = 1.

                # if not forced, then take into account the probability of the coin flip
                if not forced:
                    prod *= [self.epsilon/2, 1-self.epsilon/2][sampled_greedy_or_not]

                # normalize and then sample accordingly if proposing,
                # otherwise set to current index
                probs = probs/np.sum(probs)
                sample = i
                prod *= probs[sample]

                (sampled_next_state, sampled_next_action) = string[sample]

                # update Q function
                Q[s][a] = Q[s][a] + 0.05*(r + np.max(Q[sampled_next_state])-Q[s][a])

                # append to the sample
                shuffled_string.append(string[sample])

                curr_selected[sample] = 1.

        return prod


    def cond_imitation(self, data, propose_or_weight):
        '''This sampling scheme samples permutations according to the 
        cond_imitation distribution'''
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        # will convert the data into the usual "string" of the
        # form (state, action), (next state, next action), ...
        # we will then permute the string and then convert back 
        # to the original data format
        string = []
        shuffled_string = []
        
        for i in range(len(data)-1):
            string.append(data[i][0])
        # last action timestep
        string.append((data[i][1], data[-1]))

        # index_proceeding will be a dict whose keys
        # are (s,a) pairs and the values at the key 
        # (s,a) is simply the list of indices which proceed
        # any index at which string is equal to (s,a)
        index_proceeding = dict()

        # start at 1 because the first index proceeds nothing
        for i in range(1, len(string)):
            (s,a) = string[i-1]
            r = s
            if (s,a) in index_proceeding:
                index_proceeding[(s,a)].append(i)
            else:
                index_proceeding[(s,a)] = [i]
        
        Q = np.zeros((3,2))
        # we are now ready to "shuffle" string
        # in a uniform fashion
        # marker array for those indices which have already been selected
        curr_selected = np.zeros(len(string))

        for i in range(len(string)):
            coin_flip = self.coin_flips[i]
            probs = np.zeros(len(string))
            # if at first timestep, then we just remain the same
            if i == 0:
                shuffled_string.append(string[i])
                curr_selected[i] = 1
            # otherwise uniformly select over all allowable indices
            # (i.e., those in index_proceeding)
            else:
                (s,a) = shuffled_string[i-1]
                r = s
                indices = []
                # if we run out, then just return a flag to indicate
                # to the weight function that this "sample" must be assigned
                # weight 0
                if (s,a) not in index_proceeding:
                    return 'flag', 1.
                
                # otherwise, add each index, not already selected,
                # in the indicies proceeding (s,a) to the list
                for index in index_proceeding[(s,a)]:
                    if curr_selected[index] == 0:
                        indices.append(index)

                # once again, if run out of indices, then return the flag
                if len(indices) == 0:
                    return 'flag', 1.
                
                # sample according to the imitation distribution
                # first set the probs p-vector over these selectable indices
                for i_ in indices:
                    # calculate the Q function for each of these indices
                    Q_pot = np.zeros((3,2))
                    for s_ in range(3):
                        for a_ in range(2):
                            Q_pot[s_][a_] = Q[s_][a_]

                    (curr_state, curr_action) = string[i_]
                    
                    # update the potential Q function
                    Q_pot[s][a] = Q[s][a] + 0.05*(s + np.max(Q[curr_state])-Q[s][a])
                    max_action = np.argmax(Q_pot[curr_state])

                    # weight indices according to coin flips
                    if coin_flip == 1:
                        if curr_action == max_action:
                            probs[i_] = 1.
                    else:
                        if curr_action != max_action:
                            probs[i_] = 1.

                # if none left, then sample uniformly among potential indices
                if np.all(probs == 0):
                    for i_ in indices:
                        probs[i_] = 1

                # normalize and then sample accordingly if proposing,
                # otherwise set to current index
                probs = probs/np.sum(probs)
                if propose_or_weight:
                    sample = np.random.choice(len(string), p=probs)
                else:
                    sample = i

                (sampled_next_state, sampled_next_action) = string[sample]

                # update Q function
                Q[s][a] = Q[s][a] + 0.05*(r + np.max(Q[sampled_next_state])-Q[s][a])

                shuffled_string.append(string[sample])

                curr_selected[sample] = 1.

        # convert the shuffled string back into the original data format
        # to get shuffled_data
        shuffled_data = []
        for i in range(len(shuffled_string)-1):
            shuffled_data.append((shuffled_string[i], shuffled_string[i+1][0]))
        shuffled_data.append(shuffled_string[-1][1])

        if propose_or_weight:
            # since we permuted uniformly, the proposal weight is 1
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