#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with epsilon-LinUCB. 
This file contains:
    1. data generation process for epsilon-LinUCB
    2. data weight calculation for epsilon-LinUCB
    3. various proposal sampling processes for epsilon-LinUCB
       in the setting of non-stationarity testing in
       a contextual bandit
    4. proposal sampling weighting calculations for the above proposals

NB: set epsilon = 0 to simply get regular LinUCB
"""
import numpy as np

class ELinUCB:
    def __init__(self, T, epsilon, d, null, conditional):
        self.T = T
        self.epsilon = epsilon
        self.d = d
        self.null = null
        self.coin_flips = []
        self.conditional = conditional

    def get_dataset(self):
        data = []

        # tracking parameters in LinUCB
        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = np.random.multivariate_normal(np.ones(self.d), np.eye(self.d))
            
            for a in range(2):
                A_a_inv = np.linalg.inv(A[a])
                p[a] = np.dot(np.dot(A_a_inv,b[a]), x) + np.sqrt(np.dot(np.dot(x,A_a_inv),x))

            U = np.random.uniform()
            # take first two actions at first two timesteps (epsilon-greedily)
            if i == 0 or i == 1:
                action = i if U < 1-self.epsilon else np.random.choice(2)
                
                # collect coin flips if conditional
                if self.conditional:
                    if action == i:
                        self.coin_flips.append(1)
                    else:
                        self.coin_flips.append(0)
            else:
                # otherwise regular epsilon-greedy action selection
                action = np.argmax([p[0], p[1]]) if U < 1-self.epsilon else np.random.choice(2)
                
                # collect coin flips if conditional
                if self.conditional:
                    if action == np.argmax([p[0], p[1]]):
                        self.coin_flips.append(1)
                    else:
                        self.coin_flips.append(0)

            # sample reward according to null or alternative correspondingly
            if i == self.T-1 and not self.null:
                r = 5*(2*action-1) + np.sum(x[:10]) + np.random.normal()
            else:
                r = -5*(2*action-1) + np.sum(x[:10]) + np.random.normal()

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
                if self.conditional:
                    coin_flip = self.coin_flips[i]
                    action_to_take = i if coin_flip == 1 else 1-i
                    if action != action_to_take:
                        return 0.
                else:
                    if action == i:
                        prod *= 1-self.epsilon/2
                    else:
                        prod *= self.epsilon/2
            # otherwise, it's regular eLinUCB action selection prob
            else:
                if self.conditional:
                    coin_flip = self.coin_flips[i]
                    action_to_take = np.argmax([p[0], p[1]]) if coin_flip == 1 else 1-np.argmax([p[0], p[1]])
                    if action != action_to_take:
                        return 0.
                else:
                    if action == np.argmax([p[0], p[1]]):
                        prod *= 1-self.epsilon/2
                    else:
                        prod *= self.epsilon/2

            A[action] = A[action] + np.outer(x, x)
            b[action] = b[action] + r*x
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

    def simulation1(self, data, propose_or_weight):
        ''''The simulation1 distribution samples without replacement, 
        proportional to the policy probabilities'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        prob = 1.

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            A_a_inv_pot, b_pot, p_pot = dict(), dict(), dict()
            # set potential A, b, p dicts to calculate the value for the
            # current index
            for k in range(2):
                A_a_inv_pot[k] = np.linalg.inv(A[k])
                b_pot[k] = b[k]
                p_pot[k] = p[k]


            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if i == 0 or i == 1:
                        max_action = i
                    else:
                        for a in range(2):
                            p_pot[a] = np.dot(np.dot(A_a_inv_pot[a],b_pot[a]), x_curr) + np.sqrt(np.dot(np.dot(x_curr,A_a_inv_pot[a]),x_curr))
                        
                        max_action = np.argmax([p_pot[0], p_pot[1]])

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

            A[sampled_a] = A[sampled_a] + np.outer(sampled_x, sampled_x)
            b[sampled_a] = b[sampled_a] + sampled_r*sampled_x

        if propose_or_weight:
            return shuffled_data, prob 
        else:
            return prob


    def simulation2(self, data, propose_or_weight):
        ''''The simulation2 distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''

        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''
        if propose_or_weight:
            return self.simulation2_propose(data)
        else:
            return self.simulation2_weight(data)


    def simulation2_propose(self, data):
        ''''The simulation2 distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function just samples from the proposal'''

        prob = 1.

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            # determine if going to be greedy or not
            greedy_or_not = np.random.choice(2, p=[self.epsilon/2, 1-self.epsilon/2])

            # track to see if it was forced to take a certain choice of greedy_or_not
            forced = True

            A_a_inv_pot, b_pot, p_pot = dict(), dict(), dict()
            
            # set potential A, b, p dicts to calculate the value for the
            # current index
            for k in range(2):
                A_a_inv_pot[k] = np.linalg.inv(A[k])
                b_pot[k] = b[k]
                p_pot[k] = p[k]
            
            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if i == 0 or i == 1:
                        max_action = i
                    else:
                        for a in range(2):
                            p_pot[a] = np.dot(np.dot(A_a_inv_pot[a],b_pot[a]), x_curr) + np.sqrt(np.dot(np.dot(x_curr,A_a_inv_pot[a]),x_curr))
                        
                        max_action = np.argmax([p_pot[0], p_pot[1]])
                    
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


            # update action-context-reward trackers
            A[sampled_a] = A[sampled_a] + np.outer(sampled_x, sampled_x)
            b[sampled_a] = b[sampled_a] + sampled_r*sampled_x

        return shuffled_data, prob

    
    def simulation2_weight(self, data):
        ''''The simulation2 distribution samples, at each timestep, an action
        based on the previously selected data, epsilon-greedily wrt LinUCB and then samples 
        correspondingly from the remaining timesteps.'''
        
        '''This function calculates weights'''

        prob = 1.

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)

            (a_i, x_i) = data[i][0]
            # calculate if current action is greedy_or_not
            
            if i == 0 or i == 1:
                max_action = i
            else:
                A_a_inv_pot, b_pot, p_pot = dict(), dict(), dict()      
                # set potential A, b, p dicts to calculate the value for the
                # current index
                for k in range(2):
                    A_a_inv_pot[k] = np.linalg.inv(A[k])
                    b_pot[k] = b[k]
                    p_pot[k] = p[k]

                for a in range(2):
                    p_pot[a] = np.dot(np.dot(A_a_inv_pot[a],b_pot[a]), x_i) + np.sqrt(np.dot(np.dot(x_i,A_a_inv_pot[a]),x_i))
                
                max_action = np.argmax([p_pot[0], p_pot[1]])

            if max_action == a_i:
                sampled_greedy_or_not = 1
            else:
                sampled_greedy_or_not = 0
            

            forced = True
            A_a_inv_pot, b_pot, p_pot = dict(), dict(), dict()
                    
            # set potential A, b, p dicts to calculate the value for the
            # current index
            for k in range(2):
                A_a_inv_pot[k] = np.linalg.inv(A[k])
                b_pot[k] = b[k]
                p_pot[k] = p[k]

            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]

                    if i == 0 or i == 1:
                        max_action = i
                    else:
                        for a in range(2):
                            p_pot[a] = np.dot(np.dot(A_a_inv_pot[a],b_pot[a]), x_curr) + np.sqrt(np.dot(np.dot(x_curr,A_a_inv_pot[a]),x_curr))
                        
                        max_action = np.argmax([p_pot[0], p_pot[1]])

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
                prob *= [self.epsilon/2, 1-self.epsilon/2][sampled_greedy_or_not]

            # normalize and then sample accordingly if proposing,
            # otherwise set to current index
            p_vec = p_vec/np.sum(p_vec)
            sample = i
            curr_selected[sample] = 1.
            prob *= p_vec[sample]

            # update action-context-reward trackers
            ((sampled_a, sampled_x), sampled_r) = ((data[sample][0][0], data[sample][0][1]), data[sample][1])
            A[sampled_a] = A[sampled_a] + np.outer(sampled_x, sampled_x)
            b[sampled_a] = b[sampled_a] + sampled_r*sampled_x

        return prob


    def simulation3(self, data, propose_or_weight):
        ''''The simulation3 distribution samples without replacement, 
        conditioning on the coin flips made by elinucb'''
        
        '''The input propose_or_weight is True if doing sampling, 
        and False if calculating the weight'''

        A, b, p = dict(), dict(), dict()

        for k in range(2):
            A[k] = np.eye(self.d)
            b[k] = np.zeros(self.d)
            p[k] = 0

        shuffled_data = []

        # pointer list of already-selected indices
        curr_selected = np.zeros(self.T)

        for i in range(self.T):
            p_vec = np.zeros(self.T)
            coin_flip = self.coin_flips[i]
            A_a_inv_pot, b_pot, p_pot = dict(), dict(), dict()
            # set potential A, b, p dicts to calculate the value for the
            # current index
            for k in range(2):
                A_a_inv_pot[k] = np.linalg.inv(A[k])
                b_pot[k] = b[k]
                p_pot[k] = p[k]


            for i_ in range(self.T):
                # only go through non-selected indices
                if curr_selected[i_] == 0:
                    (a_curr, x_curr) = data[i_][0]
                    if i == 0 or i == 1:
                        max_action = i
                    else:
                        for a in range(2):
                            p_pot[a] = np.dot(np.dot(A_a_inv_pot[a],b_pot[a]), x_curr) + np.sqrt(np.dot(np.dot(x_curr,A_a_inv_pot[a]),x_curr))
                        
                        max_action = np.argmax([p_pot[0], p_pot[1]])

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

            A[sampled_a] = A[sampled_a] + np.outer(sampled_x, sampled_x)
            b[sampled_a] = b[sampled_a] + sampled_r*sampled_x

        if propose_or_weight:
            return shuffled_data, 1. 
        else:
            return 1.


    def get_proposal(self, data, style):
        if style == 'u':
            return self.uniform(data, True)
        if style == 's1':
            return self.simulation1(data, True)
        if style == 's2':
            return self.simulation2(data, True)
        if style == 's3':
            return self.simulation3(data, True)
        


    def get_proposal_weight(self, proposal, starting, style):
        if style == 'u':
            return self.uniform(proposal, False)
        if style == 's1':
            return self.simulation1(proposal, False)
        if style == 's2':
            return self.simulation2(proposal, False)
        if style == 's3':
            return self.simulation3(proposal, False)
                
                