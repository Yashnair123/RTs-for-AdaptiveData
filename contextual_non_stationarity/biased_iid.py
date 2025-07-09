#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm-specific file for non-stationarity testing with biased IID sampler. 

"""
import numpy as np

class BiasedIID:
    def __init__(self, T, epsilon, d, null):
        self.T = T
        self.epsilon = epsilon
        self.d = d
        self.null = null

    def get_dataset(self):
        data = []

        for i in range(self.T):
            # generate context; in this case, will be sparse vector in d dimensions
            x = np.random.multivariate_normal(np.ones(self.d), np.eye(self.d))
            
            action = np.random.choice(2, p=[1-self.epsilon, self.epsilon])
            # sample reward according to null or alternative correspondingly
            if i == self.T-1 and not self.null:
                r = 5*(2*action-1) + np.sum(x[:10]) + np.random.normal()
            else:
                r = -5*(2*action-1) + np.sum(x[:10]) + np.random.normal()

            data.append([[action,x],r])
        return data
           
    
    def get_data_weight(self, data):
        return 1.

    
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

    
    def get_proposal(self, data, style):
        if style == 'u':
            return self.uniform(data, True)
        

    def get_proposal_weight(self, proposal, starting, style):
        if style == 'u':
            return self.uniform(proposal, False)
                
                