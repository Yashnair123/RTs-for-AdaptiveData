import numpy as np
import copy

''''
This file contains the MCMC and MC randomization tests. Note that only
the MCMC function below can handle dependent sampling.
'''

def mcmc_construct_rand_p_value(algo, data, test_stat, style, num_samples=1000):
        '''
        Construct the unweighted MCMC randomization test's p-values using
        permuted serial sampler
        '''
        p_plus = 0.
        p_minus = 0.

        s_id = test_stat(data)

        # add the contribution of s_id to p_plus
        p_plus += 1

        proposal, prob = algo.get_proposal(data, style)


        # uniformly select the random starting point from which we run the
        # chain forwards and then backwards
        starting_point = np.random.choice(num_samples+1)

        # run the chain forwards
        prev_proposal_weight = algo.get_proposal_weight(data, proposal, style)
        prev_data_weight = algo.get_data_weight(data)
        prev_point = copy.deepcopy(data)
        for _ in range(starting_point):
            if _ > 0:
                proposal, prob = algo.get_proposal(data, style)
                prev_proposal_weight = algo.get_proposal_weight(prev_point, proposal, style)
            weight = algo.get_data_weight(proposal)
            alpha = weight*prev_proposal_weight\
                    /(prev_data_weight*prob)
            A = min(alpha,1)
            U = np.random.uniform()
            # dont transition in this case
            if U < A:
                sample = copy.deepcopy(proposal)
                prev_data_weight = weight
                prev_point = copy.deepcopy(proposal)
            else:
                sample = copy.deepcopy(prev_point)

            # calculate the test statistic on the sampled data
            s = test_stat(sample)

            # update numerators of p-values, accordingly
            p_plus += 1 if s >= s_id else 0
            p_minus += 1 if s > s_id else 0

        # run the chain backwards
        proposal, prob = algo.get_proposal(data, style)
        prev_proposal_weight = algo.get_proposal_weight(data, proposal, style)
        prev_data_weight = algo.get_data_weight(data)
        prev_point = copy.deepcopy(data)
        for _ in range(num_samples-starting_point):
            if _ > 0:
                proposal, prob = algo.get_proposal(data, style)
                prev_proposal_weight = algo.get_proposal_weight(prev_point, proposal, style)
            weight = algo.get_data_weight(proposal)
            alpha = weight*prev_proposal_weight\
                    /(prev_data_weight*prob)
            A = min(alpha,1)
            U = np.random.uniform()
            # dont transition in this case
            if U < A:
                sample = copy.deepcopy(proposal)
                prev_proposal_weight = prob
                prev_data_weight = weight
                prev_point = copy.deepcopy(proposal)
            else:
                sample = copy.deepcopy(prev_point)

            # calculate the test statistic on the sampled data
            s = test_stat(sample)

            # update numerators of p-values, accordingly
            p_plus += 1 if s >= s_id else 0
            p_minus += 1 if s > s_id else 0

        
        # as p_plus and p_minus are not yet normalized by the number
        # of samples, do so and return
        return p_plus/(num_samples+1), p_minus/(num_samples+1)



def mc_construct_rand_p_value(algo, data, test_stat, style, num_samples=1000):
        '''
        Construct the unweighted MCMC randomization test's p-values using
        permuted serial sampler
        '''
        # p_plus and p_minus will be unnormalized p values; denominator
        # is the normalizing factor. We return the correctly normalized
        # p values at the end
        denominator = 0.
        p_plus = 0.
        p_minus = 0.


        s_id = test_stat(data)

        sample_prob_set = [algo.get_proposal(data, style) for i in range(num_samples)]
        
        # the set of samples and probabilities for those samples, correspondingly
        sample_set = [sample_prob_set[i][0] for i in range(num_samples)]
        prob_set = [sample_prob_set[i][1] for i in range(num_samples)]


        # add the original data to the list of samples
        orig_data_probs = algo.get_proposal_weight(data, data, style)

        sample_set.append(data)
        prob_set.append(orig_data_probs)

        
        n_eff_denom = 0.

        ps = []

        # reverse order of list, because weights of actual resamples won't be 0 when under true data
        for ind in list(reversed(list(range(len(sample_set))))):
            sample = sample_set[ind]
            proba = prob_set[ind]
            s = test_stat(sample)

            weight = algo.get_data_weight(sample)

            # instead of simply calculating the product of all samples, 
            # recognize that the independence of all of our sampling schemes
            # guarantees that we may write the conditional probability of the
            # remaining M-1 samples given the ith as 
            # a_1 * a_2 * ... * a_{i-1}*a_{i+1}*... * a_M
            # = a_1 * a_2 * ... * a_M/a_i. The numerator a_1 * a_2 * ... * a_M
            # is constant and in the numerator and denominator and may thus be cancelled
            # so that we only need multiply by 1/a_i, the inverse of the ith proposal probability
            
            # need to behave differently if style involves simulation1, simulation2, or simulation3
            # permutations at first
            if style in ['s1s', 's2s', 's3s']:
                weight_list = []
                for second_ind in range(len(sample_set)):
                    # don't want to include same index
                    if second_ind != ind:
                        weight_list.append(algo.get_proposal_weight(sample_set[second_ind],\
                             sample_set[ind], style))

                # obtain the weight list at the first timestep so that we can divide all future ones
                # by it to ensure weights aren't too small
                weight_list.sort()
                if ind == len(sample_set)-1:
                    first_weight_list = copy.deepcopy(weight_list)
                    p_i = weight # just the first sample will have the resampling weight as 1 
                    # because we divide by it
                else:
                    p_i = np.prod(np.array(weight_list)/np.array(first_weight_list))*weight
            else:
                p_i = (1./proba)*weight
            
            ps.append(p_i) # appending to list so that denom calculation doesn't get too small

            denominator += p_i
            p_plus += p_i if s >= s_id else 0
            p_minus += p_i if s > s_id else 0
        
        ps = np.array(ps)
        ps = ps/np.sum(ps)
        n_eff_denom = np.sum(ps**2)

        return p_plus/denominator, p_minus/denominator, (1./n_eff_denom)/num_samples



#############################################################
## USE THE BELOW ONLY FOR CONFIDENCE INTERVAL CONSTRUCTION ##
#############################################################


def conf_interval_mcmc_construct_rand_p_value(algo, data, test_stat, style, b, num_samples=1000):
        '''
        Construct the unweighted MCMC randomization test's p-values using
        permuted serial sampler
        '''
        p_plus = 0.
        p_minus = 0.

        s_id = test_stat(data)

        # add the contribution of s_id to p_plus
        p_plus += 1

        proposal, prob = algo.get_proposal(data, style, b)


        # uniformly select the random starting point from which we run the
        # chain forwards and then backwards
        starting_point = np.random.choice(num_samples+1)

        # run the chain forwards
        prev_proposal_weight = algo.get_proposal_weight(data, proposal, style, b)
        prev_data_weight = algo.get_data_weight(data, b)
        prev_point = copy.deepcopy(data)
        for _ in range(starting_point):
            if _ > 0:
                proposal, prob = algo.get_proposal(data, style, b)
                prev_proposal_weight = algo.get_proposal_weight(prev_point, proposal, style, b)
            weight = algo.get_data_weight(proposal, b)
            alpha = weight*prev_proposal_weight\
                    /(prev_data_weight*prob)
            A = min(alpha,1)
            U = np.random.uniform()
            # dont transition in this case
            if U < A:
                sample = copy.deepcopy(proposal)
                prev_data_weight = weight
                prev_point = copy.deepcopy(proposal)
            else:
                sample = copy.deepcopy(prev_point)

            # calculate the test statistic on the sampled data
            s = test_stat(sample)

            # update numerators of p-values, accordingly
            p_plus += 1 if s >= s_id else 0
            p_minus += 1 if s > s_id else 0

        # run the chain backwards
        proposal, prob = algo.get_proposal(data, style, b)
        prev_proposal_weight = algo.get_proposal_weight(data, proposal, style, b)
        prev_data_weight = algo.get_data_weight(data, b)
        prev_point = copy.deepcopy(data)
        for _ in range(num_samples-starting_point):
            if _ > 0:
                proposal, prob = algo.get_proposal(data, style, b)
                prev_proposal_weight = algo.get_proposal_weight(prev_point, proposal, style, b)
            weight = algo.get_data_weight(proposal, b)
            alpha = weight*prev_proposal_weight\
                    /(prev_data_weight*prob)
            A = min(alpha,1)
            U = np.random.uniform()
            # dont transition in this case
            if U < A:
                sample = copy.deepcopy(proposal)
                prev_proposal_weight = prob
                prev_data_weight = weight
                prev_point = copy.deepcopy(proposal)
            else:
                sample = copy.deepcopy(prev_point)

            # calculate the test statistic on the sampled data
            s = test_stat(sample)

            # update numerators of p-values, accordingly
            p_plus += 1 if s >= s_id else 0
            p_minus += 1 if s > s_id else 0

        
        # as p_plus and p_minus are not yet normalized by the number
        # of samples, do so and return
        return p_plus/(num_samples+1), p_minus/(num_samples+1)



def conf_interval_mc_construct_rand_p_value(algo, data, test_stat, style, b, num_samples=1000):
        '''
        Construct the unweighted MCMC randomization test's p-values using
        permuted serial sampler
        '''
        # p_plus and p_minus will be unnormalized p values; denominator
        # is the normalizing factor. We return the correctly normalized
        # p values at the end
        denominator = 0.
        p_plus = 0.
        p_minus = 0.

        s_id = test_stat(data)

        sample_prob_set = [algo.get_proposal(data, style, b) for i in range(num_samples)]
        
        # the set of samples and probabilities for those samples, correspondingly
        sample_set = [sample_prob_set[i][0] for i in range(num_samples)]
        prob_set = [sample_prob_set[i][1] for i in range(num_samples)]


        # add the original data to the list of samples
        orig_data_probs = algo.get_proposal_weight(data, data, style, b)

        sample_set.append(data)
        prob_set.append(orig_data_probs)


        for ind in list(reversed(list(range(len(sample_set))))):
            sample = sample_set[ind]
            proba = prob_set[ind]
            s = test_stat(sample)

            weight = algo.get_data_weight(sample, b)

            # instead of simply calculating the product of all samples, 
            # recognize that the independence of all of our sampling schemes
            # guarantees that we may write the conditional probability of the
            # remaining M-1 samples given the ith as 
            # a_1 * a_2 * ... * a_{i-1}*a_{i+1}*... * a_M
            # = a_1 * a_2 * ... * a_M/a_i. The numerator a_1 * a_2 * ... * a_M
            # is constant and in the numerator and denominator and may thus be cancelled
            # so that we only need multiply by 1/a_i, the inverse of the ith proposal probability
            
            if style in ['s1s', 's2s', 's3s']:
                weight_list = []
                for second_ind in range(len(sample_set)):
                    # don't want to include same index
                    if second_ind != ind:
                        weight_list.append(algo.get_proposal_weight(sample_set[second_ind],\
                             sample_set[ind], style, b))
                # obtain the weight list at the first timestep so that we can divide all future ones
                # by it to ensure weights aren't too small
                weight_list.sort()
                if ind == len(sample_set)-1:
                    first_weight_list = copy.deepcopy(weight_list)
                    p_i = weight # just the first sample will have the resampling weight as 1 
                    # because we divide by it
                else:
                    p_i = np.prod(np.array(weight_list)/np.array(first_weight_list))*weight
            else:
                p_i = (1./proba)*weight 
        

            denominator += p_i
            p_plus += p_i if s >= s_id else 0
            p_minus += p_i if s > s_id else 0

        return p_plus/denominator, p_minus/denominator





#############################################################
## USE THE BELOW ONLY FOR SHARED CONFORMAL INTERVAL CONSTRUCTION ##
#############################################################


def shared_mc_construct_rand_p_value(algo, data, test_stat, style, b_vals, num_samples=1000):
        '''
        Construct the weighted MC randomization test's p-values using
        permuted serial sampler, and sharing samples across multiple b's
        '''
        # 

        total_sample_prob_set = []

        for b in b_vals:
            b_data = copy.deepcopy(data)
            b_data[-1] = (b_data[-1][0],b) # transform just the last reward
            
            # generate samples
            sample_prob_set = [algo.get_proposal(b_data, style) for i in range(num_samples)]

            # for sample_prob in sample_prob_set:
            #     sample = sample_prob[0]
            #     b_location = sample.index((0,b)) if (0,b) \
            #             in sample else sample.index((1,b))

            orig_data_probs = algo.get_proposal_weight(b_data, b_data, style)
            sample_prob_set.append((b_data, orig_data_probs))

            total_sample_prob_set.append(sample_prob_set)
        
        # get true weights
        true_weights = dict()
        for indexer in range(len(b_vals)):
            for i in range(num_samples):
                sample = total_sample_prob_set[indexer][i][0]
                probs = algo.get_shared_data_weight(sample, b_vals)
                true_weights[(indexer, i)] = probs

            true_weights[(indexer, num_samples+1)] = [algo.get_data_weight(data)]*len(b_vals) # weight of true data (with last reward changed) is always same
        
        p_pluses = []
        p_minuses = []
        for indexer in range(len(b_vals)):
            b = b_vals[indexer]
            b_data = copy.deepcopy(data)
            b_data[-1] = (b_data[-1][0],b) # transform just the last reward
            
            denominator = 0.
            p_plus = 0.
            p_minus = 0.

            s_id = test_stat(b_data)
            p_i = 1. * true_weights[(indexer, num_samples+1)][0] # this is the true weight of the data

            denominator += p_i
            p_plus += p_i # must add for the true dataset

            true_data_proposal_weight = total_sample_prob_set[0][-1][1] # proposal weight of true data doesn't depend on b anyways
            for indexer_ in range(len(b_vals)):
                for i in range(num_samples):
                    p_i = (true_data_proposal_weight/total_sample_prob_set[indexer_][i][1]) \
                        * true_weights[(indexer_, i)][indexer] # need true weight at indexer, not indexer_

                    sample = copy.deepcopy(total_sample_prob_set[indexer_][i][0])
                    b_location = sample.index((0,b_vals[indexer_])) if (0,b_vals[indexer_]) \
                        in sample else sample.index((1,b_vals[indexer_])) # find location of the changed b_value
                    sample[b_location] = (sample[b_location][0], b_vals[indexer]) # set the value at the b_location to be the correct b
                    
                    s = test_stat(sample)

                    denominator += p_i
                    p_plus += p_i if s >= s_id else 0
                    p_minus += p_i if s > s_id else 0

            p_plus = p_plus/denominator
            p_minus = p_minus/denominator

            p_pluses.append(p_plus)
            p_minuses.append(p_minus)

        return p_pluses, p_minuses