"""
This file contains the cognitive models used to fit the data for the different experimental tasks.
"""
import numpy as np
from scipy.stats import gamma, beta

def compute_priors(params): 
    """
    Computes the prior probabilities for the parameters. 
    """
    epsilon = 1e-10
    prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0) + epsilon)
    prior_alphas = np.log(beta.pdf(params[1:], 1.1, 1.1) + epsilon)
    priors = prior_beta + np.sum(prior_alphas)
    return priors

def initialize_Q(data):
    """
    Initializes the option values Q for the models.
    """
    Q_ = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0]))))  # Option values [context/blocks, option/action]
    return Q_

class CM_Task1_Partial:
    """
    This class contains the cognitive models used to fit the data from the partial information task.
    Contains two fitting models with different number of parameters.
    These models are used to fit the data from the confirmation bias task.
    """
    def Model_1alpha(params, data, rewards, q_initial):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        Q_ += q_initial

        # sum up likelihood and update option values
        for i in range(len(data)):
            # prediction error = outcome - q value of context and option
            deltaI_c = rewards[i, 0] - Q_[data[i, 2]-1, data[i, 0]-1] 
            # likelihood += beta * Q[context, option] - np.log(np.sum(np.exp(beta * Q[context, all options])))
            log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
            # update q value of chosen option
            # Q[context, option] += alpha * deltaI
            Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c
        return -(priors + log_lik)

    def Model_2alpha(params, data, rewards, q_initial):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        Q_ += q_initial

        for i in range(len(data)):
            deltaI_c =  rewards[i, 0] - Q_[data[i, 2]-1, data[i, 0]-1]
            log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
            Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
        return -(priors + log_lik)

class CM_Task2_Full:
    """
    This class contains cognitive models that are only fit on the full feedback blocks and free trials.
    It does not take the forced tirals into account.
    """
    def Model_2alpha(params, data): #confirmation bias model
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1]
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]
            if (data[i, 3] == 1) and (data[i, 6] == 1): # if free choice and full feedback
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option; conf alpha for PPE, disconf alpha for NPE
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
                # update q value of unchosen option; disconf alpha for PPE, conf alpha for NPE
                Q_[data[i, 2] - 1, data[i, 4] - 1] += params[2] * deltaI_u * (deltaI_u > 0) + params[1] * deltaI_u * (deltaI_u < 0)
        return -(priors + log_lik)

    def Model_4alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        
        for i in range(len(data)):
            # prediction error
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1]
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]
            if (data[i, 3] == 1) and (data[i, 6] == 1): # if free choice and full feedback
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option, different alphas for each PE
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
                Q_[data[i, 2] - 1, data[i, 4] - 1] += params[3] * deltaI_u * (deltaI_u > 0) + params[4] * deltaI_u * (deltaI_u < 0)
        return -(priors + log_lik)

class CM_Task2_Partial:
    """
    This class contains cognitive models that are only fit on the partial feedback blocks and free trials.
    It does not take the forced tirals into account.
    It is equivalent to the cognitive models of the optimism bias task but has different initial Q-values (0 instead of 0.25)
    """

    def Model_1alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        
        for i in range(len(data)):
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1]
            if (data[i, 3] == 1) and (data[i, 6] ==0): # if free choice and partial feedback
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c
        return -(priors + log_lik)

    def Model_2alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 

        for i in range(len(data)):
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 
            if (data[i, 3] == 1) and (data[i, 6] == 0): # if free choice and partial feedback
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
        return -(priors + log_lik)

class CM_Task3_Agency_Mixed:
    '''
    Contains three fitting models with different number of parameters.
    These models are used to fit the blocks from the agency task containing forced choices.
    '''

    def Model_3alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        
        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI

        return -(priors + log_lik)

    def Model_4alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        
        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI * (deltaI > 0) + params[4] * deltaI * (deltaI < 0)
        return -(priors + log_lik)

class CM_Task3_Agency_Free:
    '''
    Contains three fitting models with different number of parameters.
    These models are used to fit the data from the agency task containing only free choices.
    '''

    def Model_1alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 
        
        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI

        return -(priors + log_lik)

    def Model_2alpha(params, data):
        priors = compute_priors(params)
        log_lik = 0
        Q_ = initialize_Q(data) 

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                log_lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)

        return -(priors + log_lik)
