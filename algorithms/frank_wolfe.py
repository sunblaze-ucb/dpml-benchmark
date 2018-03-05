import csv
import math
import numpy as np
from common.common import (Algorithm, DEFAULT_NUM_ITERS)
from lossfunctions.logistic_regression import LogisticRegression
from lossfunctions.huber_svm import HuberSVM, HuberSVMRegular
from common.noise import compute_gamma_noise
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
import os

"""This file contains the implementations of non-proivate frank-wolfe and
    differential-private frank-wolfe algorithm. We assume the domain is a l1
    ball."""

def compute_alpha(corner_size, gradient, m, noise_para, theta):

    alpha = gradient*corner_size
    noise = np.random.laplace(scale=noise_para, size=m)

    alpha = alpha+noise
    corner_size = (np.ones(m)*corner_size).tolist()
    corner_num = np.arange(m).tolist()
    return list(zip(alpha, corner_size, corner_num))

def private_frank_wolfe(x, y, loss_gradient, eps, delta,
                        step_size=1, num_iters=10,
                        constraint=100, L=1):
    n = x.shape[0]
    m = x.shape[1]
    minibatch_size = 1000

    if constraint is None:
        constraint = 100

    if(n==0):
        raise Exception("No training Data")

    theta = np.zeros(m)
    # uncomment the following to start with a random theta
    # theta = (np.random.rand(x.shape[1]) - .5) * 10

    """the mu constant in frank_wolfe algorithm"""
    """We consider L1 = 1, C1 = 2*x.shape[1], T = 1000, n = len(x)"""
    corner_size = constraint
    noise_para = L*corner_size*math.sqrt(8*num_iters*math.log(1/delta))/(n*eps)
    
    if isinstance(x, csr_matrix):
        data = csr_matrix(hstack((x, csr_matrix(y).T)))
    else:
        data = np.column_stack((x, y))

    np.random.seed(ord(os.urandom(1)))
    for i in range(num_iters):
        minibatch = data[np.random.choice(data.shape[0],
                                          minibatch_size,
                                          replace=True)]
        minibatch_x = minibatch[:,:-1]
        minibatch_y = minibatch[:,-1]
        if isinstance(x, csr_matrix):
            minibatch_y = np.squeeze(np.asarray(minibatch_y.todense()))


        gradient = loss_gradient(theta, minibatch_x, minibatch_y)

        pos_alphas = compute_alpha(corner_size, gradient, m, noise_para, theta)
        neg_alphas = compute_alpha(-corner_size, gradient, m, noise_para, theta)
        alphas = pos_alphas + neg_alphas
        min_alpha, size, corner_num = min(alphas, key=(lambda x: x[0]))

        corner = np.zeros(m)
        corner[corner_num] = size

        mu = step_size/(i+step_size)
        theta = (1-mu)*theta + mu*corner

    return theta

class PrivateFrankWolfeLR(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None,  # FW does not use GD
                           num_iters=DEFAULT_NUM_ITERS,
                           l2_constraint=100, L=1):
        """Runs frank-wolfe with logistic regression"""

        return private_frank_wolfe(x, y, LogisticRegression.gradient, epsilon,
                                   delta, learning_rate, num_iters,
                                   constraint=l2_constraint, L=L)

    def name():
        return "Private Frank-Wolfe LR"

class PrivateFrankWolfeSVM(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None,  # FW does not use GD
                           num_iters=DEFAULT_NUM_ITERS,
                           l2_constraint=100, L=1):
        """Runs frank-wolfe with logistic regression"""

        return private_frank_wolfe(x, y, HuberSVM.gradient, epsilon,
                                   delta, learning_rate, num_iters,
                                   constraint=l2_constraint, L=L)

    def name():
        return "Private Frank-Wolfe SVM"
