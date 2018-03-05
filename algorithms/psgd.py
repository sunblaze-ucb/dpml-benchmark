import numpy as np
from common.common import (Algorithm, LEARNING_RATE_CONSTANT,
                           compute_classification_counts, DEFAULT_NUM_ITERS)
from lossfunctions.logistic_regression import (LogisticRegression,
                                               LogisticRegressionRegular)
from lossfunctions.huber_svm import HuberSVM, HuberSVMRegular
from common.constraints import constrain_l2_norm
from common.noise import compute_gamma_noise
import logging
from scipy.sparse import csr_matrix
import os

# Suggested in section 4.3
B = 50
CONVERGENCE_THRESHOLD = 0.01

def psgd_minibatched(x, y, loss_gradient, b, learning_rate_function,
                     num_iters, stop_early=False, sparse=False,
                     l2_constraint=None, lambda_param=None):
    """Mini-batched version of psgd"""

    n = x.shape[0]
    if n == 0:
        raise Exception("No training Data")

    tau = np.random.permutation(n)
    batches_x = None
    if sparse:
        batches_x = [x[tau[i: min(i+b,n)]] for i in range(0,n,b)]
    else:
        batches_x = [
        np.array([x[tau[j]] for j in range(i, min(i + b, n))])
        for i in range(0, n, b)
        ]

    batches_y = [
        np.array([y[tau[j]] for j in range(i, min(i + b, n))])
        for i in range(0, n, b)
    ]

    if l2_constraint is None:
        theta = np.zeros(shape=x.shape[1])

        # uncomment this line to start with a random theta
        # theta = (np.random.rand(x.shape[1]) - .5) * 10
    else:
        theta = (np.random.rand(x.shape[1]) - .5) * 2 * l2_constraint

    t = 1
    for i in range(num_iters):
        for j in range(len(batches_x)):
            batch_x = batches_x[j]
            batch_y = batches_y[j]

            theta = theta - learning_rate_function(t) * loss_gradient(
                theta, batch_x, batch_y, lambda_param=lambda_param)

            if l2_constraint is not None:
                theta = constrain_l2_norm(theta, l2_constraint)

            t += 1

    return theta

def private_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                        loss_gradient, b, L,
                        sparse=False, l2_constraint=None,
                        lr_type='constant'):

    # Suggested in section 4.3
    # step_size = 1 / np.sqrt(x.shape[0])

    learning_rate_function = None

    beta = L*L
    m = x.shape[0]
    c = learning_rate
    k = num_iters

    def constant_learning_rate_function(t):
        return learning_rate

    def decreasing_learning_rate_function(t):
        return 2 / (beta * (t + (m**c)))

    def sqrt_learning_rate_function(t):
        return 2 / (beta * (np.sqrt(t) + (m**c)))

    if lr_type == 'constant':
        l2_sensitivity = 2 * k * L * learning_rate / b
        learning_rate_function = constant_learning_rate_function
    elif lr_type == 'decreasing':
        l2_sensitivity = (4 * L / (beta*b)) * sum([1 / (((j * m) / b) + 1 + (m**c))
                                                   for j in range(k)])
        learning_rate_function = decreasing_learning_rate_function
    elif lr_type == 'sqrt':
        l2_sensitivity = (4 * L / (beta*b)) * sum([1 / (np.sqrt((j*m/b) + 1) + (m**c))
                                                   for j in range(k)])
        learning_rate_function = sqrt_learning_rate_function

    w = psgd_minibatched(x, y, loss_gradient, b, learning_rate_function,
                         num_iters, sparse=sparse, l2_constraint=l2_constraint)

    np.random.seed(ord(os.urandom(1)))
    std_dev = np.sqrt(2*np.log(2 / delta)) * l2_sensitivity / epsilon

    noise = np.random.normal(scale=std_dev, size=x.shape[1])
    theta_priv = w + noise

    if l2_constraint is not None:
        theta_priv = constrain_l2_norm(theta_priv, l2_constraint)

    return theta_priv

def private_strongly_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                                 loss_gradient, b, L,
                                 sparse=False, l2_constraint=None):

    # This is the setting from the paper
    # We tune it instead
    # R = min(1 / lambda_param, l2_constraint)
    
    R = l2_constraint
    L_reg = L + lambda_param * R
    beta = L*L + lambda_param
    gamma = lambda_param
    m = x.shape[0]

    l2_sensitivity = 2 * L_reg / (gamma * m)

    def regularized_learning_rate_function(t):
        return min(1/beta, 1/(gamma*t))

    learning_rate_function = regularized_learning_rate_function


    w = psgd_minibatched(x, y, loss_gradient, b, learning_rate_function,
                         num_iters, sparse=sparse, l2_constraint=l2_constraint, lambda_param=lambda_param)

    np.random.seed(ord(os.urandom(1)))
    std_dev = np.sqrt(2*np.log(2 / delta)) * l2_sensitivity / epsilon

    noise = np.random.normal(scale=std_dev, size=x.shape[1])
    theta_priv = w + noise

    if l2_constraint is not None:
        theta_priv = constrain_l2_norm(theta_priv, l2_constraint)

    return theta_priv


class PrivateConvexPSGDLR(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=DEFAULT_NUM_ITERS, b=B,
                           sparse=False, l2_constraint=None,
                           lr_type='constant', L=1):

        return private_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                                   LogisticRegression.gradient, b, L, sparse,
                                   l2_constraint=l2_constraint,
                                   lr_type=lr_type)

    def name():
        return ("Private Convex Permutation-Based Stochastic Gradient Descent"
                "LR")

class PrivateConvexPSGDSVM(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=DEFAULT_NUM_ITERS, b=B,
                           sparse=False, l2_constraint=None,
                           lr_type='constant', L=1):

        return private_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                                   HuberSVM.gradient, b, L, sparse,
                                   l2_constraint=l2_constraint,
                                   lr_type=lr_type)

    def name():
        return ("Private Convex Permutation-Based Stochastic Gradient Descent "
                "LR")


class PrivateStronglyConvexPSGDLR(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=DEFAULT_NUM_ITERS, b=B,
                           sparse=False, l2_constraint=None,
                           L=1):

        return private_strongly_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                                            LogisticRegressionRegular.gradient, b, L, sparse,
                                            l2_constraint=l2_constraint)

    def name():
        return ("Private Strongly Convex Permutation-Based Stochastic Gradient Descent"
                "LR")

class PrivateStronglyConvexPSGDSVM(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None, num_iters=DEFAULT_NUM_ITERS, b=B,
                           sparse=False, l2_constraint=None,
                           L=1):

        return private_strongly_convex_psgd(x, y, learning_rate, lambda_param, num_iters, epsilon, delta,
                                            HuberSVMRegular.gradient, b, L, sparse,
                                            l2_constraint=l2_constraint)

    def name():
        return ("Private Strongly Convex PSGD SVM")
