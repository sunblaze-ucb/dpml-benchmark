import csv
import math
import copy
import random
import numpy as np
from common.common import Algorithm, LEARNING_RATE_CONSTANT, DEFAULT_NUM_ITERS
from common.constraints import constrain_l2_norm
from lossfunctions.logistic_regression import LogisticRegression, LogisticRegressionRegular
from lossfunctions.huber_svm import HuberSVM, HuberSVMRegular
from scipy.sparse import csr_matrix, hstack
import os

def private_gradient_descent_minibatch(x, y, loss_gradient, eps, delta, num_iters,
                                       learning_rate, L = 1, minibatch_size = 50,
                                       l2_constraint = None, lambda_param=0):
        n = x.shape[0]
        m = x.shape[1]
        q = minibatch_size/n

        if l2_constraint == None and lambda_param == 0:
                # we are doing unconstrained, unregularized learning
                L_reg = L
                theta = np.zeros(m)

        elif l2_constraint != None and lambda_param > 0:
                # we are doing constrained, regularized learning
                L_reg = L + lambda_param*l2_constraint
                theta = (np.random.rand(x.shape[1]) - .5) * 2 * l2_constraint

        else:
                # if we are doing regularization with no L2 constraint, give up!
                # i.e. return a model that's just zeroes
                return np.zeros(m)

        std_dev = 4 * L_reg * math.sqrt(num_iters * math.log(1/delta)) / (n * eps)

        if isinstance(x, csr_matrix):
                data = csr_matrix(hstack((x, csr_matrix(y).T)))
        else:
                data = np.column_stack((x, y))
        
        np.random.seed(ord(os.urandom(1)))

        for i in range(num_iters):
                s = data.shape[0]
                minibatch = data[np.random.choice(data.shape[0],
                                                  minibatch_size,
                                                  replace=True)]

                minibatch_x = minibatch[:,:-1]
                minibatch_y = minibatch[:,-1]
                if isinstance(x, csr_matrix):
                        minibatch_y = np.squeeze(np.asarray(minibatch_y.todense()))

                gradient = loss_gradient(theta, minibatch_x, minibatch_y, lambda_param)
                noise = np.random.normal(scale=std_dev, size=m)
                theta = theta - learning_rate * (gradient+noise)
                if l2_constraint is not None:
                        theta = constrain_l2_norm(theta, l2_constraint)
                

        return theta


class PrivateGDLR(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None,
                           num_iters=DEFAULT_NUM_ITERS,
                           minibatch_size=50,
                           l2_constraint=None, L=1):
        """Runs DP_NSGD with logistic regression"""

        if learning_rate is None:
            learning_rate = 1 / x.shape[0]

        return private_gradient_descent_minibatch(
                x, y, LogisticRegressionRegular.gradient, epsilon, delta, num_iters,
                learning_rate, minibatch_size=minibatch_size,
                l2_constraint=l2_constraint, L=L, lambda_param=lambda_param)

    def name():
        return "Private Gradient Descent LR"

class PrivateGDSVM(Algorithm):
    def run_classification(x, y, epsilon, delta, lambda_param,
                           learning_rate=None,
                           num_iters=DEFAULT_NUM_ITERS,
                           minibatch_size=50,
                           l2_constraint=None, L=1):
        """Runs DP_NSGD with logistic regression"""

        if learning_rate is None:
            learning_rate = 1 / x.shape[0]

        return private_gradient_descent_minibatch(
                x, y, HuberSVMRegular.gradient, epsilon, delta, num_iters,
                learning_rate, minibatch_size=minibatch_size,
                l2_constraint=l2_constraint, L=L, lambda_param=lambda_param)

    def name():
        return "Private Gradient Descent SVM"
