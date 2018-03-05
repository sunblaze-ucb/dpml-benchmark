"""Tuning for learning rate and test of convergence"""

from cms11 import ObjectivePerturbation
from common import compute_classification_counts
import numpy as np
import logging
import sys

GRADIENT_GUESS = 0.01
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s')


def find_best_learning_rate(learning_rates, train_x, train_y, test_x, test_y):
    best_learning_rate = None
    best_error_count = float('inf')

    for learning_rate in learning_rates:
        theta = ObjectivePerturbation.run_classification(
            train_x, train_y, 0.1, 10 ** -7, learning_rate,
            GRADIENT_GUESS)

        errors = compute_classification_counts(
            test_x, test_y, theta)[1]

        if errors < best_error_count:
            best_error_count = errors
            best_learning_rate = learning_rate

    logging.info('The best learning rate was %f. The error count was %d.',
                 best_learning_rate, best_error_count)

    return best_learning_rate


def find_best_gradient_norm(learning_rate, gradient_norms, train_x, train_y,
                            test_x, test_y):
    best_gradient_norm = None
    best_error_count = float('inf')
    best_theta = None

    for gradient_norm in gradient_norms:
        logging.info('Trying gradient norm %f', gradient_norm)
        theta = ObjectivePerturbation.run_classification(
            train_x, train_y, 0.1, 10 ** -7, learning_rate, gradient_norm)

        errors = compute_classification_counts(
            test_x, test_y, theta)[1]

        if errors < best_error_count:
            best_error_count = errors
            best_gradient_norm = gradient_norm
            best_theta = theta

    logging.info('The best gradient norm was %f. The error count was %d',
                 best_gradient_norm, best_error_count)

    return best_theta


def main():

    learning_rates = [0.05, 0.01, 0.005, 0.001]
    gradient_norms = [0.05, 0.01, 0.005, 0.001]
    
    x = np.load('iris_processed_x.npy')
    y = np.load('iris_processed_y.npy')

    n = x.shape[0]
    train_size = int(n * .6)
    other_size = int(n * .2)

    logging.info('The training set has %d elements. The test and holdout sets'
                 ' have %d elements.', train_size, other_size)

    test_end = train_size + other_size

    train_x = x[:train_size]
    train_y = y[:train_size]

    test_x = x[train_size:test_end]
    test_y = y[train_size:test_end]

    holdout_x = x[test_end:]
    holdout_y = y[test_end:]

    best_learning_rate = find_best_learning_rate(learning_rates, train_x,
                                                 train_y, test_x, test_y)
    best_theta = find_best_gradient_norm(best_learning_rate,
                                         gradient_norms, train_x,
                                         train_y, test_x, test_y)

    errors_on_holdout = compute_classification_counts(holdout_x, holdout_y,
                                                      best_theta)[1]

    logging.info('There were %d errors when evaluating the holdout set',
                 errors_on_holdout)


if __name__ == '__main__':
    main()
