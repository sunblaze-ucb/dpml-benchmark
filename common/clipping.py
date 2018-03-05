import os
import sys
import csv
import math
import numpy as np
from scipy.sparse import csr_matrix, hstack

def get_norm(data, ord=2):
    if isinstance(data, csr_matrix):
        if ord == np.inf:
            return max(data.data)
        else:
            return np.linalg.norm(data.data, ord=ord)
    else:
        return np.linalg.norm(data, ord=ord)


def get_max_norm(data, ord=2):
    if isinstance(data, csr_matrix):
        if ord == np.inf:
            norms = data.data
        else:
            norms = [np.linalg.norm(data.getrow(row_num).data, ord=ord) for row_num in range(data.shape[0])]
    else:
        norms = np.linalg.norm(data.data, axis=1, ord=ord)

    return np.max(norms)

def clip_rows(data, ord=2, L=1):
    """
    Scale clip rows according the same factor to ensure that the maximum value of the
    norm of any row is L
    """
    max_norm = get_max_norm(data, ord=ord)
    print("For order {0}, max norm is {1}".format(ord, max_norm))

    normalized_data = data.copy()

    modified = 0
    for i in range(data.shape[0]):
        norm = get_norm(data[i], ord)

        if norm > L:
            modified += 1
            normalized_data[i] = L * normalized_data[i] / norm

    print("For order {0}, final max norm is {1}"
          .format(ord, get_max_norm(normalized_data, ord=ord)))
    print("Had to modify {0} rows ({1}% of total)"
          .format(modified, 100*modified / data.shape[0]))
    
    return normalized_data

def clip_rows_l1(data, L1_L=1):
    print("For order {0}, initial max norm is {1}"
          .format(np.inf, get_max_norm(data, ord=np.inf)))

    normalized_data = data.copy()

    if isinstance(data, csr_matrix):
        normalized_data.data = np.clip(normalized_data.data, -L1_L, L1_L)
    else:
        normalized_data = np.clip(normalized_data.data, -L1_L, L1_L)
        
    print("For order {0}, final max norm is {1}"
          .format(np.inf, get_max_norm(normalized_data, ord=np.inf)))
    return normalized_data


