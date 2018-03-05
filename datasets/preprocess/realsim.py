import os
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from utils.utils_sparse import read_matrix
from utils.utils_download import download_extract

url = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
FILENAME_D = 'realsim_processed_d.npy'
FILENAME_INDICES = 'realsim_processed_indices.npy'
FILENAME_INDPTR = 'realsim_processed_indptr.npy'
FILENAME_Y = 'realsim_processed_y.npy'

# preprocess implemented in numpy

def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    download_extract(url, cache_location, 'real-sim.bz2', 'bz2', 'real-sim')

    data, row, col, label_cols = read_matrix(os.path.join(cache_location, "real-sim"))
    csr = csr_matrix((data, (row, col)), shape=(72309, 20958))

    index = np.arange(np.shape(csr)[0])
    np.random.shuffle(index)
    csr = csr[index, :]
    label_cols = label_cols[index]

    np.save(os.path.join(output_location, FILENAME_D), data)
    np.save(os.path.join(output_location, FILENAME_INDICES), csr.indices)
    np.save(os.path.join(output_location, FILENAME_INDPTR), csr.indptr)
    np.save(os.path.join(output_location, FILENAME_Y), label_cols)



