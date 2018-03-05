import os
import numpy as np 
from sklearn import preprocessing
import csv
from scipy.sparse import csr_matrix
from utils.utils_sparse import read_data
from utils.utils_download import download_extract
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/'
FILENAME_X = 'gisette_processed_x.npy'
FILENAME_Y = 'gisette_processed_y.npy'

# preprocess implemented in numpy

def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    download_extract(url, cache_location, 'gisette_train.data')
    download_extract(url, cache_location, 'gisette_train.labels')

    raw_set = csv.reader(open(os.path.join(cache_location, 'gisette_train.data')), delimiter=' ')

    list_set = []
    for row in raw_set:
        list_set.append(row)
    np_set = np.array(list_set)

    continuous_cols = []
    symbolic_cols = []
    label_cols = []
    le = preprocessing.LabelEncoder()
    continuous_pos = range(np_set.shape[1])
    for i in range(np_set.shape[1]-1):
        col = np_set[:, i]
        col = np.array([float(j) for j in col])
        continuous_cols.append(col)

    combined_data = np.column_stack(symbolic_cols+continuous_cols) 

    label_set = csv.reader(open(os.path.join(cache_location, 'gisette_train.labels')), delimiter=' ')

    label_cols = []
    for row in label_set:
        label_cols.append(int(row[0]))

    all_data = np.column_stack([combined_data, label_cols])
    np.random.shuffle(all_data)

    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])

