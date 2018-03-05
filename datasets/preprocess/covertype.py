import os
import csv
import zipfile
import numpy as np
import sklearn.datasets
from sklearn import preprocessing
from utils.utils_download import download_extract
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/'
FILENAME_X = 'covertype_processed_x.npy'
FILENAME_Y = 'covertype_processed_y.npy'

# preprocess implemented in numpy


def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    download_extract(url, cache_location, 'covtype.data.gz', 'gz', 'covtype.data')

    raw_set = csv.reader(
        open(os.path.join(cache_location, 'covtype.data')), delimiter=',')

    list_set = []
    for row in raw_set:
        if row[0] != bytes:
            list_set.append(row)

    np_set = np.array(list_set)

    continuous_cols = []
    label_cols = []
    le = preprocessing.LabelEncoder()
    for i in range(np.shape(np_set)[1]):
        col = np_set[:, i]
        if i != np.shape(np_set)[1]-1:
            col = np.array([float(j) for j in col])
            if col.max() > 1:
                col = col/col.max()
            continuous_cols.append(col)
        else:
            col = np.array([int(j) for j in col])
            for j in range(col.shape[0]):
                tmp = np.zeros(7)
                tmp[col[j]-1] = 1
                label_cols.append(tmp.tolist())

    combined_data = np.column_stack([]+continuous_cols)
    final_data = combined_data

    label_width = len(label_cols[0])
    all_data = np.column_stack([final_data, label_cols])
    np.random.shuffle(all_data)
    
    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-label_width])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -label_width:])
