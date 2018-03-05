import os
import numpy as np
import sklearn.datasets
from sklearn import preprocessing
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output



FILENAME_X = 'kddcup99_processed_x.npy'
FILENAME_Y = 'kddcup99_processed_y.npy'


"""
Preprocess the kddcup99 dataset

This process is discussed in section 7.1 of CMS11:
\"For the KDDCup99 data set, the instances were preprocessed by converting
each categorial attribute to a binary vector. Each column was normalized
to ensure that the maximum value is 1, and finally, each row was normalized,
to ensure that the norm of any example is at most 1. After preprocessing,
each example was represented by a 119-dimensional vector, of norm at most 1.\"

My code to preprocess the categorial features was somewhat based upon:
https://biggyani.blogspot.com/2014/08/using-onehot-with-categorical.html

The preprecessed data gets saved to a file hardcoded into this script.
"""


def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    os.environ['SCIKIT_LEARN_DATA'] = cache_location

    subset = sklearn.datasets.fetch_kddcup99(percent10=True)

    # Randomly select 70,000 elements
    # Based on https://stackoverflow.com/a/14262743/859277

    indices = np.random.randint(subset['data'].shape[0], size=70000)

    subset['data'] = subset['data'][indices, :]
    subset['target'] = subset['target'][indices]

    symbolic_cols = []
    continuous_cols = []
    le = preprocessing.LabelEncoder()

    for i in range(subset['data'].shape[1]):
        col = subset['data'][:, i]

        if type(col[0]) == bytes:
            numeric = le.fit_transform(col)
            symbolic_cols.append(numeric)
        else:
            continuous_cols.append(col)

    symbolic_cols = convert_to_binary(symbolic_cols)

    combined_data = np.column_stack(symbolic_cols + continuous_cols)
    final_data = combined_data

    all_data = np.column_stack([final_data, format_output(subset['target'])])
    np.random.shuffle(all_data)


    print(all_data[:, -1])
    print(len(all_data))
    print(len(all_data[0]))
    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])
