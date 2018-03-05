import numpy as np
from sklearn import preprocessing

def convert_to_binary(symbolic_cols):
    """
    Convert categorial dtata attributes to binary vectors.

    Returns an array of columns of data. These columns will need to be joined
    by a call to np.column_stack() in norder to get the actual data entries
    back.
    """

    one = preprocessing.OneHotEncoder()
    symbolic_features = np.column_stack(symbolic_cols)
    symbolic_features = one.fit_transform(symbolic_features).toarray()

    # Convert back into column vectors
    # I wish there were not necessary
    final_cols = [symbolic_features[:, i]
                  for i in range(symbolic_features.shape[1])]
    return final_cols


def normalize_rows(data):
    """
    Scale all rows by the same factor to ensure that the maximum value of the
    norm of any row is 1
    """

    max_norm = 0
    for row in data:
        max_norm = max(max_norm, np.linalg.norm(row))

    if max_norm > 1:
        divisor = [max_norm for i in range(data[0].shape[0])]
        return data / divisor

    return data


def format_output(output):
    """Preprocesses the output into an array of -1 and 1"""
    filtered = [1 if val == b'normal.' else -1 for val in output]
    return np.array(filtered)