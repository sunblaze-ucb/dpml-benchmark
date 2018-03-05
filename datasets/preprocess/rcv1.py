import os
import numpy as np
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import GaussianRandomProjection
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output


# how many samples to process
SIZE = 500000

# how many samples to save to disk (after shuffling)
SHRUNK_SIZE = 50000

FILENAME_D = 'rcv1_processed_d.npy'
FILENAME_Y = 'rcv1_processed_y.npy'
FILENAME_INDICES = 'rcv1_processed_indices.npy'
FILENAME_INDPTR = 'rcv1_processed_indptr.npy'

def mk_label(n):
    if n == 0:
        return -1
    else:
        return 1

def classify(features, labels):
    training_size = int(features.shape[0] * 0.8)

    training_features = features[:training_size]
    testing_features = features[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    classifier = LogisticRegression()
    classifier.fit(training_features, training_labels)

    predicted_labels = classifier.predict(testing_features)

    eq = np.equal(testing_labels, predicted_labels)
    eq = eq.astype(float)

    accuracy = np.mean(eq)
    print("Scikit-learn classifier got accuracy {0}".format(accuracy))

def random_projection(ori_matrix, k):
    transformer = GaussianRandomProjection(n_components=k)
    return transformer.fit_transform(ori_matrix)
    
def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    print("Fetching RCV1 dataset")
    rcv1 = fetch_rcv1()

    print("Shape of the data:", rcv1.data.shape)

    print("Index of CCAT:", rcv1.target_names.tolist().index("CCAT"))

    # get the first SIZE samples
    features = rcv1.data[:SIZE]
    categories = rcv1.target[:SIZE]

    # convert labels to 1, -1
    # our classification is binary: in/out of class 33
    print("Converting labels")
    labels = np.array([mk_label(row.toarray()[0,33]) for row in categories])

    # test the sklearn classifier
    classify(features, labels)

    # shuffle the dataset
    print("Shuffling dataset")
    index = np.arange(np.shape(features)[0])
    np.random.shuffle(index)
    features = features[index, :]
    labels = labels[index]

    classify(features, labels)
    classify(features, labels)
    
    # shrink the dataset
    print("Shrinking to size")
    features = features[:SHRUNK_SIZE]
    labels = labels[:SHRUNK_SIZE]

    classify(features, labels)

    # save the dataset
    print("Saving")
    np.save(os.path.join(output_location, FILENAME_D), features.data)
    np.save(os.path.join(output_location, FILENAME_INDICES), features.indices)
    np.save(os.path.join(output_location, FILENAME_INDPTR), features.indptr)
    np.save(os.path.join(output_location, FILENAME_Y), labels)

    # print statistics
    print("Shape of the data is:", features.shape)
