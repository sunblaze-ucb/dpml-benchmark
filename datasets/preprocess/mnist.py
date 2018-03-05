import os
import numpy as np
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output


FILENAME_X = 'mnist_processed_x.npy'
FILENAME_Y = 'mnist_processed_y.npy'


def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    mnist = input_data.read_data_sets(
        os.path.join(cache_location, "MNIST_data"), one_hot=True)

    train_features = np.array(mnist.train.images)
    train_labels = np.array(mnist.train.labels)
    test_features = np.array(mnist.test.images)
    test_labels = np.array(mnist.test.labels)

    features_set = np.vstack((train_features, test_features))
    labels_set = np.vstack((train_labels, test_labels))

    label_width = len(labels_set[0])

    combined_data = np.column_stack([features_set, labels_set])
    np.random.shuffle(combined_data)

    np.save(os.path.join(output_location, FILENAME_X), combined_data[:, :-label_width])
    np.save(os.path.join(output_location, FILENAME_Y), combined_data[:, -label_width:])

