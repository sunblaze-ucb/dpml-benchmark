import numpy as np
import sklearn.datasets
import sklearn.preprocessing

def gen_dataset():
    x, y = sklearn.datasets.make_classification(
        n_samples=10000,
        n_features=40,
        n_informative=30,
        n_classes=2,
        class_sep=2.0,
        random_state=1000)
    
    y = np.array([1 if l == 1 else -1 for l in y])
    return x,y

def gen_dataset_high_dim():
    x, y = sklearn.datasets.make_classification(
        n_samples=2000,
        n_features=2000,
        n_informative=10,
        n_redundant=0,
        n_classes=2,
        class_sep=2.0,
        random_state=1)
    
    y = np.array([1 if l == 1 else -1 for l in y])
    return x,y

def random_projection(ori_matrix, k):
    transformer = sklearn.random_projection.GaussianRandomProjection(n_components=k)
    return transformer.fit_transform(ori_matrix)
