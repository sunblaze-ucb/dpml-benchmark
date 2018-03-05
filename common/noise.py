import numpy as np


def compute_gamma_noise(priv_param, data_dimensions, max_norm):
    gamma_r = np.random.gamma(data_dimensions, max_norm / priv_param)
    spherical_random_vec = np.random.normal(0, 1.0, data_dimensions)
    return (gamma_r*spherical_random_vec) / (
        np.linalg.norm(spherical_random_vec))
