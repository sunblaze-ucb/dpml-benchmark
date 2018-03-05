import numpy as np
from scipy.sparse import csr_matrix

DEFAULT_H = 0.1


class HuberSVM():
    @staticmethod
    def loss(theta, x, y, lambda_param=None, h=DEFAULT_H):
        """Loss function for Huber SVM without regularization"""

        z = y * x.dot(theta)

        losses = np.zeros(z.shape)
        high_inds = z < 1 - h  # since we want 1 - z > h
        losses[high_inds] = 1 - z[high_inds]

        low_inds = z > 1 + h  # since we want 1 - z < -h
        mid_inds = ~(high_inds+low_inds)
        num = 1 - z[mid_inds]
        losses[mid_inds] = pow(num,2)/(4*h) + num/2 + h/4
        loss = np.sum(losses)/x.shape[0]

        return loss

    @staticmethod
    def gradient(theta, x, y, lambda_param=None, h=DEFAULT_H):
        """
        Gradient function for Huber SVM without regularization
        Based on the above Huber SVM
        """

        z = y * x.dot(theta)

        high_inds = z < 1 - h  # since we want 1 - z > h
        low_inds = z > 1 + h  # since we want 1 - z < -h
        mid_inds = ~(high_inds+low_inds)
        num = 1 - z[mid_inds]

        grads = np.zeros((x.shape[0],x.shape[1]))

        if np.sum(high_inds) > 0:
          x_result = x[np.where(high_inds==1)[0],:]
          if isinstance(x_result, csr_matrix):
              x_result = x_result.toarray()
              
          grads[np.where(high_inds==1)[0],:] = (-1 
                                                * y[high_inds].reshape(
                                                    (y[high_inds].shape[0]),1)
                                                * x_result)
        if np.sum(mid_inds) > 0:
          x_result = x[np.where(mid_inds==1)[0],:]
          if isinstance(x_result, csr_matrix):
              x_result = x_result.toarray()
              
          grads[np.where(mid_inds==1)[0],:] = ((-1 
                                                * y[mid_inds].reshape(
                                                    (y[mid_inds].shape[0]),1)
                                                * x_result)
                                               * (num/(2*h) + 0.5).reshape(
                                                   (y[mid_inds].shape[0]),1))
        grad = np.mean(grads, 0)

        return grad


class HuberSVMRegular():
    @staticmethod
    def loss(theta, x, y, lambda_param, h=DEFAULT_H):
        regularization = (lambda_param/2) * np.sum(theta*theta)
        return HuberSVM.loss(theta, x, y, h) + regularization

    @staticmethod
    def gradient(theta, x, y, lambda_param, h=DEFAULT_H):
        regularization = lambda_param * theta
        return HuberSVM.gradient(theta, x, y, h) + regularization
