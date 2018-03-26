import os
import sys
import csv
import math
import numpy as np
from scipy.sparse import csr_matrix, hstack
from algorithms.gradient_descent import PrivateGDLR, PrivateGDSVM
from algorithms.psgd import PrivateConvexPSGDLR, PrivateConvexPSGDSVM, PrivateStronglyConvexPSGDLR, PrivateStronglyConvexPSGDSVM
from algorithms.frank_wolfe import (
    PrivateFrankWolfeLR, PrivateFrankWolfeSVM)
from algorithms.approximate_minima_perturbation import ApproximateMinimaPerturbationLR, ApproximateMinimaPerturbationSVM
from common.common import compute_classification_counts, compute_multiclass_counts
from common.clipping import clip_rows, clip_rows_l1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import clock
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool, Value, Array, Manager
from itertools import product
from copy import deepcopy
from common.datasets import gen_dataset, gen_dataset_high_dim
from decimal import Decimal

# To run this script, please input command line
# python gridsearch.py [alg_name] [dataset_name] [random_proj] [SVM] [eps_list...]
# To get the best performance, first run this:
# export OMP_NUM_THREADS=1
dataset_location = './datasets/data'
multivariate_datasets = ['covertype', 'mnist', 'o185', 'o313', 'o4550', 'PEMS', 'wine']
sparse_datasets = ['farm', 'dexter', 'dorothea', 'realsim', 'rcv1', 'news20']
data2shape = {'farm':(4143, 54877), 'dexter':(300, 20000), 'dorothea':(800, 100000), 'realsim':(72309, 20958), 'rcv1':(50000, 47236), 'news20':(8870, 117049)}

# How many times to repeat each experiment
NUM_REPEATS=10

# How many cores to use
CORES = 40

# The default clipping factor to use
L = 1
L1_L = 1

# Epsilons to test
all_eps_list = [0.01, 0.0316227766017, 0.1, 0.316227766017, 1, 3.16227766017, 10]

def build_binary_ys(vec_ys):
    binary_ys = []
    for i in range(vec_ys.shape[1]):
        binary_ys.append(np.array([1 if y == 1 else -1 for y in vec_ys[:, i]]))
    return binary_ys

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))

def progress_bar(pct):
    i = int(pct)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(i/5), i))
    sys.stdout.flush()

def approximate_minima_perturbation(training_features, training_labels, eps, delta, hyper, model, counter, total_configurations):
    if model == 'LR':
        theta, gamma =  ApproximateMinimaPerturbationLR.run_classification(training_features, training_labels, eps, delta, 
                                            hyper['lambda_param'],
                                            hyper['learning_rate'],
                                            hyper['iterations'],
                                            hyper['l2_constraint'],
                                            hyper['eps_frac'],
                                            hyper['eps_out_frac'],
                                            hyper['gamma'],
                                            hyper['L'])
    else:
        theta, gamma =  ApproximateMinimaPerturbationSVM.run_classification(training_features, training_labels, eps, delta, 
                                            hyper['lambda_param'],
                                            hyper['learning_rate'],
                                            hyper['iterations'],
                                            hyper['l2_constraint'],
                                            hyper['eps_frac'],
                                            hyper['eps_out_frac'],
                                            hyper['gamma'],
                                            hyper['L'])
    counter.append(0)
    progress_bar(len(counter)*100/total_configurations)
    return theta, hyper['L'], gamma

def private_gd(training_features, training_labels, eps, delta, hyper, model, counter, total_configurations):
    if model == 'LR':
        theta = PrivateGDLR.run_classification(training_features, training_labels, eps, delta,
                                                hyper['lambda_param'],
                                                hyper['learning_rate'],
                                                hyper['iterations'],
                                                hyper['minibatch_size'],
                                                hyper['l2_constraint'],
                                                hyper['L'])
    else:
        theta = PrivateGDSVM.run_classification(training_features, training_labels, eps, delta, 
                                                hyper['lambda_param'],
                                                hyper['learning_rate'],
                                                hyper['iterations'],
                                                hyper['minibatch_size'],
                                                hyper['l2_constraint'],
                                                hyper['L'])
    counter.append(0)
    progress_bar(len(counter)*100/total_configurations)
    return theta, hyper['L'], 0

def convex_psgd(training_features, training_labels, eps, delta, hyper, model, counter, total_configurations):
    if model == 'LR':
        theta = PrivateConvexPSGDLR.run_classification(training_features, training_labels, eps, delta,
                                                       hyper['lambda_param'],
                                                       hyper['learning_rate'],
                                                       hyper['iterations'],
                                                       hyper['b'],
                                                       hyper['sparse'],
                                                       hyper['l2_constraint'],
                                                       hyper['arg'],
                                                       hyper['L'])
    else:
        theta = PrivateConvexPSGDSVM.run_classification(training_features, training_labels, eps, delta,
                                                        hyper['lambda_param'],
                                                        hyper['learning_rate'],
                                                        hyper['iterations'],
                                                        hyper['b'],
                                                        hyper['sparse'],
                                                        hyper['l2_constraint'],
                                                        hyper['arg'],
                                                        hyper['L'])
    counter.append(0)
    progress_bar(len(counter)*100/total_configurations)
    return theta, hyper['L'], 0

def stronglyconvex_psgd(training_features, training_labels, eps, delta, hyper, model, counter, total_configurations):
    if model == 'LR':
        theta = PrivateStronglyConvexPSGDLR.run_classification(training_features, training_labels, eps, delta,
                                                               hyper['lambda_param'],
                                                               hyper['learning_rate'],
                                                               hyper['iterations'],
                                                               hyper['b'],
                                                               hyper['sparse'],
                                                               hyper['l2_constraint'],
                                                               hyper['L'])
    else:
        theta = PrivateStronglyConvexPSGDSVM.run_classification(training_features, training_labels, eps, delta,
                                                                hyper['lambda_param'],
                                                                hyper['learning_rate'],
                                                                hyper['iterations'],
                                                                hyper['b'],
                                                                hyper['sparse'],
                                                                hyper['l2_constraint'],
                                                                hyper['L'])
    counter.append(0)
    progress_bar(len(counter)*100/total_configurations)
    return theta, hyper['L'], 0

def private_frankwolfe(training_features, training_labels, eps, delta, hyper, model, counter, total_configurations):
    if model == 'LR':
        theta = PrivateFrankWolfeLR.run_classification(training_features, training_labels, eps, delta,
                                                       hyper['lambda_param'],
                                                       hyper['learning_rate'],
                                                       hyper['iterations'],
                                                       hyper['l1_constraint'],
                                                       hyper['L'])
    else:
        theta = PrivateFrankWolfeSVM.run_classification(training_features, training_labels, eps, delta,
                                                        hyper['lambda_param'],
                                                        hyper['learning_rate'],
                                                        hyper['iterations'],
                                                        hyper['l1_constraint'],
                                                        hyper['L'])
    counter.append(0)
    progress_bar(len(counter)*100/total_configurations)
    return theta, hyper['L'], 0

def create_directory(directory_name):
    try:
      os.stat(directory_name)
    except:
      os.mkdir(directory_name)


def main():

    print("Starting...")
    np.seterr(over='ignore')

    create_directory("./results")
    create_directory("./results/rough_results")
    create_directory("./results/rough_results/LR")
    create_directory("./results/rough_results/SVM")
    create_directory("./results/graphs")
    create_directory("./results/graphs/LR")
    create_directory("./results/graphs/SVM")

    if len(sys.argv)<4:
        print('Usage: python release.py <algorithm> <dataset> <loss=LR|SVM>')
        sys.exit(1)
    else:
        alg_name = sys.argv[1]
        dataset_name = sys.argv[2]
        model_name = sys.argv[3]

    print("Loading Dataset...")

    result_location = './results/rough_results/' + model_name

    if dataset_name == 'random':
        print("Random dataset...")
        features, labels = gen_dataset()
    elif dataset_name == 'random_highdim':
        print("Random high-dimensional dataset...")
        features, labels = gen_dataset_high_dim()
    elif dataset_name not in sparse_datasets:
        features = np.load(
            os.path.join(
                dataset_location, '{}_processed_x.npy'.format(dataset_name)))
        features = features.astype(float)
        labels = np.load(os.path.join(dataset_location, '{}_processed_y.npy'.format(dataset_name)))
        labels = labels.astype(float)
    else:
        data = np.load(os.path.join(dataset_location, '{}_processed_d.npy'.format(dataset_name)))
        indices = np.load(os.path.join(dataset_location, '{}_processed_indices.npy'.format(dataset_name)))
        indptr = np.load(os.path.join(dataset_location, '{}_processed_indptr.npy'.format(dataset_name)))
        features = csr_matrix((data, indices, indptr), shape=data2shape[dataset_name])
        labels = np.load(os.path.join(dataset_location, '{}_processed_y.npy'.format(dataset_name)))
        labels = labels.astype(float)

    training_size = int(features.shape[0] * 0.8)
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]
    labels_ = []
    if dataset_name in multivariate_datasets:
        for row in labels:
            for i in range(len(row)):
                if row[i] == 1:
                    labels_.append(i)
    else:
        labels_ = labels
    training_labels_ = labels_[:training_size]
    testing_labels_ = labels_[training_size:]

    unnormalized_features = deepcopy(features)

    print("Loaded dataset")

    # SKLEARN, un-normalized
    print("Training scikit-learn classifier on un-normalized data")
    training_features = features[:training_size]
    testing_features = features[training_size:]    
    classifier = LogisticRegression()
    classifier.fit(training_features, training_labels_)
    predicted_labels = classifier.predict(testing_features)
    eq = np.equal(testing_labels_, predicted_labels)
    eq = eq.astype(float)
    accuracy = np.mean(eq)
    print("Scikit-learn classifier got accuracy {0}".format(accuracy))
    theta = np.squeeze(np.asarray(classifier.coef_))
    print("L2 Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=2)))
    print("L1 Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=1)))
    print("Linf Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=np.inf)))

    if len(sys.argv) > 4:
        accfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'_'+sys.argv[4]+'.acc'), 'w')
        stdfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'_'+sys.argv[4]+'.std'), 'w')
        logfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'_'+sys.argv[4]+'.log'), 'w')
    else:
        accfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'.acc'), 'w')
        stdfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'.std'), 'w')
        logfile = open(os.path.join(result_location, alg_name+'_'+dataset_name+'.log'), 'w')

    acc_matrix = np.zeros([8, 9])
    std_matrix = np.zeros([8, 9])
    acc_matrix[0] = accuracy*np.ones(9)
    n = training_size

    algorithm_names = ['AMP', 'AMP-NT', 'PSGD', 'PPSGD', 'PPSSGD', 'FW']
    algorithms = {
        'AMP': {
            'fun': approximate_minima_perturbation, 
            'hyper': {
                'lambda_param': [None],
                'learning_rate': [None],
                'iterations': [None],
                'l2_constraint': [None],
                'eps_frac':[.9, .95, .98, .99],
                'eps_out_frac':[.001, .01, .1, .5],
                'gamma': [1/(n**2)],
                'L': [.1, 1, 10]
            }
        },
        'AMP-NT': {
            'fun': approximate_minima_perturbation, 
            'hyper': {
                'lambda_param': [None],
                'learning_rate': [None],
                'iterations': [None],
                'l2_constraint': [None],
                'eps_frac':[None],
                'eps_out_frac':[.01],
                'gamma': [1/(n**2)],
                'L': [1]
            }
        },
        'PSGD': {
            'fun': private_gd,
            'hyper':{
                'lambda_param':[0.0001, 0.001, 0],
                'learning_rate':[0.1, 1, 10],
                'iterations':[100, 500, 1000],
                'minibatch_size':[50],
                'l2_constraint': [None, 1, 10, 100],
                'L': [.1, 1, 10]
            }
        },
        'PPSGD': {
            'fun': convex_psgd,
            
            'hyper':{
                'lambda_param': [0.0001, 0.001],
                'learning_rate':[0.001, 0.01, 0.1],
                'iterations':[5, 10, 100],
                'b': [50],
                'sparse': [(dataset_name in sparse_datasets)],
                'l2_constraint': [None],
                'arg': ['constant'],
                'L': [.1, 1, 10]
            }
            
        },
        'PPSSGD': {
            'fun': stronglyconvex_psgd, 
            'hyper':{
                'lambda_param':[0.0001, 0.001],
                'learning_rate': [1],
                'iterations':[200],
                'b':[50],
                'sparse':[(dataset_name in sparse_datasets)],
                'l2_constraint':[1,10,100],
                'L': [.1, 1, 10]
            }
        },
        'FW': {
            'fun': private_frankwolfe, 
            'hyper':{
                'lambda_param':[0],
                'learning_rate':[1],
                'iterations':[1000, 500, 100, 50, 20, 10, 5],
                'l1_constraint':[1,10,100, 500],
                'L':[1, 0.1, 0.01]
            }
        }
    }

    alg_name_list = [alg_name]
    if alg_name == 'ALL':
        alg_name_list = ['AMP', 'AMP-NT', 'PSGD', 'PPSGD', 'PPSSGD', 'FW']
    datasets_l = {'AMP':{},'AMP-NT':{},'PSGD':{},'PPSGD':{},'PPSSGD':{},'FW':{}}

    for alg_name in alg_name_list:
        for L in algorithms[alg_name]['hyper']['L']:
            datasets_l[alg_name][L] = {}
            if alg_name == 'FW':
                features = clip_rows_l1(unnormalized_features, L) 
            else:
                features = clip_rows(unnormalized_features, 2, L)
            datasets_l[alg_name][L]['training'] = features[:training_size]
            datasets_l[alg_name][L]['testing'] = features[training_size:]

    for alg_name in alg_name_list:
        datasets = datasets_l[alg_name]
        for L, dataset in datasets.items():
            training_features = dataset['training']
            testing_features = dataset['testing']
            print('Training scikit-learn classifier on alg:{0}, L:{1}'.format(alg_name, L))
            classifier = LogisticRegression()
            classifier.fit(training_features, training_labels_)
            predicted_labels = classifier.predict(testing_features)
            eq = np.equal(testing_labels_, predicted_labels)
            eq = eq.astype(float)
            accuracy = np.mean(eq)
            print("Scikit-learn classifier got accuracy {0}".format(accuracy))
            theta = np.squeeze(np.asarray(classifier.coef_))
            print("L2 Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=2)))
            print("L1 Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=1)))
            print("Linf Norm of sklearn theta: {0}".format(np.linalg.norm(theta, ord=np.inf)))

    eps_list = []
    if len(sys.argv)>4:
       for i in range(4, len(sys.argv)):
           eps_list.append(float(sys.argv[i]))
    else:
        eps_list = all_eps_list
    delta = 1/(training_size**2)
    repeat_time = NUM_REPEATS

    for alg_name in alg_name_list:
        alg = algorithms[alg_name]['fun']
        hypers_  = algorithms[alg_name]['hyper']
        hypers = list(dict_product(hypers_))

        manager = Manager()
        counter = manager.list([])
        pool = Pool(CORES)
        result = []

        total_configurations = len(eps_list)*len(hypers)*repeat_time
        if dataset_name in multivariate_datasets:
            total_configurations = total_configurations*training_labels.shape[1]
        print('Running '+alg_name)
        start = clock()
        for eps in eps_list:
            for hyper in hypers:
                for time in range(repeat_time):
                    if dataset_name in multivariate_datasets:
                        train_ys = build_binary_ys(training_labels)
                        thetas = np.zeros(shape=(training_labels.shape[1], training_features.shape[1]))
                        for i, binary_train_y in enumerate(train_ys):
                            args = [datasets_l[alg_name][hyper['L']]['training'], binary_train_y, eps/training_labels.shape[1], delta/training_labels.shape[1], hyper, model_name, counter, total_configurations]
                            result.append(pool.apply_async(alg, args))
                    else:
                        args = [datasets_l[alg_name][hyper['L']]['training'], training_labels, eps, delta, hyper, model_name, counter, total_configurations]
                        result.append(pool.apply_async(alg, args))
        end = clock()

        results = np.array([res.get() for res in result])
        pool.close()
        pool.join()
        print()

        pool = Pool(CORES)
        correct_list = []
        result = []
        if dataset_name in multivariate_datasets:
            thetas, Ls, gammas = zip(*results)
            thetas = np.array(list(thetas))
            Ls = np.array(list(Ls))
            gammas = np.array(list(gammas))

            thetas_len = len(train_ys)
            thetas = thetas.reshape([-1, len(train_ys), len(thetas[0])])

            gammas = gammas.reshape([-1, len(train_ys)])
            gammas = np.average(gammas, axis=1)

            Ls = Ls[::thetas_len]

            results = list(zip(thetas, Ls, gammas))
            for theta, L, gamma in results:
                args = [datasets_l[alg_name][L]['testing'], testing_labels, theta]
                result.append(pool.apply_async(compute_multiclass_counts, args))
                
        else:
            for theta, L, gamma in results:
                args = [datasets_l[alg_name][L]['testing'], testing_labels, theta]
                result.append(pool.apply_async(compute_classification_counts, args))
                
        correct_incorrect_counts = np.array([res.get() for res in result])
        pool.close()
        pool.join()
        
        accuracy_list = np.array([correct/(correct+incorrect) for correct, incorrect in correct_incorrect_counts])
        gamma_list = np.array([gamma for theta, L, gamma in results]).reshape([len(eps_list), -1, repeat_time])

        correct_list = np.array(accuracy_list).reshape([len(eps_list), -1, repeat_time])

        ave_list = np.average(correct_list, axis=2)
        std_list = np.std(correct_list, axis=2)
        gamma_list = np.average(gamma_list, axis=2)

        combined_list_ = list(zip(ave_list, std_list))
        combined_list = [list(zip(i, j)) for i, j in combined_list_]
        max_correct_list = [max(i, key=(lambda x: x[0])) for i in combined_list]
        print('eps', end='')
        print('eps', end='', file=logfile)

        hyperparameter_names = sorted(list(hypers[0].keys()), key=str.lower)

        for name in hyperparameter_names:
            print('\t{0}'.format(name[:3]), end='')
            print('\t{0}'.format(name[:3]), end='', file=logfile)
        print('\tave\tstd\tgamma')
        print('\tave\tstd\tgamma', file=logfile)
        for i, eps in enumerate(eps_list):
            for j, hyper in enumerate(hypers):
                print('{:.2f}'.format(eps), end='')
                print('{:.2f}'.format(eps), end='', file=logfile)
                for name in hyperparameter_names:
                    if name == 'gamma' and hyper[name] != None:
                        print('\t{:.2e}'.format(Decimal(hyper[name])), end='')
                        print('\t{:.2e}'.format(Decimal(hyper[name])), end='', file=logfile)
                    else:
                        print('\t{0}'.format(hyper[name]), end='')
                        print('\t{0}'.format(hyper[name]), end='', file=logfile)
                print('\t{:.3f}\t{:.3f}\t{:.3e}'.format(ave_list[i, j], std_list[i, j], gamma_list[i,j]))
                print('\t{:.3f}\t{:.3f}\t{:.3e}'.format(ave_list[i, j], std_list[i, j], gamma_list[i,j]), file=logfile)
            print('------------------------------------------------------------')
            print('best result for eps:{0} is ave:{1} and std:{2}'.format(eps, max_correct_list[i][0], max_correct_list[i][1]))
            print('------------------------------------------------------------')
            print('------------------------------------------------------------', file=logfile)
            print('best result for eps:{0} is ave:{1} and std:{2}'.format(eps, max_correct_list[i][0], max_correct_list[i][1]), file=logfile)
            print('------------------------------------------------------------', file=logfile)
            print('Running Time: '+str(end-start)+'s')

        for i in range(len(eps_list)):
            alg_idx = algorithm_names.index(alg_name) + 1
            acc_matrix[alg_idx, i] = max_correct_list[i][0]
            std_matrix[alg_idx, i] = max_correct_list[i][1]

    for i in range(acc_matrix.shape[0]):
        print(','.join(str(acc_matrix[i, j]) for j in range(9)), file=accfile)
        print(','.join(str(std_matrix[i, j]) for j in range(9)), file=stdfile)

    print('Wrote results to ' + accfile.name + ' (.std, .log)')
    print('Finish Running')


if __name__ == '__main__':
    main()
