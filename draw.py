import os 
import sys
import csv
import numpy as np 
import pylab
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

csv_location='results/rough_results'
result_location='results/graphs'

def main():
    if len(sys.argv) != 4:
        print("Usage: draw.py dataset_name alg_name loss_name")
        os._exit(1)

    dataset_name = sys.argv[1]
    erm_name = sys.argv[2]
    alg_name = sys.argv[3]
    accuracy_location = os.path.join(csv_location, alg_name, erm_name+'_'+dataset_name+'.acc')
    stddev_location = os.path.join(csv_location, alg_name, erm_name+'_'+dataset_name+'.std')

    eps_list = [0.01, 0.0316227766017, 0.1, 0.316227766017, 1, 3.16227766017, 10]
    w = len(eps_list)

    accuracy_list = np.loadtxt(accuracy_location, delimiter=',')[:,:w]*100
    stddev_list = np.loadtxt(stddev_location, delimiter=',')[:,:w]*100

    name_list = ['Non-private baseline',
                 'Approximate Minima Perturbation',
                 'Hyperparameter-free Approximate Minima Perturbation',
                 'Private SGD',
                 'Private PSGD',
                 'Private Strongly-convex PSGD',
                 'Private Frank-Wolfe']

    algs_used = [0,1,2,3,4,5,6]    

    ax = plt.subplot()
    ax.set_xscale("log", nonposx='clip')

    n = accuracy_list.shape[0]

    for i in algs_used:
        ax.errorbar(eps_list, accuracy_list[i], yerr=stddev_list[i], label=name_list[i], capsize=5, capthick=2, elinewidth=1, linestyle='dashed', marker='o')

    plt.xlabel('Epsilon', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.grid(True)

    fig_name = os.path.join(result_location, alg_name, dataset_name+'_'+erm_name+'.pdf')
    with PdfPages(fig_name) as pdf:
        pdf.savefig(bbox_inches='tight')

if __name__ == '__main__':
    main()
