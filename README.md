# Differentially Private Convex Optimization Benchmark

A benchmark and implementations for differentially private convex optimization algorithms.

The algorithms implemented in this repository are as follows:
1. Approximate Minia Perturbation - An original algorithm proposed in our paper.
2. Hyperparameter-free Approximate Minima Perturbation - Hyperparameter-free version of 1.
3. Private Stochastic Gradient Descent in [scs13](http://ai2-s2-pdfs.s3.amazonaws.com/6154/ce8c02375184f7928e41c4fae532500f7175.pdf)
4. Private Convex Perturbation-Based Stochastic Gradient Descent in [wlk17](https://arxiv.org/pdf/1606.04722.pdf)
5. Private Strongly Convex Perturbation-Based Stochastic Gradient Descent in [wlk17](https://arxiv.org/pdf/1606.04722.pdf)
6. Private Frank-Wolfe in [ttz16](https://arxiv.org/pdf/1411.5417.pdf)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The codes are currently implemented using NumPy. They require Python version 3.5 or newer. You will also need to install all dependencies listed in the requirements.txt file in the repository. The recommended way to do this is through the use of a Python virtual environment.

### Virtual Environment

You can set up a virtual environment as follows:

1. Navigate to the directory that you have checked out this repository in.
2. Create a virtual environment named *venv* by running:
```bash
python3 -m venv venv
```
If any needed packages are missing, you should get an error message telling you which ones to install.

3. Activate the virtual environment by running the following command on Posix systems:
```bash
source venv/bin/activate
```
There will be a script for Windows systems located at *venv/Scripts/activate*. However, none of the code in this repository has been tested on Windows.

### Prerequisites

```
cycler==0.10.0
matplotlib==2.0.2
numpy==1.13.0
pyparsing==2.2.0
python-dateutil==2.6.0
pytz==2017.2
scipy==0.19.0
scikit-learn==0.18.1
six==1.10.0
xlrd==1.0.0
```

### Installing

#### Linux

1. Navigate to the this repository.
2. Run the following command line.

```bash
pip install -r requirements.txt
```

#### Windows

1. Navigate to the this repository.
2. Open requirements.txt in the repo, and run ''pip install'' for all of the prequisities in order.

## Running the benchmark

### Download and preprocess the datasets

1. Navigate to the ''datasets'' directory.
2. Run the following command line to download and preprocess all the benchmark datasets automatically.
```bash
python main_preprocess.py all
```
3. If you want to download one of the datasets, just replace ''all'' with the name of the dataset. All available datasets are listed as following.
```
adult, covertype, gisette, kddcup99, mnist, realsim, rcv1
```

### Run the benchmarks

1. Navigate to this repository.
2. Run algorithms on one dataset using the following command.

```bash
python gridsearch.py [ALG_NAME] [DATASET_NAME] [MODEL_NAME]
```

3. Available ALG_NAME
```
ALL: all the algorithms
AMP: Flexible Objective Perturbation
AMP-NT: Hyperparameter-free Flexible Objective Perturbation
PSGD: Private Stochastic Gradient Descent
PPSGD: Private Convex Perturbation-Based Stochastic Gradient Descent
PPSSGD: Private Convex Perturbation-Based Stochastic Gradient Descent
FW: Private Frank-Wolfe
```

4. Available DATASET_NAME
```
adult, covertype, gisette, kddcup99, mnist, realsim, rcv1
```
5. Available MODEL_NAME
```
LR: Logistic Regression
SVM: Huber SVM without kernel functions
```
6. The results are stored in csvs in ''dpml-algorithms/results/rough_results''

### Draw the graphs

1. Navigate to this repository.
2. Run the following command to get the graph after running the corresponding benchmark.
```bash
python draw.py [DATASET_NAME] [ALG_NAME] [MODEL_NAME]
```
3. The graphs are in ''dpml-algorihtms/results/graphs''.

