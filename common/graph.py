import argparse
import csv
import logging
import os
import sys
import time
import statistics
from common.common import Result
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s')


def plot_results_on_epsilon(
        results_by_dataset, output_directory, graph_id, graph_type):
    """
    Plot all results with epsilon on the x axis

    Only results with lambda_param equal to the value
    passed as lambda_param will be plotted
    """

    figs = []

    for dataset, results in results_by_dataset.items():
        fig = plt.figure()
        figs.append(fig)
        xs = defaultdict(list)
        ys = defaultdict(list)

        for key, error_rates in sorted(results.items()): #, error_rates in results.items():
            algorithm, epsilon, lmbda = key
            
            xs[algorithm].append(epsilon)

            error_rate = statistics.median(error_rates)
            ys[algorithm].append(error_rate)

        ax = fig.add_subplot(111)
        ax.set(title=dataset, xlabel='epsilon', ylabel='error rate')

        for algorithm in xs:
            ax.plot(xs[algorithm], ys[algorithm], label=algorithm)

        # based on http://jb-blog.readthedocs.io/en/latest/posts/0012-
        # matplotlib-legend-outdide-plot.html
        art = [plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))]
        full_output_path = os.path.join(
            output_directory, '{}_{}_epsilon_graph_{}.png'.format(
                dataset, graph_type, graph_id))
        plt.savefig(full_output_path, additional_artists=art,
                    bbox_inches="tight")



def parse_results(filename):
    """Parses data from a csv file"""
    all_results = defaultdict(lambda: defaultdict(list))
           
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # Skip header row
        next(reader)

        for row in reader:
            error_rate = float(row[5]) / float(row[6])
            result = Result(row[0], row[1], float(row[2]), float(row[3]),
                            convert_lambda_if_nonempty(row[4]), float(error_rate))
            key = (result.algorithm, result.epsilon,
                   result.lambda_param)

            all_results[result.dataset][key].append(error_rate)

    return all_results


def convert_lambda_if_nonempty(lambda_param):
    if lambda_param.isnumeric():
        return int(lambda_param)
    else:
        return lambda_param


def build_graph(results_file, output_directory, graph_id, graph_type):
    print("results file is {0}".format(results_file))
    results_by_dataset = parse_results(results_file)
    plot_results_on_epsilon(
        results_by_dataset, output_directory, graph_id, graph_type)
