import argparse
import os
from preprocess import (
    adult, covertype, gisette, kddcup99, mnist, realsim, rcv1)

#o185, o4550, and uber_ato are private datasets, so can only be used in experiments

CACHE_LOCATION = 'data_cache'
OUTPUT_LOCATION = 'data'

datasets = {
    'adult': adult,
    'covertype': covertype,
    'gisette': gisette, #Sparse, but can be represented in a dense format
    'kddcup99': kddcup99,
    'mnist': mnist,
    'realsim': realsim,
    'rcv1': rcv1,
}


def preprocess_all():
    for (name, dataset) in datasets.items():
        print('Preprocessing ' + name)
        dataset.preprocess(CACHE_LOCATION, OUTPUT_LOCATION)



def main():
    parser = argparse.ArgumentParser(
        description='Preprocess the specified dataset')
    parser.add_argument('dataset', type=str, help='The name of the dataset',
                        choices=list(datasets.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == 'all':
        preprocess_all()
    else:
        datasets[args.dataset].preprocess(CACHE_LOCATION, OUTPUT_LOCATION)


if __name__ == '__main__':
    main()
