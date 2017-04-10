#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

from churnr.data import extract, process
from churnr.models import train
from churnr.plots import roc, precrec, confusion

plot2pkg = {
    'roc': roc,
    'precrec': precrec,
    'confusion': confusion,
}


logger = logging.getLogger('churnr.experiment')


def run(exppath, experiment, stage):
    logger.info('Initializing experiment...')

    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    datasets = expconf['datasets']
    models = expconf['models']
    plots = expconf['plots']
    expabspath = os.path.abspath(exppath)

    # extract each dataset of the experiment
    if stage in ['extract']:
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            extract.main(exppath=expabspath, experiment=experiment, dsname=dsname, hddump=False)

    if stage in ['extract', 'process']:
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            process.main(exppath=expabspath, experiment=experiment, dsname=dsname)

    # train each model using each dataset
    if stage in ['extract', 'process', 'train']:
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            for modelname in models.keys():
                if modelname == 'global':
                    continue

                train.main(exppath=expabspath, experiment=experiment, dsname=dsname, modelname=modelname)


    if stage in ['extract', 'process', 'train', 'plot']:
        for plotname in plots.keys():
            if plotname == 'global':
                continue

            plot_fn = plot2pkg[plotname].main
            plot_fn(exppath=expabspath, experiment=experiment)


    logger.info('Done!')


def main():
    parser = argparse.ArgumentParser(description='Experiment dispatcher')
    parser.add_argument('--exppath', default='../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--stage', default='extract', help='Stage that the experiment will start from', choices=['extract', 'process', 'train', 'plot'])

    args = parser.parse_args()

    run(exppath=args.exppath, experiment=args.experiment, stage=args.stage)

if __name__ == '__main__':
    main()

