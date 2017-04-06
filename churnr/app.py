#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

from churnr.data import extract, process
from churnr.models import lstm_train, lr_train, abdt_train
from churnr.plots import roc

model2pkg = {
    'lstm': lstm_train,
    'lr': lr_train,
    'abdt': abdt_train
}

plot2pkg = {
    'roc': roc,
}


logger = logging.getLogger('churnr.experiment')

def main(exppath, experiment, stage):
    logger.info('Initializing experiment...')

    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    datasets = expconf['datasets']
    models = expconf['models']
    plots = expconf['plots']
    expabspath = os.path.abspath(exppath)

    import pudb
    pu.db

    # extract each dataset of the experiment
    if stage in ['extract']:
        for dsname, dsconf in datasets.items():
            if dsname == 'global':
                continue

        extract.main(exppath=expabspath, experiment=experiment, dsname=dsname, hddump=False)

    if stage in ['extract', 'process']:
        for dsname, dsconf in datasets.items():
            if dsname == 'global':
                continue

            process.main(exppath=expabspath, experiment=experiment, dsname=dsname)

    # train each model using each dataset
    if stage in ['extract', 'process', 'train']:
        for dsname, dsconf in datasets.items():
            if dsname == 'global':
                continue

            for modelname, modelconf in models.items():
                if modelname == 'global':
                    continue

                train_fn = model2pkg[modelname].main
                train_fn(exppath=expabspath, experiment=experiment, dsname=dsname, modelname=modelname)


    if stage in ['extract', 'process', 'train', 'plot']:
        for plotname, plotconf in plots.items():
            if plotname == 'global':
                continue

            plot_fn = plot2pkg[plotname].main
            plot_fn(exppath=expabspath, experiment=experiment)


    logger.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment dispatcher')
    parser.add_argument('--exppath', default='../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--stage', default='extract', help='Stage that the experiment will start from', choices=['extract', 'process', 'train', 'plot'])

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, stage=args.stage)

