#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import gc

import matplotlib
matplotlib.use('Agg')

import churnr
from churnr import extract, process, train, plot

logger = logging.getLogger('churnr.experiment')


def run(args):
    logger.info('Initializing experiment...')

    with open(args.exppath) as fi:
        expconf = json.load(fi)[args.experiment]

    datasets = expconf['datasets']
    models = expconf['models']
    plots = expconf['plots']
    expabspath = os.path.abspath(args.exppath)

    # extract stage
    if 'extract' in args.stages:
        # sort the datasets by size of observation or prediction window, and sample users only on
        # the largest one as to have the same set of user ids of across all datasets
        if 'preddays' not in datasets['global']:
            datasets_sorted = sorted(datasets.items(), key=lambda x: x[1].get('preddays', 0), reverse=True)
        else:
            datasets_sorted = sorted(datasets.items(), key=lambda x: x[1].get('obsdays', 0), reverse=True)
        sampleusers = True
        for dsname, _ in datasets_sorted:
            if dsname == 'global':
                continue

            extract.main(exppath=expabspath, experiment=args.experiment, dsname=dsname, hddump=False, sampleusers=sampleusers)
            sampleusers = False

    # process stage
    if 'process' in args.stages:
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            process.main(exppath=expabspath, experiment=args.experiment, dsname=dsname)

    # train stage
    if 'train' in args.stages:
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            # remove any local files stored in another run
            dspath = os.path.join(datasets['global']['procpath'], args.experiment, dsname)
            if os.path.exists(dspath):
                import shutil
                shutil.rmtree(dspath)
            os.makedirs(dspath)

            for modelname in models.keys():
                if modelname == 'global':
                    continue

                train.main(exppath=expabspath, experiment=args.experiment, dsname=dsname, modelname=modelname, debug=args.debug)

                # dispatches garbage collector after each training
                gc.collect()

    # plot stage
    if 'plot' in args.tages:
        for plotname in plots.keys():
            if plotname == 'global':
                continue

            plot.main(exppath=expabspath, experiment=args.experiment, plotname=plotname)

    logger.info('Done!')


def main():
    path = os.path.dirname(churnr.__file__)

    parser = argparse.ArgumentParser(description='Experiment dispatcher')
    parser.add_argument('--exppath', default=os.path.join(path, 'experiments.json'), help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--stages', default=['extract','process','train', 'plot'], help='Stages that will be executed', nargs='*')
    parser.add_argument('--hddump', default=False, help='Store files in local disk', action='store_true')
    parser.add_argument('--debug', default=False, help='Debug flag that sped up some stages', action='store_true')

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()

