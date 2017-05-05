#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

import matplotlib
matplotlib.use('Agg')

from churnr import extract, process, train, plot

logger = logging.getLogger('churnr.experiment')


def run(exppath, experiment, stage, singlestage, debug, hddump):
    logger.info('Initializing experiment...')

    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    datasets = expconf['datasets']
    models = expconf['models']
    plots = expconf['plots']
    expabspath = os.path.abspath(exppath)

    # extract stage
    if stage in ['extract']:
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

            extract.main(exppath=expabspath, experiment=experiment, dsname=dsname, hddump=hddump, sampleusers=sampleusers)
            sampleusers = False

    # process stage
    if stage in ['extract', 'process'] and (not singlestage or (singlestage and stage == 'process')):
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            process.main(exppath=expabspath, experiment=experiment, dsname=dsname)

    # train stage
    if stage in ['extract', 'process', 'train'] and (not singlestage or (singlestage and stage == 'train')):
        for dsname in datasets.keys():
            if dsname == 'global':
                continue

            for modelname in models.keys():
                if modelname == 'global':
                    continue

                train.main(exppath=expabspath, experiment=experiment, dsname=dsname, modelname=modelname, debug=debug)

    # plot stage
    if stage in ['extract', 'process', 'train', 'plot'] and (not singlestage or (singlestage and stage == 'plot')):
        for plotname in plots.keys():
            if plotname == 'global':
                continue

            plot.main(exppath=expabspath, experiment=experiment, plotname=plotname)

    logger.info('Done!')


def main():
    parser = argparse.ArgumentParser(description='Experiment dispatcher')
    parser.add_argument('--exppath', default='./experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--stage', default='extract', help='Stage that the experiment will start from', choices=['extract', 'process', 'train', 'plot'])
    parser.add_argument('--singlestage', default=False, help='Stage that the experiment will start from', action='store_true')
    parser.add_argument('--hddump', default=False, help='Store files in local disk', action='store_true')
    parser.add_argument('--debug', default=False, help='Debug flag that sped up some stages', action='store_true')

    args = parser.parse_args()

    run(exppath=args.exppath, experiment=args.experiment, stage=args.stage, singlestage=args.singlestage, debug=args.debug, hddump=args.hddump)

if __name__ == '__main__':
    main()

