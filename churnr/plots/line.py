#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib
from collections import OrderedDict
import json

from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.plot')


def main(exppath, experiment, plotname):
    logger.info('Plotting Precision-Recall curves for trained models...')

    with open(exppath) as f:
        expconf = json.load(f, object_pairs_hook=OrderedDict)[experiment]

    modelpath = expconf['models']['global']['modelpath']
    plotpath = expconf['plots']['global']['plotpath']

    outpath = os.path.join(plotpath, experiment)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    model_label = expconf['plots'][plotname].get('modellabel', expconf['plots']['global'].get('modellabel', 1))
    ds_label = expconf['plots'][plotname].get('dslabel', expconf['plots']['global'].get('dslabel', 1))
    xlabel = expconf['plots'][plotname].get('xlabel', '')
    xrotation = expconf['plots'][plotname].get('xrotation', 'horizontal')
    ylabel = expconf['plots'][plotname].get('ylabel', '')
    title = expconf['plots'][plotname].get('title', '')

    # for every model/dataset combination
    labels = []
    aucs = []
    f1s = []
    for dsname in expconf['datasets'].keys():
        if dsname == 'global':
            continue
        for modelname in expconf['models'].keys():
            if modelname == 'global':
                continue

            modelname_pretty = expconf['models'][modelname].get('prettyname', modelname) if model_label else ''
            dsname_pretty = expconf['datasets'][dsname].get('prettyname', dsname) if ds_label else ''
            label = '{}{}{}'.format(modelname_pretty, ' - ' if model_label + ds_label == 2 else '', dsname_pretty)
            predpath = os.path.join(modelpath, experiment, dsname, modelname, 'y_test_true_pred.gz')
            y_trpred = joblib.load(predpath)

            y_true = y_trpred[:,0]
            y_pred = y_trpred[:,1]
            y_pred_th = np.array([0.0 if i <= 0.5 else 1.0 for i in y_pred])
            f1 = f1_score(y_true, y_pred_th)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            f1s.append(f1)
            aucs.append(roc_auc)
            labels.append(label)

    # initialize the roc plot
    plt.figure()
    plt.xlim([0.0, float(len(labels)+1)])
    plt.ylim([0.5, 1.0])
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))

    x = list(range(1,len(labels)+1))
    plt.xticks(x, labels, rotation=xrotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(x, aucs, label='ROC AUC')
    plt.plot(x, f1s, label='F1 score')
    plt.legend(loc="lower left")

    # save the precision-recall plot
    plt.savefig(os.path.join(outpath, 'line.png'))

    logger.info('Line plot plotted to {}!'.format(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curve plotter')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--plotname', default='line', help='Name of the plot')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, plotname=args.plotname)

