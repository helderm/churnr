#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn
from cycler import cycler
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.plot')


def plot_roc_auc(y_trpred, label):
    y_true = y_trpred[:,0]
    y_pred = y_trpred[:,1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{0} (area = {1:.3f})'.format(label, roc_auc))


def main(exppath, experiment):
    logger.info('Plotting ROC curve for trained models...')

    with open(exppath) as f:
        expconf = json.load(f)[experiment]

    modelpath = expconf['models']['global']['modelpath']
    plotpath = expconf['plots']['global']['plotpath']

    outpath = os.path.join(plotpath, experiment)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # initialize the roc plot
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))

    # for every model/dataset combination
    for dsname in expconf['datasets'].keys():
        if dsname == 'global':
            continue
        for modelname in expconf['models'].keys():
            if modelname == 'global':
                continue

            label = '{} - {}'.format(modelname, dsname)
            predpath = os.path.join(modelpath, experiment, dsname, modelname, 'y_test_true_pred.gz')
            y_trpreds = joblib.load(predpath)
            plot_roc_auc(y_trpreds, label)

    # save the ROC plot
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outpath, 'roc.png'), bbox_inches='tight')

    logger.info('ROC curves plotted to {}!'.format(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curve plotter')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment)

