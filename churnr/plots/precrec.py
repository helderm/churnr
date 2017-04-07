#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
from cycler import cycler
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.plot')


def plot_precision_recall(y_trpred, label):
    y_true = y_trpred[:,0][:150]
    y_pred = y_trpred[:,1][:150]
    y_pred_th = np.array([0.0 if i <= 0.5 else 1.0 for i in y_pred])
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    auc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_th)
    plt.plot(rec, prec, label='{0} (area = {1:.3f}, f1 = {2:.3f})'.format(label, auc, f1))


def main(exppath, experiment):
    logger.info('Plotting Precision-Recall curves for trained models...')

    with open(exppath) as f:
        expconf = json.load(f)[experiment]

    modelpath = expconf['models']['global']['modelpath']
    plotpath = expconf['plots']['global']['plotpath']

    outpath = os.path.join(plotpath, experiment)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # initialize the roc plot
    plt.figure()
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
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
            plot_precision_recall(y_trpreds, label)

    # save the precision-recall plot
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(outpath, 'precrec.png'))

    logger.info('Precision-Recall curves plotted to {}!'.format(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curve plotter')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment)

