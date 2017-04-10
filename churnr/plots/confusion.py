#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib
import itertools

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.plot')


def plot_confusion_matrix(y_trpred, clf_name, fig, numcms):
    """ Plot a confusion matrix with the result of the classifier """
    y_true = y_trpred[:,0]
    y_pred = y_trpred[:,1]
    y_pred_th = np.array([0.0 if i <= 0.5 else 1.0 for i in y_pred])

    cnf = confusion_matrix(y_true, y_pred_th, labels=[0,1])
    cnf = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]

    ax = plt.subplot(np.ceil(numcms/3), 3, fig+1)
    ax.set_title(clf_name)
    plt.imshow(cnf, interpolation='nearest', cmap=plt.cm.Blues)
    thresh = cnf.max() / 2.0
    for i, j in itertools.product(range(cnf.shape[0]), range(cnf.shape[1])):
        ax.text(j, i, '{:.2f}'.format(cnf[i, j]), horizontalalignment="center", color="white" if cnf[i, j] > thresh else "black")

    ax.set_yticks([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticklabels(['-','+'])
    ax.set_xticklabels(['-','+'])
    if fig == 0:
        ax.set_ylabel('True')
    if fig == numcms-1:
        ax.set_xlabel('Pred')

def main(exppath, experiment):
    logger.info('Plotting confusion matrices for trained models...')

    with open(exppath) as f:
        expconf = json.load(f)[experiment]

    modelpath = expconf['models']['global']['modelpath']
    plotpath = expconf['plots']['global']['plotpath']

    outpath = os.path.join(plotpath, experiment)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # for every model/dataset combination
    fig = 0

    dsnames = [d for d in expconf['datasets'].keys() if d != 'global']
    modelnames = [d for d in expconf['models'].keys() if d != 'global']
    numcms = len(dsnames) * len(modelnames)
    for dsname in dsnames:
        for modelname in modelnames:
            label = '{} - {}'.format(modelname, dsname)
            predpath = os.path.join(modelpath, experiment, dsname, modelname, 'y_test_true_pred.gz')
            y_trpreds = joblib.load(predpath)
            plot_confusion_matrix(y_trpreds, label, fig, numcms)
            fig += 1

    # save the ROC plot
    plt.savefig(os.path.join(outpath, 'confusion.png'), bbox_inches='tight')

    logger.info('Confusion matrices plotted to {}'.format(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curve plotter')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment)

