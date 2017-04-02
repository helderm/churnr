#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
import glob
from dotenv import find_dotenv, load_dotenv
import joblib

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn
from cycler import cycler

logger = logging.getLogger('churnr.plot')

def plot_roc_auc(y_trpreds, outpath):

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


    for modelname, y_trpred in y_trpreds.items():
        y_true = y_trpred[:,0]
        y_pred = y_trpred[:,1]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='{0} (area = {1:.3f})'.format(modelname, roc_auc))

    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outpath, 'roc.png'), bbox_inches='tight')


@click.command()
@click.option('--inpath', default='../../models')
@click.option('--outpath', default='./')
def main(inpath, outpath):

    # load all the best model's predictions
    y_trpreds = {}
    for f in glob.glob(os.path.join(inpath, 'best_*')):
        if f.endswith('best_lr'):
            modelname = 'Logistic Regression'
        if f.endswith('best_lstm'):
            modelname = 'LSTM'
        if f.endswith('best_abdt'):
            modelname = 'AdaBoost DT'
        y_trpreds[modelname] = joblib.load(os.path.join(f, 'y_test_true_pred.gz'))

    plot_roc_auc(y_trpreds, outpath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
