#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib
from collections import OrderedDict
import json
import itertools

from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.plot')


def plot_confusion_matrix(confusions, labels):
    """ Plot a confusion matrix with the result of the classifier """
    fig = 1
    numcms = len(confusions)
    for cnf, title in zip(confusions, labels):
        cnf = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
        ax = plt.subplot(np.ceil(numcms/3)+1, min(numcms, 3), fig)
        ax.set_title(title)
        ax.imshow(cnf, interpolation='nearest', cmap=plt.cm.Blues)
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
        fig += 1


def plot_precision_recall(prec_recs, pr_aucs, f1s, labels):
    for prec_rec, pr_auc, f1, label in zip(prec_recs, pr_aucs, f1s, labels):
        prec = prec_rec[0]
        rec = prec_rec[1]
        plt.plot(rec, prec, label='{0} (area = {1:.2f}, f1 = {2:.2f})'.format(label, pr_auc, f1))


def plot_roc_auc(fpr_tprs, roc_aucs, labels):
    for fpr_tpr, roc_auc, label in zip(fpr_tprs, roc_aucs, labels):
        fpr = fpr_tpr[0]
        tpr = fpr_tpr[1]
        plt.plot(fpr, tpr, label='{0} (area = {1:.3f})'.format(label, roc_auc))


def main(exppath, experiment, plotname):
    #logger.info('Plotting Precision-Recall curves for trained models...')

    with open(exppath) as f:
        expconf = json.load(f, object_pairs_hook=OrderedDict)[experiment]

    modelpath = expconf['models']['global']['modelpath']
    plotpath = expconf['plots']['global']['plotpath']

    model_label = expconf['plots'][plotname].get('modellabel', expconf['plots']['global'].get('modellabel', 1))
    ds_label = expconf['plots'][plotname].get('dslabel', expconf['plots']['global'].get('dslabel', 1))
    at_k = expconf['plots'][plotname].get('at_k', expconf['plots']['global'].get('at_k', 0))
    xlabel = expconf['plots'][plotname].get('xlabel', '')
    xrotation = expconf['plots'][plotname].get('xrotation', 'horizontal')
    ylabel = expconf['plots'][plotname].get('ylabel', '')
    title = expconf['plots'][plotname].get('title', '')

    outpath = os.path.join(plotpath, experiment)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # for every model/dataset combination
    labels = []
    aucs = []
    pr_aucs = []
    f1s = []
    prec_recs = []
    fpr_tprs = []
    confusions = []
    for dsname in expconf['datasets'].keys():
        if dsname == 'global':
            continue
        for modelname in expconf['models'].keys():
            if modelname == 'global':
                continue

            # get a pretty label name
            modelname_pretty = expconf['models'][modelname].get('prettyname', modelname) if model_label else ''
            dsname_pretty = expconf['datasets'][dsname].get('prettyname', dsname) if ds_label else ''
            label = '{}{}{}'.format(modelname_pretty, ' - ' if model_label + ds_label == 2 else '', dsname_pretty)

            # fetch the true values and predictions
            predpath = os.path.join(modelpath, experiment, dsname, modelname, 'y_test_true_pred.gz')
            y_trpred = joblib.load(predpath)

            y_true = y_trpred[:,0]
            y_pred = y_trpred[:,1]
            if at_k:
                y_true = y_true[:at_k]
                y_pred = y_pred[:at_k]

            # calculate PR AUC
            prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)

            # calculate F1 by thresholding to 0.5, maybe it would be smarter to use on of the thresholds
            #   returned by precision_recall_curve()
            y_pred_th = np.array([0.0 if i <= 0.5 else 1.0 for i in y_pred])
            f1 = f1_score(y_true, y_pred_th)

            # calculate ROC curve and ROC AUC
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            # calculate the confusion matrix
            cfm = confusion_matrix(y_true, y_pred_th, labels=[0,1])

            f1s.append(f1)
            aucs.append(roc_auc)
            labels.append(label)
            pr_aucs.append(pr_auc)
            prec_recs.append((prec, rec))
            fpr_tprs.append((fpr, tpr))
            confusions.append(cfm)


    # initialize the plot
    plt.figure()
    sns.set_palette('Set2', n_colors=10)

    if plotname == 'precrec':
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plot_precision_recall(prec_recs, pr_aucs, f1s, labels)
        plt.legend(loc='best')

    if plotname == 'line':
        plt.xlim([0.0, float(len(labels)+1)])
        #plt.ylim([0.5, 1.0])

        x = list(range(1,len(labels)+1))
        plt.xticks(x, labels, rotation=xrotation)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        #plt.plot(x, aucs, label='ROC AUC')
        plt.plot(x, pr_aucs, label='PR AUC')
        #plt.plot(x, f1s, label='F1 score')
        plt.legend(loc="lower left")

    if plotname == 'roc':
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plot_roc_auc(fpr_tprs, aucs, labels)
        plt.legend(loc="lower right")

    if plotname == 'confusion':
        plot_confusion_matrix(confusions, labels)

    # save the precision-recall plot
    plt.savefig(os.path.join(outpath, '{}.png'.format(plotname)))

    logger.info('Plot saved to {}!'.format(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curve plotter')
    parser.add_argument('--exppath', default='../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--plotname', default='line', help='Name of the plot')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, plotname=args.plotname)

