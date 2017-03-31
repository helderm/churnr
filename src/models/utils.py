#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os

import numpy as np
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn


def yes_or_no(question):
    """ Simple yes or no user prompt """
    reply = str(input(question+' (y/N): ')).lower().strip()
    if len(reply) and reply[0] == 'y':
        return True
    elif not len(reply) or reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... " + question)



def get_data(xpath, ypath, threesplit=False):
    X = np.load(xpath)
    y = np.load(ypath)

    # load array from compressed file
    if X.__class__.__name__ == 'NpzFile':
        X = X[X.keys()[0]]
    if y.__class__.__name__ == 'NpzFile':
        y = y[y.keys()[0]]

    # balance classes
    rus = RandomUnderSampler(return_indices=True)
    X_rs, y, idxs = rus.fit_sample(X[:,0,:], y)
    X = X[idxs, :, :]

    # shuffle on the samples dimension
    Xs, ys = shuffle(X, y)

    # turn labels into one-hot vectors
    ysc = to_categorical(ys)

    # split into train/val and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ysc, test_size=0.15, random_state=42)
    if not threesplit:
        return X_tr, y_tr, X_te, y_te

    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
    return X_tr, y_tr, X_te, y_te, X_val, y_val


def plot_roc_auc(y_true, y_pred, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
	     lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outpath, 'roc.png'), bbox_inches='tight')

    return roc_auc

