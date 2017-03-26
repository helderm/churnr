#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import glob

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import json
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import datetime as dt

logger = logging.getLogger('churnr.lstm')

def get_model(timesteps, dim):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(2, activation='softmax')) # 2 classes, churn or non-churned

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['binary_accuracy'])
    return model


def get_data(xpath, ypath):
    X = np.load(xpath)
    y = np.load(ypath)

    # balance classes
    rus = RandomUnderSampler(return_indices=True)
    X_rs, y, idxs = rus.fit_sample(X[:,0,:], y)
    X = X[idxs, :, :]

    # shuffle on the samples dimension
    Xs, ys = shuffle(X, y)

    # turn labels into one-hot vectors
    ysc = to_categorical(ys)

    # split into train/val and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ysc, test_size=0.2, random_state=42)

    return X_tr, y_tr, X_te, y_te


def plot_roc_auc(fpr, tpr, roc_auc, outpath):
    import matplotlib.pyplot as plt
    import seaborn

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
	     lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outpath, 'roc.png'), bbox_inches='tight')


@click.command()
@click.option('--inpath', default='../../data/processed')
@click.option('--outpath', default='../../models')
def main(inpath, outpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    X, y, X_te, y_te = get_data(meta['x'], meta['y'])

    # get the model instance
    model = get_model(X.shape[1], X.shape[2])
    modeldir = os.path.join(outpath, 'lstm_s{}_t{}'.format(meta['enddate'], int(dt.datetime.now().timestamp())))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # set model callbacks
    weightsdir = os.path.join(modeldir, 'weights')
    if not os.path.exists(weightsdir):
        os.makedirs(weightsdir)
    chkp_path = os.path.join(weightsdir, 'model_e{epoch:02d}-va{val_binary_accuracy:.2f}-vl{val_loss:.2f}.hdf5')
    chkp = ModelCheckpoint(chkp_path, monitor='val_binary_accuracy', save_best_only=True, period=5)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(modeldir, 'logs'), write_images=True)
    callbacks = [chkp, reducelr, tensorboard]

    # split data into train and val sets
    tscv = TimeSeriesSplit(n_splits=3)

    # train the model for each train / val folds
    for train_idx, val_idx in tscv.split(X):
        X_train = X[train_idx,:,:]
        X_val = X[val_idx,:,:]

        y_train = y[train_idx,:]
        y_val = y[val_idx,:]

        model.fit(X_train, y_train,
                  batch_size=64, epochs=20,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks)

    # print the model test metrics
    metrics_values = model.evaluate(X_te, y_te, verbose=0)
    metrics_names = model.metrics_names if type(model.metrics_names) == list else [model.metrics_names]
    metrics = {}
    logger.info('** Test metrics **')
    for metric, metric_name in zip(metrics_values, metrics_names):
        logger.info('-- {0}: {1:.3f}'.format(metric_name, metric))
        metrics[metric_name] = metric

    # calculate roc and auc and plot it
    y_pred = model.predict(X_te)
    fpr, tpr, _ = roc_curve(y_te[:,1], y_pred[:,1])
    roc_auc = auc(fpr, tpr)
    logger.info('-- auc: {0:.3f}'.format(roc_auc))
    metrics['auc'] = roc_auc
    plot_roc_auc(fpr, tpr, roc_auc, modeldir)

    # save the metrics and config
    with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    with open(os.path.join(modeldir, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

    # check if this is the new best model
    bestmetric = None
    bestmodel = None
    for modelname in glob.glob(os.path.join(outpath, 'lstm*')):
        with open(os.path.join(modelname, 'metrics.json')) as f:
            metric = json.load(f)
        if not bestmetric or bestmetric['auc'] < metric['auc']:
            bestmetric = metric
            bestmodel = modelname
    bestpath = os.path.join(outpath, 'best_lstm')
    if os.path.exists(bestpath):
        os.unlink(bestpath)
    os.symlink(os.path.abspath(bestmodel), bestpath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
