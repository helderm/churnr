#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import json
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
import datetime as dt

def get_model(timesteps, dim):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(2, activation='softmax')) # 2 classes, churn or non-churned

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
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
    return Xs, ysc


@click.command()
@click.option('--inpath', default='../../data/processed')
@click.option('--outpath', default='../../models')
def main(inpath, outpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    X, y = get_data(meta['x'], meta['y'])

    # get the model instance
    model = get_model(X.shape[1], X.shape[2])
    modeldir = os.path.join(outpath, 'lstm_s{}_t{}'.format(meta['enddate'], int(dt.datetime.now().timestamp())))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # set model callbacks
    chkp_path = os.path.join(modeldir, 'model_e{epoch:02d}-va{val_acc:.2f}-vl{val_loss:.2f}.hdf5')
    chkp = ModelCheckpoint(chkp_path, monitor='val_acc', save_best_only=True, period=5)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(modeldir, 'logs'), write_images=True)
    callbacks = [chkp, reducelr, tensorboard]

    # split data into train and val sets
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(X):
        X_train = X[train_idx,:,:]
        X_test = X[test_idx,:,:]

        y_train = y[train_idx,:]
        y_test = y[test_idx,:]

        model.fit(X_train, y_train,
                  batch_size=64, epochs=20,
                  validation_data=(X_test, y_test),
                  callbacks=callbacks)

    with open(os.path.join(modeldir, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
