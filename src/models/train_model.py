#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import json
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle

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


def get_data(inpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)
    X = np.load(meta['x'])
    y = np.load(meta['y'])

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
def main(inpath):

    X, y = get_data(inpath)

    timesteps = X.shape[1]
    dim = X.shape[2]

    model = get_model(timesteps, dim)

    # split data into train and val sets
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(X):
        X_train = X[train_idx,:,:]
        X_test = X[test_idx,:,:]

        y_train = y[train_idx,:]
        y_test = y[test_idx,:]

        model.fit(X_train, y_train,
                  batch_size=64, epochs=10,
                  validation_data=(X_test, y_test))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
