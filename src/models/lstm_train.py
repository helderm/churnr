#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import glob

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping

import json
from keras.models import load_model
import datetime as dt

from lstm_models import light_model
from utils import get_data, plot_roc_auc

logger = logging.getLogger('churnr.lstm')


@click.command()
@click.option('--inpath', default='../../data/processed')
@click.option('--outpath', default='../../models')
def main(inpath, outpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    logger.info('Loading features at [{}] and targets at [{}]'.format(meta['x'], meta['y']))
    X, y, X_te, y_te, X_val, y_val = get_data(meta['x'], meta['y'], threesplit=True)

    # get the model instance
    logger.info('Compiling LSTM model...')
    import pudb
    pu.db

    model = light_model(data_shape=(X.shape[1], X.shape[2]))

    modeldir = os.path.join(outpath, 'lstm_s{}_t{}'.format(meta['enddate'], int(dt.datetime.now().timestamp())))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # set model callbacks
    weightsdir = os.path.join(modeldir, 'weights')
    if not os.path.exists(weightsdir):
        os.makedirs(weightsdir)
    chkp_path = os.path.join(weightsdir, 'model_best.hdf5')
    chkp = ModelCheckpoint(chkp_path, monitor='val_acc', save_best_only=True, period=1, verbose=1)
    reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.001, verbose=1, cooldown=10, epsilon=0.1)
    tensorboard = TensorBoard(log_dir=os.path.join(modeldir, 'logs'), write_images=True)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=15, verbose=1)
    callbacks = [chkp, reducelr, tensorboard, earlystop]

    # train the model for each train / val folds
    logger.info('Training model...')

    model.fit(X, y,
                batch_size=128, epochs=100,
                validation_data=(X_val, y_val),
                callbacks=callbacks)

    # reload the best model checkpoint
    logger.info('Reloading checkpoint from best model found...')
    model = load_model(chkp_path)

    # print the model test metrics
    logger.info('Evaluating model on the test set...')
    metrics_values = model.evaluate(X_te, y_te)
    metrics_names = model.metrics_names if type(model.metrics_names) == list else [model.metrics_names]
    metrics = {}
    logger.info('** Test metrics **')
    for metric, metric_name in zip(metrics_values, metrics_names):
        logger.info('-- {0}: {1:.3f}'.format(metric_name, metric))
        metrics[metric_name] = metric

    # calculate roc and auc and plot it
    y_pred = model.predict(X_te)
    roc_auc = plot_roc_auc(y_te[:,1], y_pred[:,1], modeldir)
    logger.info('-- auc: {0:.3f}'.format(roc_auc))
    metrics['auc'] = roc_auc

    # save the metrics and config
    with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    with open(os.path.join(modeldir, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

    # check if this is the new best model
    bestmetric = None
    bestmodel = None
    for modelname in glob.glob(os.path.join(outpath, 'lstm*')):
        metrics_path = os.path.join(modelname, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
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
