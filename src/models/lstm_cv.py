#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import glob
import json
import datetime as dt

from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint as sp_randint
import numpy as np

import lstm_models
from utils import get_data, plot_roc_auc, yes_or_no


logger = logging.getLogger('churnr.lstm')


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logger.info("Model with rank: {0}".format(i))
            logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            logger.info("Parameters: {0}".format(results['params'][candidate]))
            logger.info("")


@click.command()
@click.option('--inpath', default='../../data/processed')
@click.option('--outpath', default='../../models')
def main(inpath, outpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    logger.info('Loading features at [{}] and targets at [{}]'.format(meta['x'], meta['y']))
    X, y, X_te, y_te = get_data(meta['x'], meta['y'])

    # get the model instance
    logger.info('Compiling LSTM model...')
    #model = get_model(data_shape=(X.shape[1], X.shape[2]))
    estimator = KerasClassifier(build_fn=lstm_models.custom_model, data_shape=(X.shape[1], X.shape[2]))

    modeldir = os.path.join(outpath, 'lstm_s{}_t{}_cv'.format(meta['enddate'], int(dt.datetime.now().timestamp())))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    try:
        # set model callbacks
        weightsdir = os.path.join(modeldir, 'weights')
        if not os.path.exists(weightsdir):
            os.makedirs(weightsdir)
        chkp_path = os.path.join(weightsdir, 'model_best.hdf5')
        tensorboard = TensorBoard(log_dir=os.path.join(modeldir, 'logs'), write_images=True)
        callbacks = [tensorboard]

        # split data into train and val sets
        tscv = TimeSeriesSplit(n_splits=2)

        import pudb
        pu.db

        # set the hyperparameters search algorithm
        units1 = sp_randint(32, 128)
        units2 = sp_randint(32, 128)
        units3 = sp_randint(32, 128)
        units4 = sp_randint(32, 128)
        units5 = sp_randint(32, 128)

        layers = sp_randint(1, 5)
        optim = ['rmsprop', 'adagrad', 'sgd']
        #epochs = [50, 100, 150]
        #batches = [5, 10, 20]
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = dict(units1=units1, units2=units2, units3=units3, units4=units4, units5=units5, layers=layers, optim=optim)
        cv = RandomizedSearchCV(estimator=estimator, param_distributions=param_dist, n_iter=5, cv=tscv, iid=False, fit_params={'callbacks': callbacks, 'batch_size': 128, 'epochs': 20})
        cv.fit(X, y)

        # report the results
        report(cv.cv_results_)

        model = cv.best_estimator_.model

        # save the best model
        model.save(chkp_path)

        # print the model test metrics
        logger.info('Evaluating model on the test set...')
        metrics_values = model.evaluate(X_te, y_te, verbose=0)
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
        with open(os.path.join(modeldir, 'params.json'), 'w') as f:
            json.dump(cv.best_params_, f)

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
    except Exception as e:
        logger.exception(e)
        ans = yes_or_no('Delete folder at {}?'.format(modeldir))
        if ans:
            import shutil
            shutil.rmtree(modeldir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
