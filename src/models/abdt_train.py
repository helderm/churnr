#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
import glob
from dotenv import find_dotenv, load_dotenv
import json
import datetime as dt
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np

from utils import get_data, plot_roc_auc, yes_or_no

logger = logging.getLogger('churnr.abdt')


@click.command()
@click.option('--inpath', default='../../data/processed')
@click.option('--outpath', default='../../models')
def main(inpath, outpath):
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    logger.info('Loading features at [{}] and targets at [{}]'.format(meta['x'], meta['y']))
    X, y, X_te, y_te = get_data(meta['xagg'], meta['y'], onehot=False)

    # get the model instance
    logger.info('Compiling AdaBoost DT model...')

    model = AdaBoostClassifier()

    modeldir = os.path.join(outpath, 'abdt_s{}_t{}'.format(meta['enddate'], int(dt.datetime.now().timestamp())))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    try:
        # cross validate model params, using 5x2 CV
        logger.info('Cross validating model hyperparams...')
        kf = KFold(n_splits=2, shuffle=True)

        params = { 'n_estimators': [10, 30, 50, 70, 100],
                    'learning_rate': [1.0, 0.8, 0.6, 0.4],
                        }
        cv = GridSearchCV(estimator=model, param_grid=params, cv=kf, verbose=1)
        cv.fit(X,y)

        model = cv.best_estimator_

        logger.info('Training finished, saving model...')
        joblib.dump(model, os.path.join(modeldir, 'lr_model.pkl'))

        # print the model test metrics
        logger.info('Evaluating model on the test set...')
        acc = model.score(X_te, y_te)
        metrics = {'acc': acc}

        # calculate roc and auc and plot it
        y_pred = model.predict_proba(X_te)
        roc_auc = plot_roc_auc(y_te, y_pred[:,1], modeldir)
        metrics['auc'] = roc_auc

        logger.info('** Test metrics **')
        for key, val in metrics.items():
            logger.info('-- {0}: {1:.3f}'.format(key, val))

        # serialize y_true and y_pred for later roc visualization
        y_trpred = np.empty(shape=(y_te.shape[0], 2))
        y_trpred[:,0] = y_te
        y_trpred[:,1] = y_pred[:,1]
        joblib.dump(y_trpred, os.path.join(modeldir, 'y_test_true_pred.gz'))

        # save the metrics and config
        with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        with open(os.path.join(modeldir, 'config.json'), 'w') as f:
            json.dump(cv.best_params_, f)

        # check if this is the new best model
        bestmetric = None
        bestmodel = None
        for modelname in glob.glob(os.path.join(outpath, 'abdt*')):
            metrics_path = os.path.join(modelname, 'metrics.json')
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                metric = json.load(f)
            if not bestmetric or bestmetric['auc'] < metric['auc']:
                bestmetric = metric
                bestmodel = modelname
        bestpath = os.path.join(outpath, 'best_abdt')
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
