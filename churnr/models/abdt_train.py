#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import logging
import json
import joblib
import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

from churnr.utils import plot_roc_auc, yes_or_no


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.abdt')


def main(exppath, experiment, dsname, modelname):
    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    # load experiment configuration
    keys = ['procpath']
    conf = {}
    for key in keys:
        conf[key] = expconf['datasets'][dsname][key] if key in expconf['datasets'][dsname] else expconf['datasets']['global'][key]
    keys = ['modelpath']
    for key in keys:
        conf[key] = expconf['models'][modelname][key] if key in expconf['models'][modelname] else expconf['models']['global'][key]

    procpath = conf['procpath']
    modelpath = conf['modelpath']
    Xpath_tr = os.path.join(procpath, experiment, dsname, 'features_agg_train.gz')
    ypath_tr = os.path.join(procpath, experiment, dsname, 'labels_train.gz')
    Xpath_te = os.path.join(procpath, experiment, dsname, 'features_agg_test.gz')
    ypath_te = os.path.join(procpath, experiment, dsname, 'labels_test.gz')

    logger.info('Loading features from [{}] and targets from [{}]'.format(Xpath_tr, ypath_tr))

    X = joblib.load(Xpath_tr)
    y = joblib.load(ypath_tr)
    X_te = joblib.load(Xpath_te)
    y_te = joblib.load(ypath_te)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # get the model instance
    logger.info('Compiling AdaBoost DT model...')

    model = AdaBoostClassifier()

    modeldir = os.path.join(modelpath, experiment, dsname, modelname)
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

    except Exception as e:
        logger.exception(e)
        ans = yes_or_no('Delete folder at {}?'.format(modeldir))
        if ans:
            import shutil
            shutil.rmtree(modeldir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaBoost DT trainer')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset used for training')
    parser.add_argument('--modelname', default='adbt', help='Name of the model being trained')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, modelname=args.modelname)


