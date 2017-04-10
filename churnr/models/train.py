#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc, f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
import numpy as np
from scipy.stats import randint as sp_randint
import pandas as pd

from churnr.utils import yes_or_no
from churnr.models.lstm_models import custom_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.train')


models = {
    'lr': {
        'obj': LogisticRegression(),
        'params': {
            'C': [0.001, 0.01, 1, 10, 100],
            'penalty': ['l1', 'l2'],
        }
    },
    'abdt': {
        'obj': AdaBoostClassifier(),
        'params': {
            'n_estimators': [10, 40, 70, 100],
            'learning_rate': [1.0, 0.6, 0.2],
        }
    },
    'lstm': {
        'obj': None, # postpone instatiation until we have the data shape
        'params': {
            'units1': sp_randint(32, 128),
            'units2': sp_randint(32, 128),
            'optim': ['rmsprop', 'adagrad', 'sgd', 'adam'],
            'layers': sp_randint(1, 2)
        }
    },
    'rf': {
        'obj': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 40, 70, 100],
            'max_depth': [None, 10, 20],
        }
    },
    'svc': {
        'obj': SVC(probability=True),
        'params': {
            'C': [0.001, 0.01, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    },
}


def main(exppath, experiment, dsname, modelname):
    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    logger.info('Initializing training of {} model...'.format(modelname.upper()))

    # load experiment configuration
    keys = ['procpath']
    conf = {}
    for key in keys:
        conf[key] = expconf['datasets'][dsname][key] if key in expconf['datasets'][dsname] else expconf['datasets']['global'][key]
    keys = ['modelpath']
    for key in keys:
        conf[key] = expconf['models'][modelname][key] if key in expconf['models'][modelname] else expconf['models']['global'][key]

    # load data
    procpath = conf['procpath']
    modelpath = conf['modelpath']

    ypath_tr = os.path.join(procpath, experiment, dsname, 'labels_train.gz')
    if modelname == 'lstm':
        Xpath_tr = os.path.join(procpath, experiment, dsname, 'features_seq_train.gz')
        y = to_categorical(joblib.load(ypath_tr))
    else:
        Xpath_tr = os.path.join(procpath, experiment, dsname, 'features_agg_train.gz')
        y = joblib.load(ypath_tr)

    logger.info('Loading features from [{}] and targets from [{}]'.format(Xpath_tr, ypath_tr))
    X = joblib.load(Xpath_tr)

    modeldir = os.path.join(modelpath, experiment, dsname, modelname)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    try:
        # cross validate model params, using 5x2 CV
        logger.info('Cross validating model hyperparams...')

        inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        if modelname == 'lstm':
            model = KerasClassifier(build_fn=custom_model, data_shape=(X.shape[1], X.shape[2]))
            params = models[modelname]['params']
            clf = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=10, cv=inner_cv, iid=False, fit_params={'batch_size': 512, 'epochs': 20}, verbose=3, n_jobs=1, random_state=42)
        else:
            model = models[modelname]['obj']
            params = models[modelname]['params']
            clf = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv, verbose=3)

        # Non_nested parameter search and scoring
        clf.fit(X, y)

        best_params = clf.best_params_
        best_estimator = clf.best_estimator_

        logger.info('Best hyperparams found!')
        logger.info('-- params = {}'.format(best_params))
        pd.DataFrame(clf.cv_results_).to_pickle(os.path.join(modeldir, 'cv_results.pkl'))

        if modelname == 'lstm':
            fit_params= {'batch_size': 512, 'epochs': 60}
        else:
            fit_params=None

        # Nested CV with parameter optimization
        logger.info('Initializing outer cross validation loop using best hyperparams...')
        y_pred = cross_val_predict(best_estimator, X=X, y=y, cv=outer_cv, method='predict_proba', n_jobs=1, verbose=3, fit_params=fit_params)[:,1]
        y_pred_th = np.array([0.0 if i <= 0.5 else 1.0 for i in y_pred])

        logger.info('Cross validation finished, saving metadata...')

        # undo one hot vector for labels
        y = y[:,1] if len(y.shape) > 1 else y

        # print the model test metrics
        metrics = {}
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        metrics['auc'] = roc_auc
        metrics['f1'] = f1_score(y, y_pred_th)
        logger.info('** Test metrics **')
        for key, val in metrics.items():
            logger.info('-- {0}: {1:.3f}'.format(key, val))

        # serialize y_true and y_pred for later plot visualization
        y_trpred = np.empty(shape=(y.shape[0], 2))
        y_trpred[:,0] = y
        y_trpred[:,1] = y_pred
        joblib.dump(y_trpred, os.path.join(modeldir, 'y_test_true_pred.gz'))

        # save the metrics and config
        with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        with open(os.path.join(modeldir, 'config.json'), 'w') as f:
            json.dump(best_params, f)

    except Exception as e:
        logger.exception(e)
        ans = yes_or_no('Delete folder at {}?'.format(modeldir))
        if ans:
            import shutil
            shutil.rmtree(modeldir)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset used for training')
    parser.add_argument('--modelname', default='lr', help='Name of the model being trained')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, modelname=args.modelname)

