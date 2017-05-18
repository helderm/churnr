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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
import numpy as np
from scipy.stats import randint as sp_randint

from churnr.lstm_models import custom_model
from churnr.utils import extract_dataset_to_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.train')


models = {
    'lr': {
        'obj': LogisticRegression(),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'class_weight': [None, 'balanced']
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
            'units3': sp_randint(32, 128),
            'optim': ['rmsprop', 'adagrad', 'adam'],
            'layers': sp_randint(1, 4)
        }
    },
    'rf': {
        'obj': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 40, 70, 100],
            'class_weight': [None, 'balanced']
        }
    },
    'svc': {
        'obj': SVC(probability=True),
        'params': {
            'C': [0.001, 0.01, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    },
    'gnb': {
        'obj': GaussianNB(),
        'params': {
            'priors': [None, [0.9,0.1], [0.8, 0.2], [0.7,0.3], [0.6,0.4], [0.5,0.5]],
        }
    }
}


def _fit_and_predict(estimator, X, y, train, test, class_ratio, verbose, fit_params, method):
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.model_selection._validation import _index_param_value
    from imblearn.under_sampling import RandomUnderSampler

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    rus = RandomUnderSampler(ratio=class_ratio, return_indices=True, random_state=42)
    if len(X_train.shape) > 2:
        _, y_train, idxs = rus.fit_sample(X_train[:,0,:], y_train)
        X_train = X_train[idxs, :, :]
        y_train = to_categorical(y_train)
    else:
        X_train, y_train, idxs = rus.fit_sample(X_train, y_train)

    estimator.fit(X_train, y_train, **fit_params)

    func = getattr(estimator, method)
    predictions = func(X_test)
    return predictions, test


def cross_val_predict(estimator, X, y, cv, class_ratio=1.0, method='predict', n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """ Cross-validated estimates for each input data point. Based mainly on scikit
        equally-named function, but with support for undersampling
    """
    from sklearn.utils import indexable
    from sklearn.utils.validation import _num_samples
    from sklearn.externals.joblib import Parallel, delayed
    from sklearn.model_selection._validation import _check_is_permutation
    from sklearn.base import clone

    X, y, groups = indexable(X, y, None)
    cv_iter = list(cv.split(X, y, None))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, class_ratio, verbose, fit_params, method)
        for train, test in cv_iter)

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]


def load_data(modeltype, conf):
    tablename = 'features_{}_{}_tr_{}'.format(conf['experiment'], conf['dsname'], conf['enddate'])

    local_files = extract_dataset_to_disk(conf['procpath'], [tablename], conf['project'], conf['gsoutput'])

    Xs = []
    ys = []

    # read the json file with user data
    for local_file in local_files:
        with open(local_file, 'r') as f:
            text = '[' + f.read().replace('\n', ',')[:-1] + ']'
            jdata = json.loads(text)

        for user in jdata:
            features = [user['secs_played'], user['sum_secs_played'], user['iat'], user['total_streams'], user['skip_ratio']]
            Xs.append(features)
            ys.append(int(user['churn']))

        logger.info('Added {} users to queue. Current queue size is {}'.format(len(jdata), len(Xs)))

    # load data into a numpy array
    X = np.swapaxes(np.array(Xs), 1, 2)
    y = np.array(ys)
    if modeltype != 'lstm':
        # for time-invariant models, reshape the time dimension
        # as if they were independent dimensions (GurAli 2014)
        y = np.repeat(ys, X.shape[1])
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    return X, y


def main(exppath, experiment, dsname, modelname, debug):
    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    logger.info('Initializing training of {} model...'.format(modelname.upper()))

    # load experiment configuration
    keys = ['procpath', 'project', 'gsoutput', 'enddate']
    conf = {}
    for key in keys:
        conf[key] = expconf['datasets'][dsname][key] if key in expconf['datasets'][dsname] else expconf['datasets']['global'][key]
    keys = ['modelpath', 'classbalance']
    for key in keys:
        conf[key] = expconf['models'][modelname][key] if key in expconf['models'][modelname] else expconf['models']['global'][key]
    conf['experiment'] = experiment
    conf['dsname'] = dsname

    modelpath = conf['modelpath']
    classratio = conf['classbalance']
    modeltype = modelname.split('_')[0]
    conf['gsoutput'] = os.path.join(conf['gsoutput'], experiment, dsname)
    conf['procpath'] = os.path.join(conf['procpath'], experiment, dsname)

    # load data
    X, y = load_data(modeltype, conf)

    modeldir = os.path.join(modelpath, experiment, dsname, modelname)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    try:
        # cross validate model params, using 5x2 CV
        inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        if modeltype == 'lstm':
            model = KerasClassifier(build_fn=custom_model, data_shape=(X.shape[1], X.shape[2]))
            params = models[modeltype]['params']
            clf = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5 if not debug else 1, cv=inner_cv, fit_params={'batch_size': 512, 'epochs': 70 if not debug else 30}, verbose=3, n_jobs=1, random_state=42)
        else:
            model = models[modeltype]['obj']
            params = models[modeltype]['params']
            clf = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv, verbose=3)

        # Nested CV with parameter optimization
        logger.info('Initializing cross validation loop...')

        y_pred = cross_val_predict(clf, X=X, y=y, cv=outer_cv, class_ratio=classratio, method='predict_proba', n_jobs=1, verbose=3)[:,1]

        logger.info('Cross validation finished, saving metadata...')

        # undo one hot vector for labels
        y = y[:,1] if len(y.shape) > 1 else y

        # serialize y_true and y_pred for later plot visualization
        y_trpred = np.empty(shape=(y.shape[0], 2))
        y_trpred[:,0] = y
        y_trpred[:,1] = y_pred
        joblib.dump(y_trpred, os.path.join(modeldir, 'y_test_true_pred.gz'))

        X = None

        # save the metrics and config
        metrics = {}
        metrics['roc_auc'] = roc_auc_score(y, y_pred)
        metrics['pr_auc'] = average_precision_score(y, y_pred)
        logger.info('** Test metrics **')
        logger.info('-- {}'.format(str(metrics)))
        with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

        y = None

    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--exppath', default='../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset used for training')
    parser.add_argument('--modelname', default='lr', help='Name of the model being trained')
    parser.add_argument('--debug', default=False, help='Stage that the experiment will start from', action='store_true')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, modelname=args.modelname, debug=args.debug)

