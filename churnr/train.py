#!/usr/bin/env python
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
from sklearn.decomposition import TruncatedSVD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.sparse import csr_matrix, vstack
import pandas as pd

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
            'units1': sp_randint(64, 256),
            'units2': sp_randint(64, 256),
            'units3': sp_randint(64, 256),
            'optim': ['rmsprop', 'adagrad', 'adam'],
            'layers': sp_randint(1, 4)
        }
    },
    'rf': {
        'obj': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 100, 500, 1000]
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

args = None

def _fit_and_predict(estimator, X, y, train, test, class_ratio, verbose, fit_params, method):
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.model_selection._validation import _index_param_value
    from imblearn.under_sampling import RandomUnderSampler

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    rus = RandomUnderSampler(ratio=class_ratio, return_indices=True, random_state=42)
    if len(X.shape) < 2:
        X0 = X[0][train]
        y_train = y[train]

        idxs = rus.fit_sample(X0, y_train)
        X_train = np.empty(shape=(len(idxs), X.shape[0], X0.shape[1]))
        X_test = np.empty(shape=(len(test), X.shape[0], X0.shape[1]))
        for i in range(X.shape[0]):
            X_train[:,i,:] = X[i][train][idxs].toarray()
            X_test[:,i,:] = X[i][test].toarray()

        y_train = to_categorical(y_train[idxs])
    else:
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, _ = _safe_split(estimator, X, y, test, train)

        idxs = rus.fit_sample(X_train, y_train)
        X_train = X_train[idxs]
        y_train = y_train[idxs]

    clf = estimator.fit(X_train, y_train, **fit_params)

    func = getattr(estimator, method)

    logger.info('-- predict_proba()')
    predictions = func(X_test)
    return predictions, test, pd.DataFrame(clf.cv_results_)


def cross_val_predict(estimator, X, y, cv, class_ratio=1.0, method='predict', n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """ Cross-validated estimates for each input data point. Based mainly on scikit
        equally-named function, but with support for undersampling
    """
    from sklearn.utils import indexable
    #from sklearn.utils.validation import _num_samples
    from sklearn.externals.joblib import Parallel, delayed
    #from sklearn.model_selection._validation import _check_is_permutation
    from sklearn.base import clone

    # if sparse lstm matrix
    if len(X.shape) < 2:
        X0, y, groups = indexable(X[0], y, None)
        cv_iter = list(cv.split(X0, y, None))
    else:
        X, y, groups = indexable(X, y, None)
        cv_iter = list(cv.split(X, y, None))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    logger.info('-- Dispatching cross-val jobs...')

    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, class_ratio, verbose, fit_params, method)
        for train, test in cv_iter)

    logger.info('-- Concatenating results...')

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i, _ in prediction_blocks])
    cv_results = [cv_res for _, _, cv_res in prediction_blocks]

    #if not _check_is_permutation(test_indices, _num_samples(X)):
    #    raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    predictions = np.concatenate(predictions)
    return predictions[inv_test_indices], cv_results


def load_data(modeltype, conf):
    if not conf['debug']:
        tablename = 'features_{}_{}_tr'.format(conf['experiment'], conf['dsname'])
    else:
        tablename = 'features_{}_{}_tr'.format(conf['experiment'], conf['dsname'])

    local_files = extract_dataset_to_disk(conf['procpath'], [tablename], conf['project'], conf['gsoutput'])

    Xs = []
    ys = []
    Xts = []

    X = None

    # read the json file with user data
    for local_file in local_files:
        with open(local_file, 'r') as f:
            text = '[' + f.read().replace('\n', ',')[:-1] + ']'
            jdata = json.loads(text)
            text = None

        for user in jdata:
            if len(conf['features']) > 0:
                features = [user[k] for k in sorted(user.keys()) if k in conf['features'] and k not in ['user_id', 'churn', 'times']]
            else:
                features = [user[k] for k in sorted(user.keys()) if k not in ['user_id', 'churn', 'times']]

            features.insert(0, user['times'])
            features = list(zip(*sorted(zip(*features))))
            features = features[1:] # remove times
            Xs.append(features)

            ys.append(int(user['churn']))

        logger.info('-- Read {} users from file {}'.format(len(jdata), local_file))

        Xp = np.swapaxes(np.array(Xs, dtype=float), 1, 2)
        Xs = []
        if modeltype != 'lstm':
            # for time-invariant models, reshape the time dimension
            # as if they were independent dimensions (GurAli 2014)
            #y = np.repeat(ys, X.shape[1])
            #X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
            #X = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 1, X)
            # TODO: take the mean of values that are not backfilled!
            #         can I do it here on in BigQuery?
            if X == None:
                X = csr_matrix(np.mean(Xp, axis=1))
            else:
                X = vstack([X, csr_matrix(np.mean(Xp, axis=1))])
        else:
            # create one sparse matrix for each timestep
            if len(Xts) == 0:
                for i in range(Xp.shape[1]):
                    Xts.append(csr_matrix(Xp[:,i,:]))
            else:
                for i in range(Xp.shape[1]):
                    Xts[i] = vstack([Xts[i], csr_matrix(Xp[:,i,:])])

        Xp = None

    if modeltype == 'lstm':
        X = np.array(Xts)

    y = np.array(ys)
    return X, y


def main(args):
    with open(args.exppath) as fi:
        expconf = json.load(fi)[args.experiment]

    logger.info('Initializing training of {} model...'.format(args.modelname.upper()))

    if args.debug:
        import pudb
        pu.db

    # load experiment configuration
    keys = ['procpath', 'project', 'gsoutput', 'enddate']
    conf = {}
    for key in keys:
        conf[key] = expconf['datasets'][args.dsname][key] if key in expconf['datasets'][args.dsname] else expconf['datasets']['global'][key]
    keys = ['modelpath', 'classbalance', 'dimred', 'features']
    for key in keys:
        conf[key] = expconf['models'][args.modelname][key] if key in expconf['models'][args.modelname] else expconf['models']['global'][key]

    conf['experiment'] = args.experiment
    conf['dsname'] = args.dsname
    conf['debug'] = args.debug

    modelpath = conf['modelpath']
    classratio = conf['classbalance']
    modeltype = args.modelname.split('_')[0]
    conf['gsoutput'] = os.path.join(conf['gsoutput'], args.experiment, args.dsname)
    conf['procpath'] = os.path.join(conf['procpath'], args.experiment, args.dsname)

    X, y = load_data(modeltype, conf)

    # reduce dimensionality of data
    if conf['dimred'] > 0:
        logger.info('Reducing dimensionality of data to {} components...'.format(conf['dimred']))
        if modeltype == 'lstm':
            svd = TruncatedSVD(n_components=conf['dimred']).fit(X[0])

            for i, Xt in enumerate(X):
                X[i] = csr_matrix(svd.transform(Xt))
        else:
            X = TruncatedSVD(n_components=conf['dimred']).fit_transform(X)

    modeldir = os.path.join(modelpath, args.experiment, args.dsname, args.modelname)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # cross validate model params, using 5x2 CV
    inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    if modeltype == 'lstm':
        model = KerasClassifier(build_fn=custom_model, data_shape=(X.shape[0], X[0].shape[1]))
        params = models[modeltype]['params']
        fit_params = {'batch_size': 512, 'epochs': 100}
        clf = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5 if not args.debug else 1, cv=inner_cv, fit_params=fit_params, verbose=3, n_jobs=1, random_state=42)
    else:
        model = models[modeltype]['obj']
        params = models[modeltype]['params']
        clf = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv, verbose=3)

    # Nested CV with parameter optimization
    logger.info('Initializing cross validation loop...')

    y_pred, cv_results = cross_val_predict(clf, X=X, y=y, cv=outer_cv, class_ratio=classratio, method='predict_proba', n_jobs=1, verbose=3)
    y_pred = y_pred[:,1]

    logger.info('Cross validation finished, saving metadata...')

    # save cross validation results
    pd.concat(cv_results).to_csv(os.path.join(modeldir, 'cv_results.csv'))

    # undo one hot vector for labels
    y = y[:,1] if len(y.shape) > 1 else y

    # serialize y_true and y_pred for later plot visualization
    y_trpred = np.empty(shape=(y.shape[0], 2))
    y_trpred[:,0] = y
    y_trpred[:,1] = y_pred
    joblib.dump(y_trpred, os.path.join(modeldir, 'y_test_true_pred.gz'))

    # save the metrics and config
    metrics = {}
    metrics['roc_auc'] = roc_auc_score(y, y_pred)
    metrics['pr_auc'] = average_precision_score(y, y_pred)
    logger.info('** Test metrics **')
    logger.info('-- {}'.format(str(metrics)))
    with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # return data for next model
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--exppath', default='./experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset used for training')
    parser.add_argument('--modelname', default='lr', help='Name of the model being trained')
    parser.add_argument('--debug', default=False, help='Stage that the experiment will start from', action='store_true')

    args = parser.parse_args()

    main(args)

