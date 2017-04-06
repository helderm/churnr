#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib
import json
import pandas as pd
import glob
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.process')


def main(exppath, experiment, dsname):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('Initializing data preprocessing...')

    # load all the csv files
    with open(exppath) as fi:
        dsconf = json.load(fi)[experiment]['datasets']

    # load experiment configuration
    keys = ['rawpath', 'procpath', 'testsize', 'classbalance']
    conf = {}
    for key in keys:
        conf[key] = dsconf[dsname][key] if key in dsconf[dsname] else dsconf['global'][key]

    rawpath = conf['rawpath']
    procpath = conf['procpath']
    testsize = conf['testsize']
    classbalance = conf['classbalance']

    inpath = os.path.join(rawpath, experiment, dsname)

    tablenames = 'user_features_*'
    usertablename = 'user_ids_*'
    fpath = os.path.join(inpath, tablenames)
    upath = os.path.join(inpath, usertablename)
    allfiles = glob.glob(fpath)

    logger.info('Loading {} tables from {}..'.format(len(allfiles), inpath))

    # read all feature tables
    dtype = {'consumption_time': float, 'session_length': float, 'skip_ratio': float,
                'unique_pages': float, 'user_id': str, 'time': int}

    featdf = pd.concat((pd.read_csv(f, dtype=dtype) for f in allfiles))
    userdf = pd.concat((pd.read_csv(f) for f in glob.glob(upath)))

    logger.info('Merging tables into a single dataframe...')
    df = pd.merge(featdf, userdf, on='user_id', sort=True)

    # normalize
    logger.info('Normalizing features...')
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    scaled = preprocessing.scale(df[features], copy=False)
    scaled = np.c_[ scaled, df['user_id'], df['time'], df['churn'] ]
    df = pd.DataFrame(scaled, columns=features + ['user_id', 'time', 'churn'], )
    for f in features:
        df[f] = df[f].astype(np.float)
    df['time'] = df['time'].astype (np.int)

    # for lstms, reshape X into timesteps
    logger.info('Generating sequential dataset...')
    timepoints = df['time'].unique()
    num_samples = len(df[ df['time'] == timepoints[0] ])
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    timesteps = len(timepoints)
    X = np.empty(shape=(num_samples, timesteps, len(features)))
    for i, ts in enumerate(timepoints):
        X[:,i,:] = df[df['time'] == ts][features]

    # for aggregated models, get the mean of features
    logger.info('Generating aggregated dataset...')
    X_agg = np.empty(shape=(num_samples, len(features)))
    X_agg[:,:] = df.groupby('user_id')[features].mean()

    # extract labels
    logger.info('Extracting labels...')
    y = df[ df['time'] == df['time'].unique()[0] ]
    y = y['churn'].values.astype(int)

    #import pudb
    #pu.db

    # undersample
    logger.info('Undersampling...')
    rus = RandomUnderSampler(ratio=classbalance, return_indices=True, random_state=42)
    _, y, idxs = rus.fit_sample(X[:,0,:], y)
    X = X[idxs, :, :]
    X_agg = X_agg[idxs, :]

    # shuffle on the samples dimension
    logger.info('Shuffling...')
    X, X_agg, y = shuffle(X, X_agg, y)

    # assert the indexes of both Xs correspond to the same users
    for i in range(1000):
        assert np.allclose(X_agg[i,:], np.mean(X[i,:,:], axis=0))

    # split into train and test sets
    logger.info('Splitting into train and test sets...')
    idxs = np.arange(X_agg.shape[0])
    _, _, _, _, idxs_tr, idxs_te = train_test_split(X_agg, y, idxs, test_size=testsize, random_state=42)
    X_tr = X[idxs_tr,:,:]
    X_agg_tr = X_agg[idxs_tr,:]
    y_tr = y[idxs_tr]
    X_te = X[idxs_te,:,:]
    X_agg_te = X_agg[idxs_te,:]
    y_te = y[idxs_te]

    # create output data dir
    outdir = os.path.join(procpath, experiment, dsname)
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    # pickle
    logger.info('Storing matrices on {}...'.format(outdir))
    #df.to_csv(os.path.join(outdir, 'user_features_full.gz'), compression='gzip')

    xpath = os.path.join(outdir, 'features_seq_train.gz')
    joblib.dump(X_tr, xpath)
    xpath = os.path.join(outdir,  'features_agg_train.gz')
    joblib.dump(X_agg_tr, xpath)
    xpath = os.path.join(outdir, 'features_seq_test.gz')
    joblib.dump(X_te, xpath)
    xpath = os.path.join(outdir,  'features_agg_test.gz')
    joblib.dump(X_agg_te, xpath)

    ypath = os.path.join(outdir,  'labels_train.gz')
    joblib.dump(y_tr, ypath)
    ypath = os.path.join(outdir,  'labels_test.gz')
    joblib.dump(y_te, ypath)

    # write meta file
    #meta = { 'enddate': imeta['enddate'], 'x': os.path.abspath(xpath), 'y': os.path.abspath(ypath), 'xagg': os.path.abspath(xaggpath)}
    #with open(os.path.join(outpath, 'meta.json'), 'w') as f:
    #    json.dump(meta, f)

    logger.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processer')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset being transformed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname)
