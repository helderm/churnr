#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import joblib
import json
import datetime as dt
import pandas as pd
import glob
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

import churnr.utils as utils

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
    keys = ['rawpath', 'procpath', 'testsize', 'gsoutput', 'enddate', 'project', 'obsdays', 'preddays', 'timesplits']
    conf = {}
    for key in keys:
        conf[key] = dsconf[dsname][key] if key in dsconf[dsname] else dsconf['global'][key]
    if conf['enddate'] == 'yesterday':
        conf['enddate'] = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
    conf['experiment'] = experiment
    conf['dsname'] = dsname

    rawpath = conf['rawpath']
    procpath = conf['procpath']
    testsize = conf['testsize']
    gsoutput = conf['gsoutput']
    project = conf['project']

    inpath = os.path.join(rawpath, experiment, dsname)

    # if raw path is empty, assume the data is in GCS and try to download it
    if not os.path.exists(inpath):
        logger.info('Raw data not found in {}, trying to download it from GCS...'.format(inpath))
        gspath = os.path.join(gsoutput, experiment, dsname)
        tablenames = utils.get_table_names(conf)
        utils.extract_dataset_to_disk(inpath, tablenames, project, gspath)

    # read all csv tables
    tablenames = 'features_*'
    usertablename = 'users_*'
    fpath = os.path.join(inpath, tablenames)
    upath = os.path.join(inpath, usertablename)
    allfiles = glob.glob(fpath)

    logger.info('Loading {} tables from {}..'.format(len(allfiles), inpath))

    dtype = {'consumption_time': float, 'session_length': float, 'skip_ratio': float,
                'unique_pages': float, 'user_id': str, 'time': int,
                'consumption_time_std': float, 'session_length_std': float, 'skip_ratio_std': float,
                'unique_pages_std': float, 'backfill': bool}

    featdf = pd.concat((pd.read_csv(f, dtype=dtype) for f in allfiles))
    userdf = pd.concat((pd.read_csv(f) for f in glob.glob(upath)))

    logger.info('Merging tables into a single dataframe...')
    df = pd.merge(featdf, userdf, on='user_id', sort=True)

    # filter out the user_ids not present in this dataset
    # that may have been added by another dataset with a larger window
    unknown_users = list(set(df.user_id.unique()) - set(df[df.backfill==False].user_id.unique()))
    df = df[~df.user_id.isin(unknown_users)]

    nobf_df = df[df.backfill == False]

    # normalize
    logger.info('Normalizing features...')
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id' and f != 'backfill']
    scaled = preprocessing.scale(nobf_df[features], copy=False)
    scaled = np.c_[ scaled, nobf_df['user_id'], nobf_df['time'], nobf_df['churn'], nobf_df['backfill'] ]
    new_df = pd.DataFrame(scaled, columns=features + ['user_id', 'time', 'churn', 'backfill'], )
    for f in features:
        new_df[f] = new_df[f].astype(np.float)
    new_df['time'] = new_df['time'].astype(np.int)

    # join the normalized data with the ones backfilled
    df = pd.concat([new_df, df[df.backfill == True]])
    df = df.sort_values(['user_id','time'])

    # for lstms, reshape X into timesteps
    logger.info('Generating sequential dataset...')
    timepoints = df['time'].unique()
    #num_samples = len(df[ df['time'] == timepoints[0] ])
    num_samples = len(df['user_id'].unique())
    #features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    timesteps = len(timepoints)
    X = np.empty(shape=(num_samples, timesteps, len(features)))
    for i, ts in enumerate(timepoints):
        X[:,i,:] = df[df['time'] == ts][features]

    # for aggregated models, get the mean of features
    logger.info('Generating aggregated dataset...')
    num_samples = len(nobf_df['user_id'].unique())
    X_agg = np.empty(shape=(num_samples, len(features)))
    X_agg[:,:] = nobf_df.groupby('user_id', sort=True)[features].mean()

    # extract labels
    logger.info('Extracting labels...')
    y = df[ df['time'] == df['time'].unique()[0] ]
    y = y['churn'].values.astype(int)

    # shuffle on the samples dimension
    logger.info('Shuffling...')
    X, X_agg, y = shuffle(X, X_agg, y)

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

    logger.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processer')
    parser.add_argument('--exppath', default='./experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset being transformed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname)
