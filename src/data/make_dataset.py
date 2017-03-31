#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import json
import pandas as pd
import glob
from sklearn import preprocessing
import numpy as np

@click.command()
@click.option('--inpath', default='../../data/raw')
@click.option('--outpath', default='../../data/processed')
def main(inpath, outpath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing raw data...')

    # load all the csv files
    with open(os.path.join(inpath, 'meta.json')) as fi:
        meta = json.load(fi)

    tablenames = 'user_features_s{}_*'.format(meta['enddate'])
    usertablename = 'user_ids_sampled_{}'.format(meta['enddate'])
    fpath = os.path.join(inpath, tablenames)
    upath = os.path.join(inpath, usertablename)
    allfiles = glob.glob(fpath)

    logger.info('Loading {} tables from users sampled from {}..'.format(len(allfiles), meta['enddate']))

    # read all feature tables
    dtype = {'consumption_time': float, 'session_length': float, 'skip_ratio': float,
                'unique_pageviews': float, 'user_id': str, 'time': int}
    featdf = pd.concat((pd.read_csv(f, dtype=dtype) for f in allfiles))
    #df = df.sort_values(by=['time','user_id'])
    userdf = pd.read_csv(upath)

    df = pd.merge(featdf, userdf, on='user_id', sort=True)

    # delete unnecessessary columns
    #del df['user_id']

    # normalize
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    scaled = preprocessing.scale(df[features], copy=False)
    scaled = np.c_[ scaled, df['user_id'], df['time'], df['churn'] ]
    df = pd.DataFrame(scaled, columns=features + ['user_id', 'time', 'churn'], )
    for f in features:
        df[f] = df[f].astype(np.float)
    df['time'] = df['time'].astype (np.int)

    # for lstms, reshape X into timesteps
    timepoints = df['time'].unique()
    num_samples = len(df[ df['time'] == timepoints[0] ])
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    timesteps = len(timepoints)
    X = np.empty(shape=(num_samples, timesteps, len(features)))
    for i, ts in enumerate(timepoints):
        X[:,i,:] = df[df['time'] == ts][features]

    # for aggregated models, get the mean of features
    X_agg = np.empty(shape=(num_samples, len(features)))
    X_agg[:,:] = df.groupby('user_id')[features].mean()

    # extract labels
    y = df[ df['time'] == df['time'].unique()[0] ]
    y = y['churn'].values.astype(int)

    # create output data dir
    outdir = os.path.join(outpath, 's'+meta['enddate'])
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    # pickle
    logger.info('Storing normalized dataframe and matrices on {}...'.format(outdir))
    df.to_csv(os.path.join(outdir, 'user_features_full.gz'), compression='gzip')

    xpath = os.path.join(outdir, 'user_features_lstm.npz')
    np.savez_compressed(xpath, X)
    xaggpath = os.path.join(outdir,  'user_features_agg.npz')
    np.savez_compressed(xaggpath, X_agg)
    ypath = os.path.join(outdir,  'user_labels.npz')
    np.savez_compressed(ypath, y)

    # write meta file
    meta = { 'enddate': meta['enddate'], 'x': os.path.abspath(xpath), 'y': os.path.abspath(ypath), 'xagg': os.path.abspath(xaggpath) }
    with open(os.path.join(outpath, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    logger.info('Done!')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
