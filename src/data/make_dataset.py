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
    with open(os.path.join(inpath, 'meta.json')) as f:
        meta = json.load(f)

    tablenames = 'user_features_s{}_*'.format(meta['enddate'])
    usertablename = 'user_ids_sampled_{}'.format(meta['enddate'])
    fpath = os.path.join(inpath, tablenames)
    upath = os.path.join(inpath, usertablename)
    allfiles = glob.glob(fpath)

    logger.info('Loading {} tables from users sampled from {}..'.format(len(allfiles), meta['enddate']))

    # read all feature tables
    featdf = pd.concat((pd.read_csv(f) for f in allfiles))
    #df = df.sort_values(by=['time','user_id'])
    userdf = pd.read_csv(upath)

    df = pd.merge(featdf, userdf, on='user_id', sort=True)

    # delete unnecessessary columns
    #del df['user_id']

    # normalize
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    scaled = preprocessing.scale(df[features])
    scaled = np.c_[ scaled, df['user_id'], df['time'], df['churn'] ]
    df = pd.DataFrame(scaled, columns=features + ['user_id', 'time', 'churn'])

    # for lstms, reshape X into timesteps
    timepoints = df['time'].unique()
    num_samples = len(df[ df['time'] == timepoints[0] ])
    features = [f for f in df.columns if f != 'churn' and f != 'time' and f != 'user_id']
    timesteps = len(timepoints)
    X = np.empty(shape=(num_samples, timesteps, len(features)))
    for i, ts in enumerate(timepoints):
        X[:,i,:] = df[df['time'] == ts][features]

    # extract labels
    y = df[ df['time'] == df['time'].unique()[0] ]
    y = y['churn'].values.astype(int)

    # pickle
    logger.info('Storing normalized dataframe on {}...'.format(outpath))
    df.to_pickle(os.path.join(outpath, 'user_features_s{}.pkl'.format(meta['enddate'])))

    xpath = os.path.join(outpath, 'user_features_lstm_s{}.npy'.format(meta['enddate']))
    X.dump(xpath)
    ypath = os.path.join(outpath, 'user_labels_lstm_s{}.npy'.format(meta['enddate']))
    y.dump(ypath)

    # write meta file
    meta = { 'enddate': meta['enddate'], 'x': xpath, 'y': ypath }
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
