#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

import google.cloud.storage as gcs

import churnr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.submitter')


def download_gcs_file(gcs_uri, local_path, project):
    logger.info('Downloading GCS file {} to {}...'.format(gcs_uri, local_path))

    client = gcs.Client(project)

    split_uri = gcs_uri.split('/')
    bucket_name = split_uri[2]
    gcs_path = '/'.join(split_uri[3:])

    bucket = gcs.Bucket(client, bucket_name)
    blob = gcs.Blob(name=gcs_path, bucket=bucket)

    with open(local_path, 'wb') as f:
        blob.download_to_file(f)


def upload_dir_to_gcs(local_dir, gcs_uri, project):
    logger.info('Uploading local dir {} to {}...'.format(local_dir, gcs_uri))

    client = gcs.Client(project)

    split_uri = gcs_uri.split('/')
    bucket_name = split_uri[2]
    gcs_path = '/'.join(split_uri[3:])

    bucket = gcs.Bucket(client, bucket_name)

    folders2skip = len(local_dir.split('/'))+1
    for root, directories, filenames in os.walk(local_dir):
        gcs_prefix = root.split('/')[folders2skip:]
        gcs_prefix = '/'.join(gcs_prefix)
        for filename in filenames:
            local_filepath = os.path.join(root, filename)
            gcs_filepath = os.path.join(gcs_path, gcs_prefix, filename)

            blob = gcs.Blob(name=gcs_filepath, bucket=bucket)

            with open(local_filepath, 'r') as f:
                blob.upload_from_file(f)


def main():
    pkgpath = os.path.dirname(churnr.__file__)

    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--job-dir', default='gs://helder/churnr/', help='GCS path where the model is going to be stored')
    parser.add_argument('--expfile', default='experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--stages', default=['train'], help='Stages that will be executed', nargs='*')
    parser.add_argument('--models', default=[], help='Model that will be trained', nargs='*')
    parser.add_argument('--datasets', default=[], help='Datasets that will be processed', nargs='*')

    args = parser.parse_args()
    logger.info('Initializing experiment [{}]...'.format(args.experiment))

    # load experiments.json
    args.exppath = os.path.join(pkgpath, args.expfile)
    with open(args.exppath, 'r') as f:
        conf = json.load(f)

    # override some paths
    procpath = '/tmp/data/processed'
    modelpath = '/tmp/models'
    plotpath = '/tmp/plots'
    conf[args.experiment]['datasets']['global']['procpath'] = procpath
    conf[args.experiment]['models']['global']['modelpath'] = modelpath
    conf[args.experiment]['plots']['global']['plotpath'] = plotpath
    conf = { k:v for k,v in conf.items() if k == args.experiment}
    project = conf[args.experiment]['datasets']['global']['project']

    logger.info('Saving experiments file with modified paths...')
    logger.info('--- {}'.format(conf))

    with open(args.exppath, 'w') as f:
        json.dump(conf, f)

    from churnr.app import run
    run(args)

    # sync model outputs to gcs
    jobdir = os.path.join(args.job_dir, args.experiment)
    upload_dir_to_gcs(modelpath, jobdir, project)

    logger.info('Done!')

if __name__ == '__main__':
    main()
