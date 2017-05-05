#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

import google.cloud.storage as gcs


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
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--job-dir', default='gs://helder/churnr/', help='GCS path where the model is going to be stored')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--project', default='user_lifecycle', help='Name of the GCP project')
    parser.add_argument('--debug', default=False, help='Debug flag that sped up some stages', action='store_true')
    parser.add_argument('--stage', default='process', help='Stage that the experiment will start from', choices=['extract', 'process', 'train', 'plot'])
    parser.add_argument('--singlestage', default=False, help='Stage that the experiment will start from', action='store_true')

    args = parser.parse_args()
    logger.info('Initializing experiment [{}]...'.format(args.experiment))

    # download experiments.json
    exppath_gcs = os.path.join(args.job_dir, 'experiments.json')
    exppath = os.path.join('/tmp/', 'experiments.json')
    download_gcs_file(exppath_gcs, exppath, args.project)
    with open(exppath, 'r') as f:
        conf = json.load(f)

    # override some paths
    procpath = '/tmp/data/processed'
    modelpath = '/tmp/models'
    plotpath = '/tmp/plots'
    conf[args.experiment]['datasets']['global']['procpath'] = procpath
    conf[args.experiment]['models']['global']['modelpath'] = modelpath
    conf[args.experiment]['plots']['global']['plotpath'] = plotpath
    conf = { k:v for k,v in conf.items() if k == args.experiment}

    logger.info('Saving experiments file with modified paths...')
    logger.info('--- {}'.format(conf))

    with open(exppath, 'w') as f:
        json.dump(conf, f)

    from churnr.app import run
    run(exppath=exppath, experiment=args.experiment, stage=args.stage, singlestage=args.singlestage, debug=args.debug, hddump=False)

    # sync model outputs to gcs
    jobdir = os.path.join(args.job_dir, args.experiment)
    #upload_dir_to_gcs(procpath, jobdir, args.project)
    #upload_dir_to_gcs(plotpath, jobdir, args.project)
    upload_dir_to_gcs(modelpath, jobdir, args.project)

    logger.info('Done!')

if __name__ == '__main__':
    main()
