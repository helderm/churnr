# -*- coding: utf-8 -*-
import os
import shutil
import logging
import google.cloud.storage as gcs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.extract')


def yes_or_no(question):
    """ Simple yes or no user prompt """
    reply = str(input(question+' (y/N): ')).lower().strip()
    if len(reply) and reply[0] == 'y':
        return True
    elif not len(reply) or reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... " + question)


def extract_dataset_to_disk(datapath, tablenames, project, gsoutput):
    """ Extract dataset to local files on disk"""

    logger.info('Extracting {} files to {}...'.format(len(tablenames), datapath))

    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.makedirs(datapath)

    client = gcs.Client(project)

    split_uri = gsoutput.split('/')
    bucket_name = split_uri[2]

    bucket = gcs.Bucket(client, bucket_name)

    filepath = '/'.join(split_uri[3:])
    cntr = 0
    for filename in tablenames:
        if cntr % 10 == 0:
            logger.info('{} out of {} files imported...'.format(cntr, len(tablenames)))
        cntr += 1

        # try unsharded file first
        blob = gcs.Blob(name=os.path.join(filepath, filename), bucket=bucket)
        if blob.exists():
            local_file = os.path.join(datapath, filename)
            with open(local_file, 'wb') as f:
                blob.download_to_file(f)
            continue

        # sharded file
        filename_shard = filename + '{0:012d}'.format(0)
        blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)
        if blob.exists():
            count = 0
            while blob.exists():
                local_file = os.path.join(datapath, filename_shard)
                logger.info('Downloading blob {} to local file {}'.format(blob.path, local_file))

                with open(local_file, 'wb') as f:
                    blob.download_to_file(f)
                count += 1
                filename_shard = filename + '{0:012d}'.format(count)
                blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)
        else:
            raise Exception('Blob {} not found in GCS!'.format(os.path.join(filepath, filename)))


def get_table_names(conf):
    features_table = 'features_{}_{}'.format(conf['experiment'], conf['dsname'])
    users_table = 'users_{}_{}'.format(conf['experiment'], conf['dsname'])
    return [features_table, users_table]
