# -*- coding: utf-8 -*-
import os
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

    client = gcs.Client(project)

    split_uri = gsoutput.split('/')
    bucket_name = split_uri[2]

    bucket = gcs.Bucket(client, bucket_name)

    filepath = '/'.join(split_uri[3:])
    local_files = []
    for filename in tablenames:

        # try unsharded file first
        blob = gcs.Blob(name=os.path.join(filepath, filename), bucket=bucket)
        if blob.exists():
            local_file = os.path.join(datapath, filename)
            if not os.path.exists(local_file):
                logger.info('-- Downloading blob [{}] to local file [{}]'.format(blob.path, local_file))
                with open(local_file, 'wb') as f:
                    blob.download_to_file(f)
            else:
                logger.info('-- File [{}] already found on local disk, reusing it...'.format(local_file))

            local_files.append(local_file)
            continue

        # sharded file
        filename_shard = filename + '{0:012d}'.format(0)
        blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)
        if blob.exists():
            count = 0
            while blob.exists():
                local_file = os.path.join(datapath, filename_shard)

                if not os.path.exists(local_file):
                    logger.info('-- Downloading blob [{}] to local file [{}]'.format(blob.path, local_file))
                    with open(local_file, 'wb') as f:
                        blob.download_to_file(f)
                else:
                    logger.info('-- File [{}] already found on local disk, reusing it...'.format(local_file))

                local_files.append(local_file)
                count += 1
                filename_shard = filename + '{0:012d}'.format(count)
                blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)
        else:
            raise Exception('Blob {} not found in GCS!'.format(os.path.join(filepath, filename)))

    return local_files


def get_table_names(conf):
    features_table = 'features_{}_{}'.format(conf['experiment'], conf['dsname'])
    users_table = 'users_{}_{}'.format(conf['experiment'], conf['dsname'])
    return [features_table, users_table]
