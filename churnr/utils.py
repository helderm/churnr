# -*- coding: utf-8 -*-
import os
import shutil
import logging
import datetime as dt
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

        blob = gcs.Blob(name=os.path.join(filepath, filename), bucket=bucket)

        local_file = os.path.join(datapath, filename)
        with open(local_file, 'wb') as f:
            blob.download_to_file(f)


def get_table_names(conf):

    currdate = dt.datetime.strptime(conf['enddate']+'000000', '%Y%m%d%H%M%S') - dt.timedelta(days=conf['obsdays'] + conf['preddays'] - 1)

    tablenames = []
    ft_table_name = 'features_{}_{}_'.format(conf['experiment'], conf['dsname'])
    for i in range(conf['obsdays']):
        datestr = currdate.strftime('%Y%m%d')

        for j in range(conf['timesplits']):
            ft_table = ft_table_name + '{split}_{currdate}'.format(currdate=datestr, split=j)
            tablenames.append(ft_table)

        currdate = currdate + dt.timedelta(days=1)

    tablenames.append('users_{}_{}'.format(conf['experiment'], conf['dsname']))

    return tablenames

