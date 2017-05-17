#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import uuid
import argparse
from google.cloud import bigquery as bq
import google.cloud.storage as gcs
import datetime as dt
import time
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.process')

FEATURES = ['skip_ratio', 'secs_played']
FEATURES_IAT = ['iat'] # intertimestep features
FEATURES_SUM = ['sum_secs_played']
FEATURES_COUNT = ['total_streams']

FEATURES_ALL = FEATURES + FEATURES_IAT + FEATURES_SUM + FEATURES_COUNT

def main(exppath, experiment, dsname):
    """ Process the data in a format suitable for training """
    logger.info('Starting data processing for dataset {}...'.format(dsname))

    with open(exppath) as f:
        exp = json.load(f)

    # load experiment configuration
    keys = ['project', 'dataset', 'enddate', 'timesplits', 'gsoutput', 'testsize', 'retainedshare']
    conf = {}
    for key in keys:
        conf[key] = exp[experiment]['datasets'][dsname][key] if key in exp[experiment]['datasets'][dsname] else exp[experiment]['datasets']['global'][key]
    if conf['enddate'] == 'yesterday':
        conf['enddate'] = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
    conf['experiment'] = experiment
    conf['dsname'] = dsname

    gspath = os.path.join(conf['gsoutput'], experiment, dsname)

    try:
        client = bq.Client(project=conf['project'])

        # create dataset
        ds = client.dataset(conf['dataset'])
        if not ds.exists():
            ds.location = 'EU'
            ds.create()

        features_table = ds.table(name='features_{}_{}_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
        users_table = ds.table(name='users_{}_{}_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))

        # normalize feature values
        features_table, jobs = normalize_features(features_table, ds, client, conf)
        wait_for_jobs(jobs)

        # aggregate the rows into one per user with all its timesteps
        features_table, jobs = aggregate_features(features_table, users_table, ds, client, conf)
        wait_for_jobs(jobs)

        # split into train, val and test sets
        train_features_table, test_features_table, val_features_table, jobs = train_test_val_split(features_table, ds, client, conf)
        wait_for_jobs(jobs)

        # undersample the training set
        train_us_features_table, jobs = undersample_features(train_features_table, ds, client, conf)
        wait_for_jobs(jobs)

        # export to gcs
        tables = [train_features_table, train_us_features_table, test_features_table, val_features_table]
        jobs = dump_features_to_gcs(tables, gspath, conf['project'], client)
        wait_for_jobs(jobs)

        logger.info('Data processing of dataset {} done!'.format(dsname))

    except Exception as e:
        logger.exception(e)
        raise e


def normalize_features(ftable, ds, client, conf):
    """ Normalize the feature tables with a zero mean and unit variance """
    logger.info('Normalizing features...')

    select_with = ''
    select = ''
    select_union = ''
    for f in FEATURES_ALL:
        select_with += 'AVG({feat}) AS {feat}_avg, STDDEV_POP({feat}) AS {feat}_stddev,'.format(feat=f)
        select += '( ({feat} - {feat}_avg ) / {feat}_stddev ) AS {feat},'.format(feat=f)
        select_union += '{feat},'.format(feat=f if f != 'sum_secs_played' else '( ({feat} - (SELECT {feat}_avg FROM avgs_stdvs)) / (SELECT {feat}_stddev FROM avgs_stdvs) ) AS {feat}'.format(feat=f))

    nftable = ds.table(name='features_{}_{}_n_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if nftable.exists():
        nftable.delete()

    query = """\
     WITH
       avgs_stdvs AS (
       SELECT {select_with} 0
       FROM
         `{project}.{dataset}.{table}`
       WHERE
         backfill = FALSE )
     SELECT
       {select}
       user_id,
       time
     FROM
       `{project}.{dataset}.{table}`,
       avgs_stdvs
     WHERE
         backfill = FALSE
     UNION ALL
     SELECT
         {select_union}
        user_id,
        time
     FROM
       `{project}.{dataset}.{table}`,
       avgs_stdvs
     WHERE
         backfill = TRUE
    """.format(select_with=select_with, select=select, select_union=select_union, table=ftable.name, project=conf['project'], dataset=ds.name)

    jobname = 'features_norm_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = nftable
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    return nftable, [job]


def aggregate_features(ftable, utable, ds, client, conf):
    """ """
    logger.info('Aggregating features...')

    agftable = ds.table(name='features_{}_{}_a_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if agftable.exists():
        agftable.delete()

    select = ''
    for f in FEATURES_ALL:
        select += 'ARRAY_AGG(n.{feat}) as {feat},'.format(feat=f)

    query = """
        SELECT
            {select}
            ARRAY_AGG(n.time) as times,
            ANY_VALUE(u.churn) as churn,
            u.user_id
        FROM
            `{project}.{dataset}.{ftable}` as n
        JOIN
            `{project}.{dataset}.{utable}` as u
        ON n.user_id = u.user_id
        GROUP BY
            u.user_id
    """.format(project=conf['project'], dataset=ds.name, ftable=ftable.name,
                utable=utable.name, select=select)

    jobname = 'features_agg_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = agftable
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    return agftable, [job]


def train_test_val_split(ftable, ds, client, conf):
    """ """
    logger.info('Splitting features into train, val and test sets...')

    share = conf['testsize']
    jobs = []

    testft = ds.table(name='features_{}_{}_te_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if testft.exists():
        testft.delete()

    query = """
        SELECT
            *
        FROM
            `{project}.{dataset}.{ftable}`
        WHERE
           MOD(ABS(FARM_FINGERPRINT(SHA1(user_id))), {share}) = 0
    """.format(project=conf['project'], dataset=ds.name, ftable=ftable.name,
                share=int(1/share))

    jobname = 'features_split_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = testft
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    jobs.append(job)

    valft = ds.table(name='features_{}_{}_val_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if valft.exists():
        valft.delete()

    query = """
        SELECT
            *
        FROM
            `{project}.{dataset}.{ftable}`
        WHERE
           MOD(ABS(FARM_FINGERPRINT(SHA1(user_id))), {share}) = 1
    """.format(project=conf['project'], dataset=ds.name, ftable=ftable.name,
                share=int(1/share))

    jobname = 'features_split_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = valft
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    jobs.append(job)

    trft = ds.table(name='features_{}_{}_tr_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if trft.exists():
        trft.delete()

    query = """
        SELECT
            *
        FROM
            `{project}.{dataset}.{ftable}`
        WHERE
           MOD(ABS(FARM_FINGERPRINT(SHA1(user_id))), {share}) NOT IN (0,1)
    """.format(project=conf['project'], dataset=ds.name, ftable=ftable.name,
                share=int(1/share))

    jobname = 'features_split_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = trft
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    jobs.append(job)

    return trft, testft, valft, jobs


def undersample_features(ftable, ds, client, conf):
    """ """
    logger.info('Undersampling...')
    share = conf['retainedshare']

    usft = ds.table(name='features_{}_{}_trus_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if usft.exists():
        usft.delete()

    query = """
        SELECT
          *
        FROM
          `{project}.{dataset}.{table}`
        WHERE
          MOD(ABS(FARM_FINGERPRINT(MD5(user_id))), CAST(1/(
              SELECT
                SUM(churn) / COUNT(churn)
              FROM
                `{project}.{dataset}.{table}`) AS INT64)) < {share}
          AND churn = 0
        UNION ALL
        SELECT
          *
        FROM
          `{project}.{dataset}.{table}`
        WHERE
          churn = 1
    """.format(project=conf['project'], dataset=ds.name, table=ftable.name,
                share=share)

    jobname = 'features_undersample_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = usft
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    return usft, [job]


def dump_features_to_gcs(ft_tables, dest, project, client):
    """ Dump generated tables as files on Google Cloud Storage"""
    logger.info('Dumping {} tables to {}...'.format(len(ft_tables), dest))

    gs_client = gcs.Client(project)
    split_uri = dest.split('/')
    filepath = '/'.join(split_uri[3:])
    bucket_name = split_uri[2]
    bucket = gcs.Bucket(gs_client, bucket_name)

    jobs = []
    for ft_table in ft_tables:
        filename_shard = ft_table.name + '{0:012d}'.format(0)
        blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)
        if blob.exists():
            count = 0
            while blob.exists():
                logger.info(' -- Removing blob {}'.format(blob.path))
                blob.delete()

                count += 1
                filename_shard = ft_table.name + '{0:012d}'.format(count)
                blob = gcs.Blob(name=os.path.join(filepath, filename_shard), bucket=bucket)

        path = dest + '/' + ft_table.name + '*'
        jobname = 'features_dump_job_' + str(uuid.uuid4())
        job = client.extract_table_to_storage(jobname, ft_table, path)
        job.destination_format = 'NEWLINE_DELIMITED_JSON'
        job.begin()

        jobs.append(job)

    return jobs


def wait_for_jobs(jobs):
    """ wait for async GCP jobs to finish """
    logger.info("Waiting for {} jobs to finish...".format(len(jobs)))

    cntr = 0
    for job in jobs:
        if cntr % 10 == 0:
            logger.info('{} out of {} jobs finished...'.format(cntr, len(jobs)))
        cntr += 1

        numtries = 9000
        job.reload()
        while job.state != 'DONE':
            time.sleep(10)
            job.reload()
            numtries -= 1

            if numtries == 0:
                logger.error('Job {} timed out! Aborting!'.format(job.name))
                raise Exception('Async job timed out')

        if job.errors:
            logger.error('Job {} failed! Aborting!'.format(job.name))
            raise Exception('Async job failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processer')
    parser.add_argument('--exppath', default='./experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset being transformed')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname)


