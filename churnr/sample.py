#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import uuid
import argparse
from google.cloud import bigquery as bq
import datetime as dt
import time
import json


SECS_IN_DAY = 86400

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.extract')

FEATURES = ['skip_ratio', 'secs_played']
FEATURES_IAT = ['iat'] # intertimestep features
FEATURES_SUM = ['sum_secs_played']
FEATURES_COUNT = ['total_streams']

def main(exppath, experiment, dsname, sampleusers):
    """ Extract data from several sources on BigQuery using intermediary tables
        and dump the final output as a file into ../data/raw
    """
    logger.info('Starting BigQuery data extraction...')

    with open(exppath) as f:
        exp = json.load(f)

    # load experiment configuration
    keys = ['project', 'dataset', 'enddate', 'obsdays', 'preddays', 'shareusers', 'timesplits', 'gsoutput', 'rawpath', 'actdays']
    conf = {}
    for key in keys:
        conf[key] = exp[experiment]['datasets'][dsname][key] if key in exp[experiment]['datasets'][dsname] else exp[experiment]['datasets']['global'][key]
    if conf['enddate'] == 'yesterday':
        conf['enddate'] = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
    conf['experiment'] = experiment
    conf['dsname'] = dsname

    try:
        client = bq.Client(project=conf['project'])

        # create dataset
        ds = client.dataset(conf['dataset'])
        if not ds.exists():
            ds.location = 'EU'
            ds.create()

        # randomly sample users who are active on the last week
        if sampleusers:
            user_table_tmp, jobs = fetch_user_samples(ds, client, conf)
            wait_for_jobs(jobs)
            user_table_tmp, jobs = add_user_info(user_table_tmp, ds, client, conf)
            wait_for_jobs(jobs)
        else:
            user_table_tmp = ds.table(name='users_{}_sampled'.format(conf['experiment']))

        # filter features from raw sources
        features_table_raw, jobs = sample_raw_features(user_table_tmp, ds, client, conf)
        wait_for_jobs(jobs)

        logger.info('Dataset {} sampled successfully!'.format(dsname))

    except Exception as e:
        logger.exception(e)
        raise e


def fetch_user_samples(ds, client, conf):
    """ Fetch a random sample of users based on a user_id hash """

    logger.info('Fetching user samples...')

    if conf['obsdays'] < conf['actdays']:
        raise Exception("Obs window smaller than activation window, this probably wont work!")

    totaldays = conf['actdays'] + conf['preddays']

    # CST timezone
    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S') - dt.timedelta(days=totaldays - 1)

    user_table = ds.table(name='users_{}_sampled'.format(conf['experiment']))
    if user_table.exists():
        user_table.delete()

    query = ''
    for k in range(conf['actdays']):
        datestr = currdate.strftime('%Y%m%d')
        currdate_ts = get_utctimestamp(currdate)
        currdate_end_ts = (currdate_ts + SECS_IN_DAY)

        query += """\
            SELECT DISTINCT(uss{k}.user_id)
            FROM `royalty-reporting.reporting_end_content.reporting_end_content_{date}` as uss{k}
                JOIN `royalty-reporting.reporting_end_content_decrypted.reporting_end_content_decrypted_{date}` as red{k}
                ON uss{k}.user_id=red{k}.user_id and uss{k}.time=red{k}.time
                JOIN `business-critical-data.user_snapshot.user_snapshot_{date}` as usn{k}
                ON uss{k}.user_id=usn{k}.user_id
                WHERE uss{k}.ms_played > 30000 AND uss{k}.platform IN ("android","android-tablet")
                    AND MOD(ABS(FARM_FINGERPRINT(uss{k}.user_id)), {share}) = 0 AND usn{k}.reporting_country IN ("BR", "US", "MX")
                    AND DATE_DIFF(PARSE_DATE("%E4Y%m%d", "{date}"), PARSE_DATE("%E4Y-%m-%d", usn{k}.registration_date), DAY) > {obswindow}
                    AND UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', uss{k}.time)) >= {startslot}
                    AND UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', uss{k}.time)) < {endslot}
                    AND red{k}.play_context_decrypted NOT LIKE "%%:search:%%"
                UNION DISTINCT (""".format(date=datestr, share=int(1 / conf['shareusers']), startslot=currdate_ts,
                    endslot=currdate_end_ts, k=k, obswindow=conf['obsdays'])

        currdate = currdate - dt.timedelta(days=1)
    query = query[:-17]
    query += ')' * (conf['actdays']-1)

    jobname = 'user_ids_sampled_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = user_table
    job.allow_large_results = True
    job.write_disposition = 'WRITE_APPEND'
    job.use_legacy_sql = False
    job.begin()

    return user_table, [job]


def add_user_info(user_table, ds, client, conf):
    """ Add some columns to the user table """

    logger.info('Getting more info about sampled users...')

    totaldays = conf['actdays'] + conf['preddays']
    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S') - dt.timedelta(days=totaldays - 1)
    datestr = currdate.strftime('%Y%m%d')

    query = """\
        SELECT uss.user_id as user_id, DATE_DIFF(PARSE_DATE("%E4Y%m%d", "{date}"), PARSE_DATE("%E4Y-%m-%d", usn.registration_date), DAY) as days_since_registration,
            usn.product as product, usn.reporting_country as reporting_country
        FROM `{project}.{dataset}.{table}` as uss
            JOIN `business-critical-data.user_snapshot.user_snapshot_{date}` as usn
            ON uss.user_id=usn.user_id
            """.format(date=datestr, project=conf['project'], dataset=ds.name, table=user_table.name)

    jobname = 'users_info_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = user_table
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.use_legacy_sql = False
    job.begin()

    return user_table, [job]


def sample_raw_features(users_table, ds, client, conf):
    """ Filter all feature tables into a single one """

    logger.info('Filtering source feature tables...')

    features_table = ds.table(name='features_{}_{}_s'.format(conf['experiment'], conf['dsname']))
    if features_table.exists():
        features_table.delete()

    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')
    totaldays = conf['obsdays'] + conf['preddays']

    select = ''
    for f in FEATURES:
        if f == 'skip_ratio':
            select += 'r.skipped,'
        elif f == 'secs_played':
            select += 'CAST(r.ms_played/1000 as INT64) as secs_played,'
        else:
            select += 'r.{feat},'.format(feat=f)

    jobs = []
    for i in range(totaldays):
        datestr = currdate.strftime('%Y%m%d')
        logger.info('Filtering day {}'.format(datestr))

        query = """\
              SELECT
                r.user_id,
                {select}
                UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', r.time)) AS timestampp,
                r.client_type,
                r.product_type,
                r.platform,
                r.hardware,
                r.location.latitude,
                r.location.longitude,
                r.time as timestr,
                d.play_context_decrypted
              FROM
                `royalty-reporting.reporting_end_content.reporting_end_content_{date}` r
              JOIN
                `royalty-reporting.reporting_end_content_decrypted.reporting_end_content_decrypted_{date}` d
              ON r.user_id=d.user_id AND r.time=d.time
              WHERE
                r.user_id IN (SELECT user_id FROM `{project}.{dataset}.{table}`)
                AND r.ms_played > 0.0
                AND d.play_context_decrypted NOT LIKE "%%:search:%%"
        """.format(date=datestr, project=conf['project'], dataset=ds.name, table=users_table.name,
                    select=select)

        jobname = 'features_filter_job_' + str(uuid.uuid4())
        job = client.run_async_query(jobname, query)
        job.destination = features_table
        job.allow_large_results = True
        job.write_disposition = 'WRITE_APPEND'
        job.use_legacy_sql = False
        job.begin()

        if (i+1) % 15 == 0:
            wait_for_jobs(jobs)
            jobs = []

        jobs.append(job)
        currdate = currdate - dt.timedelta(days=1)

    return features_table, jobs


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


def get_utctimestamp(dtobj):
    from calendar import timegm
    iso_string = dtobj.strftime('%Y%m%d%H%M%S') + 'Z'
    return timegm(time.strptime(iso_string.replace('Z', 'GMT'), '%Y%m%d%H%M%S%Z'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data extracter')
    parser.add_argument('--exppath', default='./experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset being transformed')
    parser.add_argument('--sampleusers', default=False, help='If True, will fetch new user samples', action='store_true')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, sampleusers=args.sampleusers)


