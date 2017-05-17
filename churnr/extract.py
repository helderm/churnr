#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import uuid
import argparse
from google.cloud import bigquery as bq
import datetime as dt
import time
import math
import json


SECS_IN_DAY = 86400

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.extract')

FEATURES = ['skip_ratio', 'secs_played']
FEATURES_IAT = ['iat'] # intertimestep features
FEATURES_SUM = ['sum_secs_played']
FEATURES_COUNT = ['total_streams']

def main(exppath, experiment, dsname, hddump, sampleusers):
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
        features_table_raw, jobs = join_feature_tables(user_table_tmp, ds, client, conf)
        wait_for_jobs(jobs)

        # calculate all inter timesteps features
        features_table_raw, jobs = fetch_intertimestep_features(features_table_raw, ds, client, conf)
        wait_for_jobs(jobs)

        # build feature set
        features_table, jobs, timesplits = fetch_features(features_table_raw, ds, client, conf)
        wait_for_jobs(jobs)

        # calculate the churn label for each user and each timestep
        user_table, jobs = calculate_churn(user_table_tmp, features_table, ds, client, conf)
        wait_for_jobs(jobs)

        # filter out features values in the prediction window
        features_table, jobs = filter_features_table(features_table, max(timesplits), ds, client, conf)
        wait_for_jobs(jobs)

        # backfill missing users on feature tables
        jobs = backfill_missing_users(user_table, features_table, timesplits, ds, client, conf)
        wait_for_jobs(jobs)

        logger.info('Dataset {} extracted successfully!'.format(dsname))

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
                JOIN `business-critical-data.user_snapshot.user_snapshot_{date}` as usn{k}
                ON uss{k}.user_id=usn{k}.user_id
                WHERE uss{k}.ms_played > 30000 AND uss{k}.platform IN ("android","android-tablet")
                    AND MOD(ABS(FARM_FINGERPRINT(uss{k}.user_id)), {share}) = 0 AND usn{k}.reporting_country IN ("BR", "US", "MX")
                    AND DATE_DIFF(PARSE_DATE("%E4Y%m%d", "{date}"), PARSE_DATE("%E4Y-%m-%d", usn{k}.registration_date), DAY) > {obswindow}
                    AND UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', uss{k}.time)) >= {startslot}
                    AND UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', uss{k}.time)) < {endslot}
                UNION DISTINCT (""".format(date=datestr, share=int(1 / conf['shareusers']), startslot=currdate_ts,
                    endslot=currdate_end_ts, k=k, obswindow=conf['obsdays'])

        currdate = currdate + dt.timedelta(days=1)
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


def fetch_features(ftable_raw, ds, client, conf):
    """ Fetch the features for each user session, splsum_secs_playeding into timesplits """

    totaldays = conf['obsdays'] + conf['preddays']
    timesplits = conf['timesplits']

    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')

    jobs = []
    tsl = []
    features_table = ds.table(name='features_{}_{}_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if features_table.exists():
        features_table.delete()

    # query user features
    select_inner = ''
    select = ''
    for f in FEATURES + FEATURES_IAT:
        if f == 'skip_ratio':
            select += 'SUM(CAST(skipped as INT64)) / COUNT(skipped) AS skip_ratio,'
            select_inner += 'skipped,'.format(feat=f)
        else:
            select += 'avg({feat}) as {feat}, ifnull(stddev({feat}),0) as {feat}_std,'.format(feat=f)
            select_inner += '{feat},'.format(feat=f)

    for f in FEATURES_SUM:
        select += 'MAX({feat}) as {feat},'.format(feat=f)
        select_inner += '{feat},'.format(feat=f)

    for f in FEATURES_COUNT:
        select += 'COUNT(user_id) as {feat},'.format(feat=f)

    for i in range(totaldays):

        # calculate the time splits
        currtimeslot_start = get_utctimestamp(currdate)
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Fetching features for day {}'.format(datestr))

        for j in range(timesplits):
            if j+1 != timesplits:
                # calculate the timeslot range
                currtimeslot_end = currtimeslot_start + math.ceil(SECS_IN_DAY / timesplits) + 0.999
            else:
                # get the remaining secs lost due to rounding on the last slot. Will it be biased?
                currtimeslot_end = get_utctimestamp(currdate) + SECS_IN_DAY + 0.999


            query = """\
                WITH
                  user_raw_features AS (
                  SELECT
                    {select_inner}
                    user_id
                  FROM
                    `{project}.{dataset}.{table}`
                  WHERE
                    timestampp >= {startslot} AND
                    timestampp < {endslot})

                SELECT user_id, CAST({timesplit} AS INT64) as time, {select} false as backfill \
                FROM user_raw_features
                GROUP BY user_id\
            """.format(project=conf['project'], dataset=ds.name, table=ftable_raw.name,
                    startslot=int(currtimeslot_start), endslot=int(currtimeslot_end), select=select,
                    timesplit=currtimeslot_start, select_inner=select_inner)

            jobname = 'features_job_' + str(uuid.uuid4())
            job = client.run_async_query(jobname, query)
            job.destination = features_table
            job.allow_large_results = True
            job.use_legacy_sql = False
            job.write_disposition = 'WRITE_APPEND'
            job.begin()

            jobs.append(job)
            tsl.append(currtimeslot_start)

            currtimeslot_start += int(math.ceil(SECS_IN_DAY / timesplits))

        if (i+1) % 4 == 0:
            wait_for_jobs(jobs)
            jobs = []

        currdate = currdate - dt.timedelta(days=1)

    tsl = sorted(tsl)[:-(conf['preddays']*conf['timesplits'])]
    return features_table, jobs, tsl


def filter_features_table(features_table, max_timesplit, ds, client, conf):
    """ filter out feature values in the prediction window """

    logger.info('Filtering out features in the prediction window...')

    query = """
        SELECT * FROM `{project}.{dataset}.{table}`
        WHERE time <= {timesplit}
    """.format(project=conf['project'], dataset=ds.name, table=features_table.name, timesplit=max_timesplit)

    jobname = 'filter_features_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = features_table
    job.allow_large_results = True
    job.use_legacy_sql = False
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return features_table, [job]


def backfill_missing_users(users_table, features_table, timesplits, ds, client, conf):
    """ add the remaining missing users to the feature tables """

    jobs = []
    project = conf['project']

    select_query = ''
    for f in FEATURES + FEATURES_IAT:
        if f == 'skip_ratio':
            select_query += '0.0 as {feat},'.format(feat=f)
        else:
            select_query += '0.0 as {feat}, 0.0 as {feat}_std,'.format(feat=f)

    for f in FEATURES_COUNT:
        select_query += '0 as {feat},'.format(feat=f)

    for f in FEATURES_SUM:
        select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played),0) as {feat},'.format(feat=f)

    logger.info('Backfilling features...')
    for i, ts in enumerate(timesplits):
        query = """
	    SELECT
              {select}
	      user_id,
              CAST({timesplit} as INT64) as time,
              TRUE as backfill
	    FROM
              (SELECT us.user_id,
                COUNTIF(ft.time <= {timesplit}) OVER (PARTITION BY ft.user_id ORDER BY ft.time ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) as countt,
                MAX(ft.sum_secs_played) OVER (PARTITION BY ft.user_id ORDER BY ft.time ROWS BETWEEN 1 PRECEDING and 1 PRECEDING) as max_prev_sum_secs_played
               FROM `{project}.{dataset}.{utable}` us JOIN
                 `{project}.{dataset}.{ftable}` ft ON us.user_id=ft.user_id
	       WHERE
		us.user_id NOT IN (
		  SELECT
		    user_id
		  FROM
		    `{project}.{dataset}.{ftable}`
		  WHERE
		    time = {timesplit} ) )
	    GROUP BY
	      user_id
        """.format(project=project, dataset=ds.name, ftable=features_table.name, timesplit=ts, ds=conf['dsname'], utable=users_table.name, select=select_query)

        jobname = 'backfill_missing_features_job_' + str(uuid.uuid4())
        job = client.run_async_query(jobname, query)
        job.destination = features_table
        job.allow_large_results = True
        job.use_legacy_sql = False
        job.write_disposition = 'WRITE_APPEND'
        job.begin()

        jobs.append(job)

        if (i+1) % 15 == 0:
            wait_for_jobs(jobs)
            jobs = []

    return jobs


def calculate_churn(users_sampled, features_table, ds, client, conf):
    """ Calculate the users that will churn by checking the churn window """
    logger.info('Calculating churning and non-churning labels...')

    jobs = []

    project = conf['project']
    preddays = conf['preddays']
    enddate = conf['enddate']
    predstart = dt.datetime.strptime(enddate+'060000', '%Y%m%d%H%M%S') - dt.timedelta(days=preddays - 1)

    user_table = ds.table(name='users_{}_{}_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if user_table.exists():
        user_table.delete()

    # build the query part that will join all users who had streams in the churn window
    in_query = """
        (   SELECT user_id from `{project}.{dataset}.{table}`
            WHERE time >= {predstart}
                AND secs_played > 0 )
    """.format(project=project, dataset=ds.name, table=features_table.name, predstart=get_utctimestamp(predstart))

    # retained query
    query = """\
        select *, 0 as churn
            from `{project}.{dataset}.{table}`
            where user_id in {in_query}
    """.format(in_query=in_query, project=conf['project'], dataset=ds.name, table=users_sampled.name)

    jobname = 'retained_features_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = user_table
    job.allow_large_results = True
    job.write_disposition = 'WRITE_APPEND'
    job.use_legacy_sql = False
    job.begin()
    jobs.append(job)

    # churners query
    query = """\
        select *, 1 as churn
            from `{project}.{dataset}.{table}`
            where user_id not in {in_query}
    """.format(in_query=in_query, project=conf['project'], dataset=ds.name, table=users_sampled.name)

    jobname = 'churned_features_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = user_table
    job.allow_large_results = True
    job.write_disposition = 'WRITE_APPEND'
    job.use_legacy_sql = False
    job.begin()
    jobs.append(job)

    return user_table, jobs


def join_feature_tables(users_table, ds, client, conf):
    """ Join all feature tables into a single one for later csv exporting """

    logger.info('Filtering source feature tables...')

    features_table = ds.table(name='features_{}_{}_r_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if features_table.exists():
        features_table.delete()

    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')
    totaldays = conf['obsdays'] + conf['preddays']

    select = ''
    select_inner = ''
    for f in FEATURES:
        if f == 'skip_ratio':
            select += 'skipped,'
            select_inner += 'skipped,'
        elif f == 'secs_played':
            select += 'secs_played,'
            select_inner += 'CAST(ms_played/1000 as INT64) as secs_played,'
        else:
            select += '{feat},'.format(feat=f)
            select_inner += '{feat},'.format(feat=f)

    jobs = []
    for i in range(totaldays):
        datestr = currdate.strftime('%Y%m%d')

        query = """\
            WITH filterq AS (
              SELECT
                user_id,
                {select_inner}
                UNIX_SECONDS(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S', time)) AS timestampp,
                time as timestr
              FROM
                `royalty-reporting.reporting_end_content.reporting_end_content_{date}`
              WHERE
                user_id IN (SELECT user_id FROM `{project}.{dataset}.{table}`)
                    AND ms_played > 0.0
            )
            SELECT {select} user_id, timestampp, timestr FROM filterq

        """.format(date=datestr, project=conf['project'], dataset=ds.name, table=users_table.name,
                    select=select, select_inner=select_inner)

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


def fetch_intertimestep_features(features_table, ds, client, conf):
    """ Fetch the inter timesteps feaures """

    logger.info('Calculating temporal features...')

    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')
    totaldays = conf['obsdays'] + conf['preddays']

    startdate = currdate - dt.timedelta(days=totaldays, minutes=1)
    select = ''
    for f in FEATURES_IAT:
        if f == 'iat':
            select += 'timestampp - (LAG(timestampp,1,{startdate}) OVER (PARTITION BY user_id ORDER BY timestampp) ) AS iat,'.format(startdate=get_utctimestamp(startdate))
        else:
            raise Exception()

    for f in FEATURES_SUM:
        if f == 'sum_secs_played':
            select += 'ifnull(SUM(secs_played) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) as sum_secs_played,'
        else:
            raise Exception()

    query = """ \
        SELECT
            {select}
            *
        FROM
            `{project}.{dataset}.{table}`
    """.format(project=conf['project'], dataset=ds.name, table=features_table.name, select=select)

    jobname = 'features_intertimestep_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = features_table
    job.allow_large_results = True
    job.use_legacy_sql = False
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return features_table, [job]


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
    parser.add_argument('--hddump', default=False, help='If True, will dump the tables to the local filesystem', action='store_true')
    parser.add_argument('--sampleusers', default=False, help='If True, will fetch new user samples', action='store_true')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, hddump=args.hddump, sampleusers=args.sampleusers)


