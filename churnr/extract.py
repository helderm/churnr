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

FEATURES = ['unique_top_types', 'skip_ratio', 'secs_played', 'secs_played_userplaylist', 'secs_played_usercollection', 'secs_played_catalog', 'secs_played_editorialplaylist', 'secs_played_radio', 'secs_played_mix', 'secs_played_soundsof', 'secs_played_personalizedplaylist', 'secs_played_charts', 'secs_played_unknown', 'secs_played_running', 'top_type', 'platform', 'client_type', 'latitude', 'longitude']
FEATURES_IAT = ['iat'] # intertimestep features
FEATURES_SUM = ['sum_secs_played', 'sum_secs_played_userplaylist', 'sum_secs_played_usercollection', 'sum_secs_played_catalog', 'sum_secs_played_editorialplaylist', 'sum_secs_played_radio', 'sum_secs_played_mix', 'sum_secs_played_soundsof', 'sum_secs_played_personalizedplaylist', 'sum_secs_played_charts', 'sum_secs_played_unknown', 'sum_secs_played_running']
FEATURES_COUNT = ['total_streams', 'total_streams_userplaylist', 'total_streams_usercollection', 'total_streams_catalog', 'total_streams_editorialplaylist', 'total_streams_radio', 'total_streams_mix', 'total_streams_soundsof', 'total_streams_personalizedplaylist', 'total_streams_charts', 'total_streams_unknown', 'total_streams_running']


def main(exppath, experiment, dsname):
    """ Extract data from several sources on BigQuery using intermediary tables
        and dump the final output as a file into ../data/raw
    """
    logger.info('Starting extraction of dataset {}...'.format(dsname))

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

        features_table_p = ds.table(name='features_{}_{}_p'.format(conf['experiment'], conf['dsname']))
        features_table = ds.table(name='features_{}_{}_e'.format(conf['experiment'], conf['dsname']))
        user_table_tmp = ds.table(name='users_{}_sampled'.format(conf['experiment']))
        user_table = ds.table(name='users_{}_{}'.format(conf['experiment'], conf['dsname']))

        if features_table.exists():
            features_table.delete()

        if dsname != '56_30d':
            if features_table_p.exists():
                features_table_p.delete()

            features_table_p_in = ds.table(name='features_{}_56_30d_p'.format(conf['experiment']))
            features_table_p, jobs = filter_time_windows(features_table_p, features_table_p_in, ds, client, conf)
            wait_for_jobs(jobs)

        # calculate all inter timesteps features
        features_table, jobs = fetch_intertimestep_features(features_table, features_table_p, ds, client, conf)
        wait_for_jobs(jobs)

        # build feature set
        features_table_days, jobs, timesplits = fetch_features(features_table, ds, client, conf)
        wait_for_jobs(jobs)

        # join separate daily feature tables into a single one again
        features_table, jobs = join_features(features_table, features_table_days, ds, client, conf)
        wait_for_jobs(jobs)
        for f in features_table_days:
            f.delete()

        # calculate the churn label for each user and each timestep
        user_table, jobs = calculate_churn(user_table, user_table_tmp, features_table, ds, client, conf)
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


def filter_time_windows(ftable, ftable_in, ds, client, conf):
    totaldays = conf['obsdays'] + conf['preddays']

    enddate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')
    totaldays = conf['obsdays'] + conf['preddays']

    startdate = enddate - dt.timedelta(days=totaldays)
    starttime = get_utctimestamp(startdate)
    endtime = get_utctimestamp(enddate)

    query = """ \
        SELECT
            *
        FROM
            `{project}.{dataset}.{table}`
        where timestampp >= {starttime} and timestampp <= {endtime}
    """.format(project=conf['project'], dataset=ds.name, table=ftable_in.name, starttime=starttime, endtime=endtime)

    jobname = 'features_filterobs_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = ftable
    job.allow_large_results = True
    job.use_legacy_sql = False
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return ftable, [job]


def fetch_features(ftable_raw, ds, client, conf):
    """ Fetch the features for each user session, spliting into timesplits """

    totaldays = conf['obsdays'] + conf['preddays']
    timesplits = conf['timesplits']

    currdate = dt.datetime.strptime(conf['enddate']+'060000', '%Y%m%d%H%M%S')

    jobs = []
    features_tables = []
    tsl = []

    # query user features
    select_inner = ''
    select = ''
    for f in FEATURES + FEATURES_IAT:
        if f == 'unique_top_types':
            select += 'COUNT(DISTINCT(r.top_type)) AS unique_top_types,'
        elif f == 'skip_ratio':
            select += 'SUM(CAST(skipped as INT64)) / COUNT(skipped) AS skip_ratio,'
            select_inner += 'skipped,'
        elif f in ['top_type', 'platform', 'latitude', 'longitude', 'client_type']:
            select += 'ANY_VALUE(t.{feat}) as {feat},'.format(feat=f)
        elif f == 'secs_played_userplaylist':
            select += 'avg(case when r.top_type = "UserPlaylist" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "UserPlaylist" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_usercollection':
            select += 'avg(case when r.top_type = "UserCollection" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "UserCollection" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_catalog':
            select += 'avg(case when r.top_type = "Catalog" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Catalog" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_editorialplaylist':
            select += 'avg(case when r.top_type = "EditorialPlaylist" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "EditorialPlaylist" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_radio':
            select += 'avg(case when r.top_type = "Radio" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Radio" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_mix':
            select += 'avg(case when r.top_type = "Mix" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Mix" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_soundsof':
            select += 'avg(case when r.top_type = "SoundsOf" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "SoundsOf" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_personalizedplaylist':
            select += 'avg(case when r.top_type = "PersonalizedPlaylist" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "PersonalizedPlaylist" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_charts':
            select += 'avg(case when r.top_type = "Charts" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Charts" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_unknown':
            select += 'avg(case when r.top_type = "Unknown" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Unknown" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        elif f == 'secs_played_running':
            select += 'avg(case when r.top_type = "Running" then secs_played else 0 end) as {feat}, ifnull(stddev(case when r.top_type = "Running" then secs_played else 0 end),0) as {feat}_std,'.format(feat=f)
        else:
            select += 'avg({feat}) as {feat}, ifnull(stddev({feat}),0) as {feat}_std,'.format(feat=f)
            select_inner += '{feat},'.format(feat=f)

    for f in FEATURES_SUM:
        select += 'MAX({feat}) as {feat},'.format(feat=f)
        select_inner += '{feat},'.format(feat=f)

    for f in FEATURES_COUNT:
        if f == 'total_streams':
            select += 'COUNT(*) as {feat},'.format(feat=f)
        elif f == 'total_streams_userplaylist':
            select += 'SUM(case when r.top_type = "UserPlaylist" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_usercollection':
            select += 'SUM(case when r.top_type = "UserCollection" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_catalog':
            select += 'SUM(case when r.top_type = "Catalog" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_editorialplaylist':
            select += 'SUM(case when r.top_type = "EditorialPlaylist" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_radio':
            select += 'SUM(case when r.top_type = "Radio" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_mix':
            select += 'SUM(case when r.top_type = "Mix" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_soundsof':
            select += 'SUM(case when r.top_type = "SoundsOf" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_personalizedplaylist':
            select += 'SUM(case when r.top_type = "PersonalizedPlaylist" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_charts':
            select += 'SUM(case when r.top_type = "Charts" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_unknown':
            select += 'SUM(case when r.top_type = "Unknown" then 1 else 0 end) as {feat},'.format(feat=f)
        elif f == 'total_streams_running':
            select += 'SUM(case when r.top_type = "Running" then 1 else 0 end) as {feat},'.format(feat=f)

    for i in range(totaldays):

        # calculate the time splits
        currtimeslot_start = get_utctimestamp(currdate)
        datestr = currdate.strftime('%Y%m%d')

        features_table = ds.table(name='features_{}_{}_{}_tmp'.format(conf['experiment'], conf['dsname'], datestr))
        if features_table.exists():
            features_table.delete()

        features_tables.append(features_table)
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
                    user_id,
                    top_type
                  FROM
                    `{project}.{dataset}.{table}`
                  WHERE
                    timestampp >= {startslot} AND
                    timestampp < {endslot}),
		  user_top_features AS (
		  SELECT
		    user_id,
		    top_type,
		    platform,
		    client_type,
		    case when latitude != 999.9 then latitude else NULL end as latitude,
		    case when longitude != 999.9 then longitude else NULL end as longitude,
		    RANK() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC) AS rank
		  FROM
                    `{project}.{dataset}.{table}`
		  WHERE
                    timestampp >= {startslot} AND
                    timestampp < {endslot}
		  GROUP BY
		    user_id,
		    top_type,
	            platform,
                    client_type,
                    latitude,
                    longitude)
                SELECT r.user_id, CAST({timesplit} AS INT64) as time, {select} false as backfill \
                FROM user_raw_features r
                JOIN user_top_features t
                ON r.user_id=t.user_id
                GROUP BY r.user_id\
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

        if (i+1) % 15 == 0:
            wait_for_jobs(jobs)
            jobs = []

        currdate = currdate - dt.timedelta(days=1)

    # return the timesplits that will be used for training
    tsl = sorted(tsl)[:-((conf['actdays']+conf['preddays'])*conf['timesplits'])]
    return features_tables, jobs, tsl


def join_features(ftable_out, ftables_days, ds, client, conf):
    """ join daily feature tables into a single one  """

    logger.info('Joining feature tables...')

    features_table = ds.table(name='features_{}_{}_{}'.format(conf['experiment'], conf['dsname'], conf['enddate']))
    if features_table.exists():
        features_table.delete()

    query = ''
    for table in ftables_days:
        query += """SELECT * FROM `{project}.{dataset}.{table}`
            UNION ALL """.format(project=conf['project'], dataset=ds.name, table=table.name)
    query = query[:-11]

    jobname = 'join_features_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = ftable_out
    job.allow_large_results = True
    job.use_legacy_sql = False
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return ftable_out, [job]


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
        elif f in ['top_type', 'platform', 'client_type']:
            select_query += '"Unknown" as {feat},'.format(feat=f)
        elif f in ['latitude', 'longitude']:
            select_query += 'cast(null as float64) as {feat},'.format(feat=f)
        elif f == 'unique_top_types':
            select_query += '0 as {feat},'.format(feat=f)
        else:
            select_query += '0.0 as {feat}, 0.0 as {feat}_std,'.format(feat=f)

    for f in FEATURES_COUNT:
        select_query += '0 as {feat},'.format(feat=f)

    for f in FEATURES_SUM:
        select_query += '0 as {feat},'.format(feat=f)

    """for f in FEATURES_SUM:
        if f == 'sum_secs_played_userplaylist':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_userplaylist),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_usercollection':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_usercollection),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_catalog':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_catalog),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_editorialplaylist':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_editorialplaylist),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_radio':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_radio),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_mix':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_mix),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_soundsof':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_soundsof),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_personalizedplaylist':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_personalizedplaylist),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_charts':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_charts),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_unknown':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_unknown),0) as {feat},'.format(feat=f)
        elif f == 'sum_secs_played_running':
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played_running),0) as {feat},'.format(feat=f)
        else:
            select_query += 'ifnull(MAX(countt*max_prev_sum_secs_played),0) as {feat},'.format(feat=f)
    """
    logger.info('Backfilling features...')
    for i, ts in enumerate(timesplits):
        query = """
	    SELECT
              {select}
	      user_id,
              CAST({timesplit} as INT64) as time,
              TRUE as backfill
	    FROM
              (SELECT us.user_id
               FROM `{project}.{dataset}.{utable}` us
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

        if (i+1) % 2 == 0:
            wait_for_jobs(jobs)
            jobs = []

    return jobs


def calculate_churn(utable_out, users_sampled, features_table, ds, client, conf):
    """ Calculate the users that will churn by checking the churn window """
    logger.info('Calculating churning and non-churning labels...')

    jobs = []

    project = conf['project']
    preddays = conf['preddays']
    enddate = conf['enddate']
    predstart = dt.datetime.strptime(enddate+'060000', '%Y%m%d%H%M%S') - dt.timedelta(days=preddays - 1)

    if utable_out.exists():
        utable_out.delete()

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
    job.destination = utable_out
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
    job.destination = utable_out
    job.allow_large_results = True
    job.write_disposition = 'WRITE_APPEND'
    job.use_legacy_sql = False
    job.begin()
    jobs.append(job)

    return utable_out, jobs


def fetch_intertimestep_features(ftable_out, ftable_in, ds, client, conf):
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
            select += 'ifnull(SUM(secs_played) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played,'
        elif f == 'sum_secs_played_userplaylist':
            select += 'ifnull(SUM(case when top_type = "UserPlaylist" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_userplaylist,'
        elif f == 'sum_secs_played_usercollection':
            select += 'ifnull(SUM(case when top_type = "UserCollection" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_usercollection,'
        elif f == 'sum_secs_played_catalog':
            select += 'ifnull(SUM(case when top_type = "Catalog" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_catalog,'
        elif f == 'sum_secs_played_editorialplaylist':
            select += 'ifnull(SUM(case when top_type = "EditorialPlaylist" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_editorialplaylist,'
        elif f == 'sum_secs_played_radio':
            select += 'ifnull(SUM(case when top_type = "Radio" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_radio,'
        elif f == 'sum_secs_played_mix':
            select += 'ifnull(SUM(case when top_type = "Mix" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_mix,'
        elif f == 'sum_secs_played_soundsof':
            select += 'ifnull(SUM(case when top_type = "SoundsOf" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_soundsof,'
        elif f == 'sum_secs_played_personalizedplaylist':
            select += 'ifnull(SUM(case when top_type = "PersonalizedPlaylist" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_personalizedplaylist,'
        elif f == 'sum_secs_played_charts':
            select += 'ifnull(SUM(case when top_type = "Charts" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_charts,'
        elif f == 'sum_secs_played_unknown':
            select += 'ifnull(SUM(case when top_type = "Unknown" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_unknown,'
        elif f == 'sum_secs_played_running':
            select += 'ifnull(SUM(case when top_type = "Running" then secs_played else 0 end) OVER(PARTITION BY user_id ORDER BY timestampp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) as sum_secs_played_running,'
        else:
            raise Exception()

    query = """ \
        SELECT
            {select}
            *
        FROM
            `{project}.{dataset}.{table}`
    """.format(project=conf['project'], dataset=ds.name, table=ftable_in.name, select=select)

    jobname = 'features_intertimestep_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = ftable_out
    job.allow_large_results = True
    job.use_legacy_sql = False
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return ftable_out, [job]


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

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname)


