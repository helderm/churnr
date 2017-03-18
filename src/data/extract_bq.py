#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
import uuid
from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery as bq
import google.cloud.storage as gcs
import datetime as dt
import math
import json

SECS_IN_DAY = 86399
#yesterday = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
yesterday = '20170316'

logger = logging.getLogger(__name__)

@click.command()
@click.option('--project', default='user-lifecycle')
@click.option('--dataset', default='helder')
@click.option('--enddate', default=yesterday)
@click.option('--numdays', default=3)
@click.option('--shareusers', default=0.001)
@click.option('--timesplits', default=3)
@click.option('--churndays', default=4)
@click.option('--gsoutput', default='gs://helder/data/raw')
@click.option('--hdoutput', default='../../data/raw')
def main(project, dataset, enddate, numdays, shareusers, timesplits, churndays, gsoutput, hdoutput):
    """ Extract data from several sources on BigQuery using intermediary tables
        and dump the final output as a file into ../data/raw
    """
    logger.info('Starting BigQuery data extraction...')

    # assert args
    #assert numdays > 7

    client = bq.Client(project=project)

    # create dataset
    ds = client.dataset(dataset)
    if not ds.exists():
        ds.location = 'EU'
        ds.create()

    # randomly sample users who are active on the last week
    user_table, jobs = fetch_user_samples(enddate, shareusers, ds, client)
    wait_for_jobs(jobs)

    # make samples distinct
    jobs = distinct_users(user_table, project, ds, client)
    wait_for_jobs(jobs)

    # build feature set
    ft_tables, jobs = fetch_features(user_table, numdays + churndays, enddate, timesplits, project, ds, client)
    wait_for_jobs(jobs)

    # calculate the churn label for each user and each timestep
    jobs = calculate_churn(numdays, churndays, enddate, timesplits, project, ds, client)
    wait_for_jobs(jobs)

    # backfill missing users on feature tables
    jobs = backfill_missing_users(numdays, churndays, enddate, timesplits, project, ds, client)
    wait_for_jobs(jobs)

    # serialize features
    tables = ft_tables[-numdays*timesplits:] + [user_table]
    asw = yes_or_no('Dump files to GCS?')
    if asw:
        jobs = dump_features_to_gcs(tables, gsoutput, client)

    # dump to disk
    if asw and yes_or_no('Dump files to disk?'):
        wait_for_jobs(jobs)
        extract_dataset_to_disk(hdoutput, gsoutput, tables, project)

    # save metadata file with args that generated this dataset
    meta = { 'enddate': enddate, 'timesplits': timesplits, 'project': project, 'dataset': dataset,
            'numdays': numdays, 'churndays': churndays, 'shareusers': shareusers, 'gsoutput': gsoutput }
    with open(os.path.join(hdoutput, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    logger.info('Finished! Data dumped to BigQuery{}'.format(' and to '+gsoutput if asw else ''))


def fetch_user_samples(enddate, shareusers, ds, client):
    """ Fetch a random sample of users based on a user_id hash """

    user_table = ds.table(name='user_ids_sampled_{date}'.format(date=enddate))
    if user_table.exists():
        user_table.delete()

    currdate = dt.datetime.strptime(enddate, '%Y%m%d')
    jobs = []
    for i in range(7):
        datestr = currdate.strftime('%Y%m%d')

        # select a random sample of users based on a user_id hash and some wanted features
        query = """\
            SELECT uss.user_id as user_id, usn.reporting_country as reporting_country, usn.registration_date as registration_date,
                    uss.platform as platform, usn.product as product
                FROM [activation-insights:sessions.features_{date}] as uss
                JOIN [business-critical-data:user_snapshot.user_snapshot_{date}] as usn
                ON uss.user_id = usn.user_id
                WHERE uss.consumption_time > 0.0 AND uss.platform IN ("android","android-tablet")
                    AND ABS(HASH(uss.user_id)) % {share} == 0 AND usn.reporting_country IN ("BR", "US", "MX")
        """.format(date=datestr, share=int(1 / shareusers))

        logger.info('Selecting a random sample of users ({} of users on {})'.format(shareusers, datestr))

        jobname = 'user_ids_sampled_job_' + str(uuid.uuid4())
        job = client.run_async_query(jobname, query)
        job.destination = user_table
        job.allow_large_results = True
        job.write_disposition = 'WRITE_APPEND'
        job.begin()

        jobs.append(job)

        currdate = currdate - dt.timedelta(days=1)

    return user_table, jobs


def distinct_users(user_table, project, ds, client):
    """ Distinct the user table """

    query = """\
       SELECT user_id, reporting_country, registration_date, platform, product FROM [{project}:{dataset}.{table}]
        GROUP BY user_id, reporting_country, registration_date, platform, product\
    """.format(project=project, dataset=ds.name, table=user_table.name)

    logger.info('Fetching distict user ids from samples...')

    jobname = 'user_distinct_samples_job_' + str(uuid.uuid4())
    job = client.run_async_query(jobname, query)
    job.destination = user_table
    job.allow_large_results = True
    job.write_disposition = 'WRITE_TRUNCATE'
    job.begin()

    return [job]


def fetch_features(user_table, numdays, enddate, timesplits, project, ds, client):
    """ Fetch the features for each user session, splitting into timesplits """

    jobs = []
    currdate = dt.datetime.strptime(enddate+'000000', '%Y%m%d%H%M%S')

    ft_tables = []
    ft_table_name = 'user_features_s{sampledate}_'.format(sampledate=enddate)
    for i in range(numdays):

        # calculate the time splits
        currtimeslot_start = currdate.timestamp()
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Fetching features for day {}'.format(datestr))

        for j in range(timesplits):
            if j+1 != timesplits:
                # calculate the timeslot range
                currtimeslot_end = currtimeslot_start + math.ceil(SECS_IN_DAY / timesplits) + 0.999
            else:
                # get the remaining secs lost due to rounding on the last slot. Will it be biased?
                currtimeslot_end = currdate.timestamp() + SECS_IN_DAY + 0.999

            # dump the features table
            ft_table = ds.table(name=ft_table_name+'{split}_{currdate}'.format(currdate=datestr, split=j))
            if ft_table.exists():
                ft_table.delete()

            # query user features
            query = """\
                SELECT  user_id, avg(consumption_time) as consumption_time, avg(session_length) as session_length, avg(skip_ratio) as skip_ratio, avg(unique_pages) as unique_pages, 0 as churn FROM\
                        (SELECT user_id, consumption_time/1000 as consumption_time, session_length, skip_ratio, unique_pages\
                            FROM [activation-insights:sessions.features_{date}]\
                            WHERE user_id IN (SELECT user_id FROM [{project}:{dataset}.{table}])\
                                AND TIMESTAMP_TO_MSEC(TIMESTAMP(start_time)) >= {startslot} AND TIMESTAMP_TO_MSEC(TIMESTAMP(start_time)) < {endslot}
                                AND session_length > 0.0)
                GROUP BY user_id\
            """.format(date=datestr, project=project, dataset=ds.name, table=user_table.name,
                        startslot=int(currtimeslot_start*1000), endslot=int(currtimeslot_end*1000))

            jobname = 'user_features_job_' + str(uuid.uuid4())
            job = client.run_async_query(jobname, query)
            job.destination = ft_table
            job.allow_large_results = True
            job.write_disposition = 'WRITE_APPEND'
            job.begin()

            jobs.append(job)
            ft_tables.append(ft_table)

            currtimeslot_start += math.ceil(SECS_IN_DAY / timesplits)

        currdate = currdate - dt.timedelta(days=1)

    return ft_tables, jobs


def backfill_missing_users(numdays, churndays, enddate, timesplits, project, ds, client):
    """ add the remaining missing users to the feature tables who did in fact churn """

    jobs = []
    currdate = dt.datetime.strptime(enddate+'000000', '%Y%m%d%H%M%S') - dt.timedelta(days=numdays + churndays - 1)

    ft_table_name = 'user_features_s{sampledate}_'.format(sampledate=enddate)
    for i in range(numdays):

        # calculate the time splits
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Backfilling features for day {}'.format(datestr))

        for j in range(timesplits):

            ft_table = ds.table(name=ft_table_name+'{split}_{currdate}'.format(currdate=datestr, split=j))

            # append missing values to feature table
            query = """\
                SELECT  user_id, 0.0 as consumption_time, 0.0 as session_length, 0.0 as skip_ratio, 0.0 as unique_pages, 1 as churn \
                            FROM[{project}:{dataset}.user_ids_sampled_{enddate}]\
                            WHERE user_id NOT IN (SELECT user_id FROM [{project}:{dataset}.{table}])\
            """.format(enddate=enddate, project=project, dataset=ds.name, table=ft_table.name)

            jobname = 'backfill_missing_user_features_job_' + str(uuid.uuid4())
            job = client.run_async_query(jobname, query)
            job.destination = ft_table
            job.allow_large_results = True
            job.write_disposition = 'WRITE_APPEND'
            job.begin()

            jobs.append(job)

        currdate = currdate + dt.timedelta(days=1)

    return jobs


def calculate_churn(numdays, churndays, enddate, timesplits, project, ds, client):
    """ add to the feature tables missing users who are not considered churners yet """

    jobs = []
    currdate = dt.datetime.strptime(enddate+'000000', '%Y%m%d%H%M%S') - dt.timedelta(days=numdays + churndays - 1)

    ft_table_name = 'user_features_s{sampledate}_'.format(sampledate=enddate)
    for i in range(numdays):

        # calculate the time splits
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Checking retention of users on day {}'.format(datestr))

        for j in range(timesplits):

            ft_table = ds.table(name=ft_table_name+'{split}_{currdate}'.format(currdate=datestr, split=j))

            # build the query part that will join all users who had streams in the churn window
            union_query = ''
            for k in range(churndays+1):
                datestr_2 = (currdate + dt.timedelta(days=k)).strftime('%Y%m%d')

                if k > 0:
                    # check all timesplit of future days
                    for l in range(timesplits):
                        union_query += """select user_id from `{project}.{dataset}.user_features_s{enddate}_{split}_{currdate}`
                                    where consumption_time > 0.0 union distinct """.format(enddate=enddate, split=l, currdate=datestr_2, project=project, dataset=ds.name)
                else:
                    # check the future timesplits for the same day
                    for l in range(j+1, timesplits-j):
                        union_query += """select user_id from `{project}.{dataset}.user_features_s{enddate}_{split}_{currdate}`
                                    where consumption_time > 0.0 union distinct """.format(enddate=enddate, split=l, currdate=datestr_2, project=project, dataset=ds.name)

            union_query = union_query[:-16]

            # append sessions from users that did not stream in this timesplit but who are also not churners
            query = """\
                select us.user_id, 0.0 as consumption_time, 0.0 as session_length, 0.0 as skip_ratio, 0.0 as unique_pages, 0 as churn FROM (SELECT user_id FROM `{project}.{dataset}.user_ids_sampled_{enddate}`
                    where user_id not in (select user_id from `{project}.{dataset}.{table}` where session_length > 0.0)) as us
                join ({union}) as su1
                on us.user_id = su1.user_id
            """.format(enddate=enddate, project=project, dataset=ds.name, table=ft_table.name, union=union_query)

            jobname = 'retained_user_features_job_' + str(uuid.uuid4())
            job = client.run_async_query(jobname, query)
            job.destination = ft_table
            job.allow_large_results = True
            job.write_disposition = 'WRITE_APPEND'
            job.use_legacy_sql = False
            job.begin()

            jobs.append(job)

        currdate = currdate + dt.timedelta(days=1)

    return jobs


def dump_features_to_gcs(ft_tables, dest, client):
    """ Dump generated tables as files on Google Cloud Storage"""

    logger.info('Dumping {} tables to {}...'.format(len(ft_tables), dest))

    jobs = []
    for ft_table in ft_tables:
        path = dest + '/' + ft_table.name
        jobname = 'features_dump_job_' + str(uuid.uuid4())
        job = client.extract_table_to_storage(jobname, ft_table, path)
        job.begin()

        jobs.append(job)

    return jobs


def extract_dataset_to_disk(hdoutput, gsoutput, tables, project):
    """ Extract dataset to local files on disk"""

    logger.info('Dumping {} tables to disk at {}...'.format(len(tables), hdoutput))

    client = gcs.Client(project)

    split_uri = gsoutput.split('/')
    bucket_name = split_uri[2]

    bucket = gcs.Bucket(client, bucket_name)

    filepath = '/'.join(split_uri[3:])
    for table in tables:
        filename = table.name

        blob = gcs.Blob(name=os.path.join(filepath, filename), bucket=bucket)

        local_file = os.path.join(hdoutput, filename)
        with open(local_file, 'wb') as f:
            blob.download_to_file(f)


def wait_for_jobs(jobs):
    """ wait for async GCP jobs to finish """
    logger.info("Waiting for {} jobs to finish...".format(len(jobs)))

    import time
    for job in jobs:
        numtries = 35
        job.reload()
        while job.state != 'DONE':
            time.sleep(3)
            job.reload()
            numtries -= 1

            if numtries == 0:
                logger.error('Job {} timed out! Aborting!'.format(job.name))
                raise Exception('Async job timed out')

        if job.errors:
            logger.error('Job {} failed! Aborting!'.format(job.name))
            raise Exception('Async job failed')


def yes_or_no(question):
    """ Simple yes or no user prompt """
    reply = str(input(question+' (y/N): ')).lower().strip()
    if len(reply) and reply[0] == 'y':
        return True
    elif not len(reply) or reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... " + question)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
