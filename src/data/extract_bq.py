# -*- coding: utf-8 -*-
import os
import click
import logging
import uuid
from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery as bq
import datetime as dt
import math

SECS_IN_DAY = 86399
yesterday = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
#yesterday = '20170313'

logger = logging.getLogger(__name__)

@click.command()
@click.option('--project', default='user-lifecycle')
@click.option('--dataset', default='helder')
@click.option('--enddate', default=yesterday)
@click.option('--numdays', default=7)
@click.option('--shareusers', default=0.0001)
@click.option('--timesplits', default=3)
@click.option('--output', default='gs://helder/data/raw')
def main(project, dataset, enddate, numdays, shareusers, timesplits, output):
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
    ft_tables, jobs = fetch_features(user_table, numdays, enddate, timesplits, project, ds, client)
    wait_for_jobs(jobs)

    # backfill missing users on feature tables
    jobs = backfill_missing_users(numdays, enddate, timesplits, project, ds, client)
    wait_for_jobs(jobs)

    # serialize features
    jobs = dump_features_to_gcs(ft_tables, output, client)
    wait_for_jobs(jobs)

    logger.info('Finished! Data dumped to {}'.format(output))


def fetch_user_samples(enddate, shareusers, ds, client):

    user_table = ds.table(name='user_ids_sampled_{date}'.format(date=enddate))
    if user_table.exists():
        user_table.delete()

    currdate = dt.datetime.strptime(enddate, '%Y%m%d')
    jobs = []
    for i in range(7):
        datestr = currdate.strftime('%Y%m%d')

        query = """\
           SELECT user_id FROM [activation-insights:sessions.features_{date}]\
               WHERE ABS(HASH(user_id )) % {share} == 0
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

    query = """\
       SELECT user_id FROM [{project}:{dataset}.{table}] GROUP BY user_id\
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

            # query avg session length
            query = """\
                SELECT  user_id, avg(session_length) as session_length, avg(skip_ratio) as skip_ratio, avg(unique_pages) as unique_pages FROM\
                        (SELECT user_id , session_length, skip_ratio, unique_pages\
                            FROM [activation-insights:sessions.features_{date}]\
                            WHERE user_id IN (SELECT user_id FROM [{project}:{dataset}.{table}])\
                                AND TIMESTAMP_TO_MSEC(TIMESTAMP(start_time)) >= {startslot} AND TIMESTAMP_TO_MSEC(TIMESTAMP(start_time)) < {endslot})
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


def backfill_missing_users(numdays, enddate, timesplits, project, ds, client):

    jobs = []
    currdate = dt.datetime.strptime(enddate+'000000', '%Y%m%d%H%M%S')

    ft_table_name = 'user_features_s{sampledate}_'.format(sampledate=enddate)
    for i in range(numdays):

        # calculate the time splits
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Backfilling features for day {}'.format(datestr))

        for j in range(timesplits):

            ft_table = ds.table(name=ft_table_name+'{split}_{currdate}'.format(currdate=datestr, split=j))

            # append missing values to feature table
            query = """\
                SELECT  user_id, 0.0 as session_length, 0.0 as skip_ratio, 0.0 as unique_pages\
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

        currdate = currdate - dt.timedelta(days=1)

    return jobs


def dump_features_to_gcs(ft_tables, dest, client):

    logger.info('Dumping {} tables to {}...'.format(len(ft_tables), dest))

    jobs = []
    for ft_table in ft_tables:
        path = dest + '/' + ft_table.name
        jobname = 'features_dump_job_' + str(uuid.uuid4())
        job = client.extract_table_to_storage(jobname, ft_table, path)
        job.begin()

        jobs.append(job)

    return jobs


def wait_for_jobs(jobs):

    logger.info("Waiting for {} jobs to finish...".format(len(jobs)))

    import time
    for job in jobs:
        numtries = 30
        job.reload()
        while job.state != 'DONE':
            time.sleep(3)
            job.reload()
            numtries -= 1
            assert numtries >= 0


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
