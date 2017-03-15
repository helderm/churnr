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
#yesterday = (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y%m%d')
yesterday = '20170313'

@click.command()
@click.option('--project', default='user-lifecycle')
@click.option('--dataset', default='helder')
@click.option('--enddate', default=yesterday)
@click.option('--numdays', default=7)
@click.option('--shareusers', default=0.01)
@click.option('--timesplits', default=3)
def main(project, dataset, enddate, numdays, shareusers, timesplits):
    """ Extract data from several sources on BigQuery using intermediary tables
        and dump the final output as a file into ../data/raw
    """
    logger = logging.getLogger(__name__)
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
    jobs = []
    user_table = ds.table(name='user_ids_sampled_{date}'.format(date=enddate))
    if user_table.exists():
        user_table.delete()

    currdate = dt.datetime.strptime(enddate, '%Y%m%d')
    for i in range(7):
        datestr = currdate.strftime('%Y%m%d')

        query = """\
           SELECT user_id FROM [user-attributes-bq:user_sessions_aggregated.user_sessions_aggregated_{date}]\
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

    logger.info('Waiting for jobs to finish...')
    wait_for_jobs(jobs)

    # dispatch jobs for each day
    jobs = []
    currdate = dt.datetime.strptime(enddate+'000000', '%Y%m%d%H%M%S')
    for i in range(numdays):

        # calculate the time splits
        currtimeslot_start = currdate.timestamp()
        datestr = currdate.strftime('%Y%m%d')

        logger.info('Fetching features for day {}'.format(datestr))

        # dump the features table
        sl_table = ds.table(name='user_sessions_length_s{sampledate}_{currdate}'.format(sampledate=enddate, currdate=datestr))
        if sl_table.exists():
            sl_table.delete()

        for j in range(timesplits):
            if j+1 != timesplits:
                # calculate the timeslot range
                currtimeslot_end = currtimeslot_start + math.ceil(SECS_IN_DAY / timesplits) + 0.999
            else:
                # get the remaining secs lost due to rounding on the last slot. Will it be biased?
                currtimeslot_end = currdate.timestamp() + SECS_IN_DAY + 0.999

            # query avg session length
            query = """\
                SELECT  user_id, avg(session_length) as avg_session_length, {startslot} as slot_start, {endslot} as slot_end FROM\
                        (SELECT user_id , (sessions.end_time - sessions.start_time) as session_length\
                            FROM [user-attributes-bq:user_sessions_aggregated.user_sessions_aggregated_{date}]\
                            WHERE user_id IN (SELECT user_id FROM [{project}:{dataset}.{table}])\
                                AND sessions.start_time >= {startslot} AND sessions.start_time < {endslot})
                GROUP BY user_id\
            """.format(date=datestr, project=project, dataset=dataset, table=user_table.name,
                        startslot=int(currtimeslot_start*1000), endslot=int(currtimeslot_end*1000))

            jobname = 'user_sessions_length_job_' + str(uuid.uuid4())
            job = client.run_async_query(jobname, query)
            job.destination = sl_table
            job.allow_large_results = True
            job.write_disposition = 'WRITE_APPEND'
            job.begin()

            jobs.append(job)

            currtimeslot_start += math.ceil(SECS_IN_DAY / timesplits)

        currdate = currdate - dt.timedelta(days=1)

    logger.info('Waiting for jobs to finish...')
    wait_for_jobs(jobs)


def wait_for_jobs(jobs):
    import time
    for job in jobs:
        numtries = 20
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
