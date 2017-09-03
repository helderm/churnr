churnr
==============================

A churn prediction project for a music streaming service

### Requirements
* [Anaconda](https://www.anaconda.com/download/) Python 2.7
* [Google Cloud SDK](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version)

### Installing
```bash
$ make env
$ source activate churnr
$ make reqs
```

### Create Data
```bash
$ make data
```

### Dispatch Training Job
```bash
$ make submit
```

### Download Model Predictions
```bash
$ make download
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── README.md      <- Data description and source paths
    │
    ├── models             <- Model predictions
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── churnr             <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── app.py         <- Application entry point for the experiment dispatcher
    │   │
    │   ├── submitter.py   <- Script for submitting training jobs to CloudML
    │   │
    │   ├── sample.pys     <- Script for sampling user ids for the experimentss
    │   │
    │   ├── scala/parse.sh <- Dispatches a play context parser job on Dataflow on the sampled data in 'sample.py'
    │   │
    │   ├── extract.py     <- Engineer features and aggregate into timesteps the data parse at 'parse.sh'
    │   │
    │   └── process.py     <- Normalize data from 'extract.py' and export to files in GCS
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



