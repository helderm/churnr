#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import churnr as pyt

def get_requirements(file_name='requirements.txt'):
    try:
        filename = open(file_name)
        lines = [i.strip() for i in filename.readlines() if len(i.rstrip()) > 0 and 'libs' not in i]
        filename.close()
    except:
        return []

    return lines


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return ''

reqs = get_requirements()

setup(
    name=pyt.__name__,
    version=pyt.__version__,
    description=pyt.__description__,
    long_description=read('README.rst') + '\n\n' + read('HISTORY.rst'),
    author=pyt.__author__,
    author_email=pyt.__email__,
    url=pyt.__url__,
    packages=find_packages(),
    platforms=['any'],
    include_package_data=True,
    install_requires=reqs,
    license="BSD",
    zip_safe=False,
    keywords=pyt.__name__,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=get_requirements('requirements-dev.txt'),
    entry_points={
        'console_scripts': [
            pyt.__name__ + ' = ' + pyt.__name__ + '.app:main'
        ]
    },

)
