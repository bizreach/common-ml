#!/usr/bin/env python

from setuptools import setup

setup(
    name="common-ml",
    version="0.0.3",
    packages=['commonml',
              'commonml.sklearn',
              'commonml.elasticsearch',
              'commonml.text',
              'commonml.utils'],
    author="BizReach, Inc.",
    license="Apache Software License",
    description=("Common Machine Learning Library"),
    keywords="machine learning",
    url="https://github.com/bizreach/common-ml",
    download_url='https://github.com/bizreach/common-ml/tarball/0.0.3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
