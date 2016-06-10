#!/usr/bin/env python

from setuptools import setup

setup(
    name="common-ml",
    version="0.0.4",
    packages=['commonml',
              'commonml.sklearn',
              'commonml.elasticsearch',
              'commonml.text',
              'commonml.utils'],
    author="Shinsuke Sugaya",
    author_email="shinsuke.sugaya@bizreach.co.jp",
    license="Apache Software License",
    description=("Common Machine Learning Library"),
    keywords="machine learning",
    url="https://github.com/bizreach/common-ml",
    download_url='https://github.com/bizreach/common-ml/tarball/0.0.4',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
