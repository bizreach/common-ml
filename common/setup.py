import os
import sys

from setuptools import setup, find_packages

# Don't import commonml module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'commonml'))
from version import VERSION

# Environment-specific dependencies.
extras = {
    'elasticsearch': ['elasticsearch>=2.0.0'],
    'runner': ['pyyaml'],
    'chainer': ['chainer'],
    'image': ['pillow'],
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name="common-ml",
    version=VERSION,
    packages=[package for package in find_packages()
              if package.startswith('commonml')],
    author="BizReach AI Team",
    author_email="shinsuke.sugaya@bizreach.co.jp",
    license="Apache Software License",
    description=("Common Machine Learning Library"),
    keywords="machine learning",
    url="https://github.com/bizreach/common-ml",
    download_url='https://github.com/bizreach/common-ml/tarball/' + VERSION,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=['six', 'numpy', 'scipy', 'scikit-learn>=0.17'],
    extras_require=extras,
)
