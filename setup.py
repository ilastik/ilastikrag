#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools_scm

_version = setuptools_scm.get_version(write_to="ilastikrag/_version.py")

setup(name='ilastikrag',
      version=_version,
      description='ND Region Adjacency Graph with edge feature algorithms',
      author='Stuart Berg',
      author_email='bergs@janelia.hhmi.org',
      url='github.com/stuarteberg/ilastikrag',
      packages=find_packages()
      ## see conda-recipe/meta.yaml for dependency information
      ##install_requires=['numpy', 'h5py', 'pandas', 'vigra', 'networkx']
)
