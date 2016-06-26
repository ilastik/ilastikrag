#!/usr/bin/env python

import os
import subprocess
from setuptools import setup, find_packages

def determine_version(default_version_file):
    # If running in conda-build, read version from the environment.
    if 'PKG_VERSION' in os.environ:
        auto_version = os.environ['PKG_VERSION']
    else:
        # Try getting it from the latest git tag
        try:
            commit_details = subprocess.check_output("git describe --tags --long HEAD".split())
            last_tag, commits_since_tag = commit_details.split('-')[:2]
            auto_version = last_tag
            if int(commits_since_tag) > 0:
                # Append commit count to version
                # (BTW, the .post## convention is recognized by PEP440)
                # For example: '0.2.post5'
                auto_version = last_tag + '.post' + commits_since_tag
        except subprocess.CalledProcessError:
            # Weird: We're not in a git repo or conda-bld/work.
            # Read the default value from the source code, I guess.
            version_globals = {}
            execfile(default_version_file, version_globals)
            auto_version = version_globals['__version__']
    return auto_version

VERSION_FILE = 'ilastikrag/version.py'
auto_version = determine_version(VERSION_FILE)

# Cache version.py, then overwrite it with the auto-version
with open(VERSION_FILE, 'r') as f:
    orig_version_contents = f.read()

with open(VERSION_FILE, 'w') as f:
    f.write("__version__ = '{}'\n".format(auto_version))

try:
    setup(name='ilastikrag',
          version=auto_version,
          description='ND Region Adjacency Graph with edge feature algorithms',
          author='Stuart Berg',
          author_email='bergs@janelia.hhmi.org',
          url='github.com/stuarteberg/ilastikrag',
          packages=find_packages()
          ## see conda-recipe/meta.yaml for dependency information
          ##install_requires=['numpy', 'h5py', 'pandas', 'vigra', 'networkx']
         )
finally:
    # Restore the version.py source as it was
    with open(VERSION_FILE, 'w') as f:
        f.write(orig_version_contents)
