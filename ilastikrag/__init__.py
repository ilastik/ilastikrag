from __future__ import absolute_import
from .version import __version__
from . import accumulators
from .accumulators import BaseEdgeAccumulator
from .accumulators import BaseSpAccumulator
from .rag import Rag

# Convenient for running tests even after install:
# nosetests ilastikrag.tests
from . import tests
