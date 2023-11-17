from . import accumulators
from .accumulators import BaseEdgeAccumulator
from .accumulators import BaseSpAccumulator
from .rag import Rag

try:
    from ._version import version
except ModuleNotFoundError:
    raise RuntimeError(
        "Couldn't determine ilastik version - if you are developing ilastik please "
        "make sure to use an editable install (`pip install -e .`). "
        "Otherwise, please report this issue to the ilastik development team: team@ilastik.org"
    )

##################
# # Version info ##
##################

__version__ = version