"""
tempnet



Logging
-------
The package sets up a default logger on import. You can adjust the level:

>>> import tempnet
>>> tempnet.set_log_level("DEBUG")


Author
------
Alexandre Bovet <alexandre.bovet@uzh.ch> 


Contributors
............

- Jonas I. Liechti <j-i-l@t4d.ch>
- Yasaman Asgari <yasaman.asgari@uzh.ch>
- Samuel Koovely <samuel.koovely@uzh.ch>

License
-------
GNU Lesser General Public License v3 or later (LGPLv3+).

"""
try:
    # try to import version (provided by hatch (see pyproject.toml)
    from ._version import __version__
except ImportError:
    # Fallback if the package wasn't installed properly
    __version__ = "unknown"

import logging

from .logger import setup_logger, get_logger

# Default log level
setup_logger()  # Set up the logger with the default level

from .temporal_network import (  # noqa: F401
    ContTempNetwork,
    ContTempInstNetwork,
)
from .faster_expm import (  # noqa: F401
    compute_subspace_expm,
    compute_subspace_expm_parallel,
    sparse_lapl_expm)

from .utils import (  # noqa: F401
    csc_row_normalize,
    inplace_csr_row_normalize,
    remove_nnz_rowcol,
    set_to_ones,
    set_to_zeroes,
)

from .synth_temp_network import (  # noqa: F401
    SynthTempNetwork,
    Individual,
    make_step_block_probs,
)

def set_log_level(level):
    """
    Set the logging level for the package.

    Parameters
    ----------
    level : str
        The logging level as a string (e.g., 'DEBUG', 'INFO').
    """
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    if level in level_dict:
        logger = get_logger()
        logger.setLevel(level_dict[level])
        for handler in logger.handlers:
            handler.setLevel(level_dict[level])
    else:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Choose from {list(level_dict.keys())}."
        )
