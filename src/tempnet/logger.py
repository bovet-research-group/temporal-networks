"""
#
# Temporal networks `tempnet`
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# Copyright (C) 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>, 
#                    Yasaman Asgari <yasaman.asgari@uzh.ch>, 
#                    Samuel Koovely <samuel.koovely@uzh.ch>, 
#                    Jonas Liechti <jonas@t4d.ch>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""

"""Logging utilities for :mod:`tempnet`."""

import os
import logging

# Create a logger instance
logger = logging.getLogger("flowstab")


class CustomPathnameFilter(logging.Filter):
    """Shorten log-record path names to the last two path components."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Shorten the pathname stored on a log record in place.

        Parameters
        ----------
        record : :class:`logging.LogRecord`
            Log record whose ``pathname`` attribute is shortened to the last
            two path components when possible.

        Returns
        -------
        bool
            Always ``True`` so the record remains eligible for emission.
        """
        full_path = record.pathname
        path_parts = full_path.split(os.sep)

        if len(path_parts) > 2:
            record.pathname = os.sep.join(path_parts[-2:])
        return True


def setup_logger(log_level: int = logging.INFO) -> None:
    """Set up the package logger.

    Parameters
    ----------
    log_level : int, default=logging.INFO
        Logging threshold passed to :meth:`logging.Logger.setLevel` and to the
        console :class:`logging.StreamHandler`, for example
        :data:`logging.DEBUG` or :data:`logging.INFO`.

    Returns
    -------
    None
        The global package logger is configured in place.
    """
    logger.setLevel(log_level)

    logger.addFilter(CustomPathnameFilter())

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d'
        '- PID:%(process)d - %(message)s')
    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger if it doesn't have handlers
    if not logger.hasHandlers():
        logger.addHandler(ch)


def get_logger() -> logging.Logger:
    """Return the package logger.

    Returns
    -------
    :class:`logging.Logger`
        Logger instance shared by the package modules.
    """
    return logger
