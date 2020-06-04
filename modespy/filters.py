"""
This module establishes standard filter vectors and functions to
implement discrete PID-style stepsize controllers.  The following filters
are provided:

    - **First order filters:**
        - ``Elementary``: [1] (Also used as fallback filter if no step history
          exists)

    - **Second order filters:**
        - ``H211D``: [1, 1, -1] / 2
        - ``H211b``: [1, 1, -1] / (b) (* Default `b` = 4)
        - ``H211PI``: [1, 1, 0] / 6
        - ``PI3333``: [2, -1, 0] / 3
        - ``PI3040``: [7, -4, 0] / 10
        - ``PI4020``: [3, -1, 0] / 5

    - **Third order filters:**
        - ``H312D``: [1, 2, 1, -3, -1] / 4,
        - ``H312b``: [1, 2, 1, -3, -1] / (b) (* Default `b` = 8)
        - ``H312PID``: [1, 2, 1, 0, 0] / 18
        - ``H321D``: [5, 2 - 3, 1, 3] / 4
        - ``H321``: [6, 1, -5, 15, 3] / 18

    .. note:: Filters marked (*) must be passed as a tuple to MODES
       ``__init__`` with a given divisor `b` e.g. ('H312b', 8).  This allows
       for a non-default value of the `b` parameter in these cases.
"""

# Last updated: 31 May 2020 by Eric J. Whitney

import numpy as np
from typing import Tuple, Optional, Union

FiltArg = Optional[Union[str, Tuple[str, float]]]  # Type alias.

# Filters marked (*) must be passed as tuple to MODES with a divisor `b`.
FILTERS = {
    # First order filters.
    'Elementary': np.array([1]),  # Fallback filter if no step history.

    # Second order filters.
    'H211D': np.array([1, 1, -1]) / 2,
    'H211b': np.array([1, 1, -1]),  # (* Default b = 4)
    'H211PI': np.array([1, 1, 0]) / 6,
    'PI3333': np.array([2, -1, 0]) / 3,
    'PI3040': np.array([7, -4, 0]) / 10,
    'PI4020': np.array([3, -1, 0]) / 5,

    # Third order filters.
    'H312D': np.array([1, 2, 1, -3, -1]) / 4,
    'H312b': np.array([1, 2, 1, -3, -1]),  # (* Default b = 8)
    'H312PID': np.array([1, 2, 1, 0, 0]) / 18,
    'H321D': np.array([5, 2 - 3, 1, 3]) / 4,
    'H321': np.array([6, 1, -5, 15, 3]) / 18,

    # None is included to allow the option of declaring 'no control'.
    None: None
}


def filter_order(vec: np.ndarray) -> int:
    """Function returning the order of the given filter vector `vec`."""
    return len(vec) // 2 + 1


def filter_r(vec: np.ndarray, p: int, hs: np.array, errs: np.array,
             unit_errors: bool) -> float:
    """
    Compute the step size ratio based on filtered interpretation of previous
    step sizes and associated controller errors.

    Parameters
    ----------
    vec : array_like
        Filter vector.
    p : int
        Order of method being filtered, used when scaling errors.
    hs : array_like
        Recent step sizes with ``len(hs)`` >= filter order.
    errs : array_like
        Recent controller errors corresponding to `hs`.
    unit_errors : bool
        True if scaling errors to unit step sizes is required.

    Returns
    -------
    r : float
        Step size ratio obtained via filtering.
    """
    steps = filter_order(vec)
    if len(errs) < steps or len(hs) < steps:
        raise ValueError(f"Insufficient error / step history for filter.")

    # Make reversed, trimmed copy of inputs.
    err_vec = np.array([errs[-1 - i] for i in range(0, steps)])
    h_vec = np.array([hs[-1 - i] for i in range(0, steps)])

    # Scaled error is (1 / error) ** (1 / p), adding 1e-16 to avoid div. zero.
    p_adj = p + 1 - (1 if unit_errors else 0)
    scl_errs = (1 / (err_vec + 1e-16)) ** (1 / p_adj)
    step_ratios = h_vec[0:-1] / h_vec[1:]
    return float(np.prod(np.concatenate([scl_errs, step_ratios]) ** vec))


def _parse_filter_arg(filt_arg):
    """Helper function for resolving different filter arguments passed to
    ``MODES``."""
    if isinstance(filt_arg, Tuple):
        return FILTERS[filt_arg[0]] / filt_arg[1]
    else:
        return FILTERS[filt_arg]
