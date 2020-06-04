r"""
**MODESpy** implements generic methods for constructing and using
parameterised linear multi-step methods to solve Ordinary Differential
Equation Initial Value Problems (ODE IVPs). It is a Python implementation
based on the MODES toolbox (Version 1.0.0) written in MATLAB from Lund
University (see Authors below).

These methods are of variable step-size and order, and are regulated using
digital control theory. MODESpy allows experimentation with different
methods and regulation settings. Included is also a library of standard
problems, which can be used to evaluate methods and settings.

MODESpy depends on SciPy and NumPy which should be installed before use.

Version
-------
The current version is **0.9.0**.  **MODESpy** is designed for Python >= 3.7
and is platform agnostic.

Authors
-------
* Python version 0.9.0 written by Eric J. Whitney, 2020.
* Original MATLAB toolbox implementation by:
    - C. Arévalo - Center for Mathematical Sciences, Lund Institute of
      Technology, Lund, Sweden.
    - E Jonsson-Glans, J. Olander - HiQ Ace AB, Linköping, Sweden.
    - M. Selva-Soto - Department of Mathematical Engineering, University of
      Concepción, Chile.
    - G. Söderlind - Center for Mathematical Sciences, Lund University, Lund,
      Sweden.

The current version of the MATLAB toolbox is located at
https://github.com/mss1972/MODES_v1.0
"""

from modespy.filters import FILTERS, filter_order, filter_r
from modespy.methods import SolverType, Multistep, METHODS
from modespy.modes import MODES

import sys

assert sys.version_info >= (3, 7)
