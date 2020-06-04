r"""
This module provides functions that define :math:`\theta` coefficients,
corresponding values of *p*-order / *k*-step, and other parameters associated
with different *Explicit*, *Implicit* and *Implicit + 1* methods.

* **Explicit methods of order** :math:`p = k`:

    * Families of methods:

        * Adams-Bashforth: :math:`p \in [1, \infty]`
        * EDF: :math:`p \in [1, \infty ]`
        * Nystrom: :math:`p \in [2, 5]`
        * EDC1: :math:`p \in [3, 6]` containing EDC22, EDC23, EDC24, EDC45
        * EDC2: :math:`p \in [3, 6]` containing EDC22, EDC23, EDC34, EDC45
        * EDC3: :math:`p \in [3, 6]` containing EDC22, EDC33, EDC24, EDC45
        * EDC4: :math:`p \in [3, 6]` containing EDC22, EDC33, EDC34, EDC45

    * Individual / single-order methods:

        * EDC22: *p* = 3
        * EDC23: *p* = 4
        * EDC33: *p* = 4
        * EDC24: *p* = 5
        * EDC34: *p* = 5
        * EDC45: *p* = 6

* **Implicit methods of order** :math:`p = k`:

    * Families of methods:

        * Backward Differentiation Formula (BDF) :math:`p \in [1, 5]`
        * Rockswold: :math:`p \in [2, 4]`

    * Individual / single-order methods:

        * Kregel: *p* = 3

* **Implicit methods of order** :math:`p = k+1`:

    * Families of methods:

        * Adams-Moulton: :math:`p \in [3, \infty]`
        * dcBDF: :math:`p \in [2, \infty]`

        .. note:: **(Version 0.9.0)** dcBDF :math:`\theta` values require
           checking as the original MATALB code has conflicting statements in
           this area.

        * IDC1: :math:`p \in [4, 7]` containing IDC23, IDC24, IDC45, IDC56
        * IDC2: :math:`p \in [4, 7]` containing IDC23, IDC34, IDC45, IDC56

    * Individual / single-order methods:

        * Milne2 *p* = 4

        .. note:: **(Version 0.9.0)** Milne2 *p* = 4 source code is not
           implemented / quarantined.  Refer to source code for explanation.

        * Milne4: *p* = 5

        .. note:: **(Version 0.9.0)** Milne4 is implemented but does not
           appear to be stable or work with the current algorithm.

        * IDC23: *p* = 4
        * IDC24: *p* = 5
        * IDC34: *p* = 5
        * IDC45: *p* = 6
        * IDC56: *p* = 7
"""

# Last updated: 4 June 2020 by Eric J. Whitney

from abc import ABC
from enum import Enum
import numpy as np
from types import MethodType
from typing import Dict, Sequence

# noinspection PyArgumentList
SolverType = Enum('SolverType', ('EXPLICIT', 'IMPLICIT', 'IMPLICIT_PLUS'))
"""Sets uniform constant labels for different solution methods."""

PI_2 = np.pi / 2


# ----------------------------------------------------------------------------

class Multistep(ABC):
    """This is the abstract base class for all multi-step ODE methods.

    Attributes
    ----------
    p_defaults : (int, int)
        Recommended *p* range (low, high).

    p_limits : (int, int) or (int, None)
        Absoluste *p* limits (low, high) or alternatively (low, None) if
        there  is noupper limit.

.. note:: `p_default` and `p_limit` values must separately provided for each
   instance if `p_theta` is not supplied during creation.
    """

    def __init__(self, solver_type: SolverType,
                 p_theta: Dict[int, Sequence[float]] = None):
        """
        Parameters
        ----------
        p_theta : Dict[int, array-like], optional
            Dict containing each order *p* as a key and a corresponding
            sequence of theta values that define the method.
        """
        self.solver_type = solver_type
        if p_theta is not None:
            self._p_theta = {k: np.asarray(v) for k, v in p_theta.items()}
            self.p_defaults = min(p_theta.keys()), max(p_theta.keys())
            self.p_limits = self.p_defaults

    def k(self, p: int) -> int:
        """Returns the number of steps `k` used by this method to generate a
        new point based the given order `p`. """
        if self.solver_type == SolverType.IMPLICIT_PLUS:
            return p - 1
        else:
            return p

    def theta(self, p: int) -> np.ndarray:
        r"""Returns an array of :math:`\theta` values corresponding to the
        given order `p`. """
        if p < self.p_limits[0] or p > (self.p_limits[1] or np.Inf):
            raise ValueError(f"p = {p} invalid.")
        return self._p_theta[p]


# Explicit - Specific Implementations ----------------------------------------

adams_bashforth = Multistep(SolverType.EXPLICIT, None)
adams_bashforth.theta = MethodType(lambda self, p: np.ones(p - 1) * PI_2,
                                   adams_bashforth)
adams_bashforth.p_limits = (1, None)
adams_bashforth.p_defaults = (1, 8)

# Theta arrays for EDC methods used in multiple places.
_edc22 = [np.arctan(14 / 3), PI_2]
_edc23 = [np.arctan(49 / 6), PI_2, PI_2]
_edc33 = [np.arctan(7 / 2), np.arctan(39 / 4), PI_2]
_edc24 = [np.arctan(1121 / 90), PI_2, PI_2, PI_2]
_edc34 = [np.arctan(53 / 10), np.arctan(219 / 10), PI_2, PI_2]
_edc45 = [np.arctan(193 / 45), np.arctan(121 / 10), np.arctan(692 / 15),
          PI_2, PI_2]

edf = Multistep(SolverType.EXPLICIT, None)
edf.theta = MethodType(lambda self, p: np.arctan(range(2, p + 1)), edf)
edf.p_limits = (1, None)
edf.p_defaults = (1, 8)

# Implicit - Specific Implementations ----------------------------------------

bdf = Multistep(SolverType.IMPLICIT, None)
bdf.theta = MethodType(lambda self, p: np.zeros(p), bdf)
bdf.p_limits = (1, 6)
bdf.p_defaults = (1, 5)

# Implicit + 1 - Specific Implementations ------------------------------------

adams_moulton = Multistep(SolverType.IMPLICIT_PLUS, None)
adams_moulton.theta = MethodType(lambda self, p: np.ones(p - 2) * PI_2,
                                 adams_moulton)
adams_moulton.p_limits = (2, None)
adams_moulton.p_defaults = (2, 8)

dc_bdf = Multistep(SolverType.IMPLICIT_PLUS, None)
dc_bdf.theta = MethodType(lambda self, p: np.ones(p - 2) * PI_2, dc_bdf)
dc_bdf.p_limits = (2, None)
dc_bdf.p_defaults = (2, 8)

# Theta arrays for IDC methods used in multiple places.
_idc23 = [np.arctan(7 / 6), PI_2]
_idc24 = [np.arctan(26 / 15), PI_2, PI_2]
_idc34 = [np.arctan(4 / 5), np.arctan(33 / 20), PI_2]
_idc45 = [np.arctan(28 / 45), np.arctan(11 / 10), np.arctan(32 / 15), PI_2]
_idc56 = [np.arctan(43 / 84), np.arctan(6 / 7), np.arctan(29 / 21),
          np.arctan(55 / 21), PI_2]

# ----------------------------------------------------------------------------
# Collection of all multistep methods with associated order/s (p) and
# theta vectors.

METHODS = {
    # Explicit Methods -------------------------------------------------------
    'Adams-Bashforth': adams_bashforth,

    'EDF': edf,

    'Nystrom': Multistep(SolverType.EXPLICIT, {
        2: [0.0],
        3: [np.arctan(-2 / 3), PI_2],
        4: [np.arctan(-5 / 3), PI_2, PI_2],
        5: [np.arctan(-133 / 45), PI_2, PI_2, PI_2]}),

    # Single order EDC methods.
    'EDC22': Multistep(SolverType.EXPLICIT, {3: _edc22}),
    'EDC23': Multistep(SolverType.EXPLICIT, {4: _edc23}),
    'EDC33': Multistep(SolverType.EXPLICIT, {4: _edc33}),
    'EDC24': Multistep(SolverType.EXPLICIT, {5: _edc24}),
    'EDC34': Multistep(SolverType.EXPLICIT, {5: _edc34}),
    'EDC45': Multistep(SolverType.EXPLICIT, {6: _edc45}),

    # EDC Families.
    'EDC1': Multistep(SolverType.EXPLICIT, {
        3: _edc22, 4: _edc23, 5: _edc24, 6: _edc45}),

    'EDC2': Multistep(SolverType.EXPLICIT, {
        3: _edc22, 4: _edc23, 5: _edc34, 6: _edc45}),

    'EDC3': Multistep(SolverType.EXPLICIT, {
        3: _edc22, 4: _edc33, 5: _edc24, 6: _edc45}),

    'EDC4': Multistep(SolverType.EXPLICIT, {
        3: _edc22, 4: _edc33, 5: _edc34, 6: _edc45}),

    # Implicit Methods -------------------------------------------------------
    'BDF': bdf,

    'Kregel': Multistep(SolverType.IMPLICIT, {
        3: [np.arctan(154/543), np.arctan(-11/78), 0]}),

    'Rockswold': Multistep(SolverType.IMPLICIT, {
        2: [np.arctan(101 / 150), np.arctan(37 / 50)],
        3: [np.arctan(73 / 350), np.arctan(71 / 200), PI_2],
        4: [np.arctan(29 / 160), np.arctan(161 / 360), PI_2, PI_2]}),

    # Implicit + 1 Methods ---------------------------------------------------
    'Adams-Moulton': adams_moulton,

    'dcBDF': dc_bdf,

    # ------------------------------------------------------------------------
    # Note: Milne2 source code below is quarantined. The theta vector
    # appears to be a special case and seems incomplete using the approach
    # shown in the original MATLAB implementation, where it fails to
    # generate the corrector due to size mismatch in the returned values,
    # E.G. refer ``modes.m``:
    #       polCoeffM(pmax+1-(ki+1):pmax+1,:,i,2) = polIp(...)
    #
    # In the Python version it fails to generate the predictor polynomial,
    # for essentially the same reason.
    #
    # ...
    #
    # 'Milne2': Multistep(SolverType.IMPLICIT_PLUS, {
    #     4: [np.arctan(1 / 3)]}),
    # ------------------------------------------------------------------------

    'Milne4': Multistep(SolverType.IMPLICIT_PLUS, {
        5: [np.arctan(4 / 15), PI_2, PI_2]}),

    # Single order IDC methods.
    'IDC23': Multistep(SolverType.IMPLICIT_PLUS, {4: _idc23}),
    'IDC24': Multistep(SolverType.IMPLICIT_PLUS, {5: _idc24}),
    'IDC34': Multistep(SolverType.IMPLICIT_PLUS, {5: _idc34}),
    'IDC45': Multistep(SolverType.IMPLICIT_PLUS, {6: _idc45}),
    'IDC56': Multistep(SolverType.IMPLICIT_PLUS, {7: _idc56}),

    # IDC families.
    'IDC1': Multistep(SolverType.IMPLICIT_PLUS, {
        4: _idc23, 5: _idc24, 6: _idc45, 7: _idc56}),

    'IDC2': Multistep(SolverType.IMPLICIT_PLUS, {
        4: _idc23, 5: _idc34, 6: _idc45, 7: _idc56})

}
