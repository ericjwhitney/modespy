r"""
This module implements the `MODES` class which is drop-in compatible with
SciPy function ``scipy.integrate.solve_ivp()``.  This is the main mechanism by
which the parameterised multi-step solver is called.  See examples below
demonstrating the calling methods.

**Differences to MATLAB Implementation**

* **(Version 0.9.0)** Dense output added by Eric J. Whitney.
* **(Version 0.9.0)** Fallback implicit solver added by Eric J. Whitney.
* **(Version 0.9.0)** The requirement for counting a certain number of
  previous steps (``orderControlStarts``) before allowing order control has
  been removed from the main loop.  The purpose of this was to ensure that a
  higher order would have enough previous steps to be able to compute a new
  point.  In this implementation this requirement is met by design and is
  therefore checked on the fly.
* **(Version 0.9.0)** There is a subtle change in the how the polynomial
  values are unpacked for Implicit + 1 methods when *k* = 1.  Refer to
  ``_coeffs_implicit_plus()`` for details.
* **(Version 0.9.0)** The code implementating the original MATLAB method of
  setting up the predictor polynomial is quarantined, as it appears to be
  already covered by other code.  Refer to ``predict()`` for details.

Examples
--------

As a simple first example we solve the Lotka-Volterra problem and plot the
solution.  This is a non-stiff problem originating from biology.  We will
use the Adams-Bashforth method which does not required the Jacobian.

>>> from scipy.integrate import solve_ivp
>>> import matplotlib.pyplot as plt
>>> from modespy import MODES
>>> from modespy.std_problems import lotka_rhs
>>> sol = solve_ivp(fun=lotka_rhs, t_span=[0, 2], y0=[1, 1], method=MODES,
...                 modes_method='Adams-Bashforth')
>>> plt.figure()
>>> plt.plot(sol.t, sol.y[0,:], '-or', sol.t, sol.y[1,:], '-ob',
... markerfacecolor='none')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$y_0$, $y_1$')
>>> plt.title('LOTKA-VOLTERRA')
>>> plt.grid()

This next example shows how the solution can be started with a known
analytic solution, which can then also be used to compute real errors. For
interest we will also plot the dynamic solver order used during the solution.

>>> from math import exp, sqrt, cos, sin
>>> w_n, z = 3.0, 0.15  # Underdamped.
>>> x_0, u_0 = 1, 0
>>> w_d = w_n * sqrt(1 - z ** 2)

>>> def pendulum_eqns(t, y):
...     x = y[1]  # Free pendulum split into two 1st order ODEs.
...     x_dot = -2 * z * w_n * y[1] - w_n ** 2 * y[0]
...     return x, x_dot

>>> def pendulum_solution(t):  # Analytic solution.
...     x = exp(-z * w_n * t) * (
...             x_0 * cos(w_d * t) + (u_0 + z * w_n * x_0) / w_d * sin(w_d *
...                                                                    t))
...     x_dot = (exp(-t * w_n * z) *
...              (cos(t * w_d) * (u_0 + w_n * x_0 * z) - w_d * x_0 *
...               sin(t * w_d)) - w_n * z * exp(-t * w_n * z) *
...               (sin(t * w_d) * (u_0 + w_n * x_0 * z) / w_d + x_0 *
...                cos(t * w_d)))
...     return x, x_dot

>>> modes_stats = {}
>>> sol = solve_ivp(fun=pendulum_eqns, t_span=[0, 10], y0=[x_0, u_0],
...                 method=MODES,
...                 modes_stats=modes_stats,  # Gather statistics.
...                 modes_p=(3, 7),  # Force higher order for analytic startup.
...                 modes_config={'analytic_fn': pendulum_solution,
...                               'analytic_start': True,  # Known statup pts.
...                               'verbose': True})
>>> x_analytic = [y[0] for y in modes_stats['y_analytic']]  # Extract just `x`.

>>> plt.figure()
>>> plt.plot(sol.t, sol.y[0, :], '-ob', markerfacecolor='none',
...          label='COMPUTED')
>>> plt.plot(modes_stats['t'], x_analytic, '--sr', markerfacecolor='none',
...          label='ANALYTIC')
>>> plt.legend(loc='upper right')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$y$')
>>> plt.title('PENDULUM')
>>> plt.grid()

>>> plt.figure()
>>> plt.plot(modes_stats['t'], modes_stats['err_norm_analytic'], '-ob',
...          markerfacecolor='none')
>>> plt.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$||y - y_{EXACT}||$')
>>> plt.title('ERROR 2-NORM')
>>> plt.grid()

>>> plt.figure()
>>> plt.plot(modes_stats['t'], modes_stats['p'], 'k.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$p$')
>>> plt.yticks(np.arange(1, max(modes_stats['p']) + 1, 1))
>>> plt.title('SOLVER ORDER')
>>> plt.grid()

As a final more complex example we solve the **H**\ igh **I**\ rradiance
**RES**\ ponse which is a stiff problem originating from plant physiology.
A number of the *tunable* parameters are adjusted to show how this can be done.

>>> from modespy.std_problems import hires_jac, hires_rhs
>>> modes_stats = {}
>>> sol = solve_ivp(fun=hires_rhs, # doctest:+ELLIPSIS
...                 t_span=[0, 321.8122],
...                 y0=[1, 0, 0, 0, 0, 0, 0, 0.0057],
...                 t_eval=np.linspace(0.0, 321.8122, 200),  # Dense `t` vals.
...                 tol=1e-3,  # Nominated tolerance.
...                 method=MODES,
...                 jac=hires_jac,  # Jacobian function supplied.
...                 modes_method='dcBDF',
...                 modes_p=(2, 6),  # Specified order `p` range.
...                 modes_start_idx=2,  # Starting solver (p = 4 in this case).
...                 modes_filter=('H312b', 10),  # Non-default filter `b`.
...                 modes_stats=modes_stats,  # Gather stats on errors, `p`, etc.
...                 modes_config={'impl_solver': 'hybr',
...                               'impl_backup': None,
...                               'impl_its': 10,
...                               'verbose': True}
...                 )
MODES Solution -> p = [2, 3, 4, 5, 6], Method = dcBDF, Filter = ('H312b', 10)
... 18 RHS evaluations used during startup.
...
... 1207 RHS Evaluations: t = 2.4585, h = 0.010501, p = 4 (34 Rejected Steps)
... 1575 RHS Evaluations: t = 3.2594, h = 0.16758, p = 2 (34 Rejected Steps)
... 2561 RHS Evaluations: t = 117.98, h = 2.904, p = 2 (34 Rejected Steps)

>>> plt.figure()
>>> for i, y in enumerate(sol.y):
...     plt.plot(sol.t, y, '.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$y$')
>>> plt.title('HIRES')
>>> plt.margins(x=0)
>>> plt.grid()

Here we plot some statistics.  Note that when plotting statistics we use
``modes_stats['t']`` values (and not ``sol.t``).  These values align with the
data and represent actual computation which can be used to observe the
variable solver steps. They may not align with the normal output ``sol`` data
which may have a different length.

>>> plt.figure()
>>> h = [t2 - t1 for t1, t2 in zip(modes_stats['t'], modes_stats['t'][1:])]
>>> plt.plot(modes_stats['t'][:-1], h, 'k.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$h$')
>>> plt.title('STEP SIZE VARIATION')
>>> plt.yscale('log')
>>> plt.margins(x=0)
>>> plt.grid(which='both', axis='both')

>>> plt.figure()
>>> plt.plot(modes_stats['t'], modes_stats['err_norm'], 'k.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$||E||$')
>>> >>> plt.title('ERROR VARIATION')
>>> plt.yscale('log')
>>> plt.margins(x=0)
>>> plt.grid()

>>> plt.figure()
>>> plt.plot(modes_stats['t'], modes_stats['p'], 'k.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('$p$')
>>> plt.title('SOLVER ORDER')
>>> plt.yticks(np.arange(1, max(modes_stats['p']) + 1, 1))
>>> plt.margins(x=0)
>>> plt.grid()

>>> plt.figure()
>>> plt.plot(modes_stats['t'], modes_stats['rej_steps'], 'k.-')
>>> plt.xlabel('$t$')
>>> plt.ylabel('REJECTED STEPS')
>>> plt.title('ACCUMULATION OF REJECTED STEPS')
>>> plt.margins(x=0)
>>> plt.grid()
>>> plt.show()

Future
------

* Addition of a timer call within ``_step_impl()`` for each order and
  running averaging would allow work factors to be automatically computed.
"""

# Last updated: 4 June 2020 by Eric J. Whitney

from collections import deque
from dataclasses import dataclass
from functools import partial
import numpy as np
import scipy.integrate as sp_int
from scipy.linalg import pascal
from scipy.optimize import root
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from modespy.filters import (FILTERS, filter_r, filter_order, FiltArg,
                             _parse_filter_arg)
from modespy.methods import SolverType, Multistep, adams_bashforth, METHODS
from modespy.util import pad_coeffs, newton_const, polyval_cols

# Type aliases.
ScalarVector = Union[float, Sequence[float], Type[np.ndarray]]
RHSEqnSystem = Callable[[float, ScalarVector, Optional[Any]], ScalarVector]


# ----------------------------------------------------------------------------

@dataclass
class SolverInstance:
    """Container class used to hold all runtime data for a single instance
    of a specific ODE solver type operating on a single order."""
    p: int                           # Solver order.
    method: Multistep                # See methods.py.
    filter: np.ndarray               # See filters.py.
    q_len: int                       # Recent history storage length.
    h: deque                         # Recent step sizes.
    r_ctrl: deque                    # Recent step control ratios.
    poly_pred: Optional[np.ndarray]  # Current predictor polynomial.
    poly_corr: Optional[np.ndarray]  # Current corrector polynomial.
    running: bool                    # Status of this instance.

    @property
    def theta(self):  # Shorthand method.
        return self.method.theta(self.p)


# ----------------------------------------------------------------------------

class AnalyticDenseOutput(sp_int.DenseOutput):
    """AnalyticDenseOutput provides a DenseOutput representation of the
    user-supplied analytic function.  This is used when interpolating dense
    points in the very early startup phase of the solution where an analytic
    solution is available."""

    def __init__(self, t_old: float, t: float,
                 fun: Callable[[float], ScalarVector]):
        super().__init__(t_old, t)
        self.fun = fun

    def _call_impl(self, t: float):
        return np.asarray(self.fun(t))


# ----------------------------------------------------------------------------

class MODESDenseOutput(sp_int.DenseOutput):
    """MODESDenseOutput provides interpolation for intermediate values
    between MODES timesteps.  This is done by storing the corrector
    polynomial of the main running solver / order After each successful
    step, along with the solver type and the last time step (`t_old` → `t`)."""

    def __init__(self, t_old: float, t: float, sol_type: SolverType,
                 coeffs: np.ndarray):
        super().__init__(t_old, t)
        self.sol_type, self.coeffs = sol_type, coeffs.copy()

    def _call_impl(self, t: float):
        if self.sol_type == SolverType.IMPLICIT:
            h = t - self.t
        else:
            h = t - self.t_old

        return polyval_cols(self.coeffs, h)


# ----------------------------------------------------------------------------

class MODES(sp_int.OdeSolver):
    r"""This class implements the generic multi-step, multi-order ODE solver.
    It is derived from SciPy OdeSolver class so that it can be called
    directly by scipy.integrate ``solve_ivp()``.

    **Implementation Details**

    * The number of Jacobian LU-factorisations ``self.nlu`` are not tracked
      as is done for other ODE methods, because this depends on the choice of
      implicit solver which is variable.

    * Dense output uses the same interpolating polynomial developed for
      each time step as the solution progresses (i.e. the order and timestep
      will vary). Therefore the precision of dense output is the same as the
      solver precision at each point in the solution.

    .. note::  Because `solve_ivp()` calls `_step_impl()` on the solver
      before `_dense_output_impl()` is ever polled, there is no specific
      dense output interpolation available for any of the intermediate points
      computed during startup.  This is similar to other SciPy multistep
      methods, but could be adapted if desired.

    * Use of an analytic function for startup point generation is considered
      a research tool only.  At present it appears to be unreliable where
      higher starting orders are used i.e. :math:`p_{init} \leq 3`
    """

    def __init__(self, fun: RHSEqnSystem, t0: float, y0: ScalarVector,
                 t_bound: float, tol: float = 1e-4,
                 jac: Union[RHSEqnSystem, np.ndarray, None] = None,
                 vectorized=False,
                 first_step: Union[float, str] = 'Deterministic',
                 modes_p: Optional[Sequence[int]] = None,
                 modes_method: Union[str, Sequence[str]] = 'Adams-Moulton',
                 modes_filter: Union[FiltArg, Sequence[FiltArg]] = 'PI3333',
                 modes_start_idx: Optional[int] = None,
                 modes_config: Dict = None,
                 modes_stats: dict = None,
                 **extraneous):
        """
        Sets up the MODES solution and is called by ``solve_ivp()``.

        Parameters
        ----------
        fun : callable
            Right-hand side of the system supplied by ``solve_ivp()``.  The
            calling signature is ``fun(t, y)`` and it can be vectorized;
            refer to SciPy documentation for details.
        t0 : float
            Initial time.
        y0 : array_like, shape (n,)
            Initial state.  The size of this array sets the problem dimension.
        t_bound : float
            Boundary time - representing the end of the integration. The
            integration direction is determined from this value and `t0`.
        tol : float, optional
            Desired solution tolerance (default = 1e-4).  Implementation and
            scaling of errors is complex; refer to source code and MODES
            papers for more details.  See also `s_rel`, `s_abs` and `w_mat`
            in modes_config.
        jac : None, array_like or callable, optional
            Jacobian matrix of the right-hand side of the system with
            respect to `y`.  A Jacobian is only needed if an Implicit or
            Implicit + 1 solver is part of the run.  The Jacobian matrix
            has shape (n, n) and its element (i, j) is equal to
            :math:`df_i / dy_j`. There are three ways to define the Jacobian:

            * If array_like, the Jacobian is assumed to be constant.
            * If callable, the Jacobian is assumed to depend on both
              `t` and `y`; it will be called as ``jac(t, y)`` when required.
            * If None (default), the Jacobian will be approximated by
              finite differences when required.
        vectorized : bool, optional
            Whether `fun` is implemented in a vectorized fashion. Default is
            False.
        first_step : float, 'Deterministic' or 'Random', optional
            * If float, this sets the initial step size.
            * If 'Deterministic' (default) this sets the initial step size
              using a deterministic algorithm; refer to ``startup()`` for
              details.
            * If 'Random' (default) this sets the initial step size
              using a randomised algorithm; refer to ``startup()`` for
              details.
        modes_p : array_like, optional
            This specifies the orders `p` to be used by the solvers.  The type
            of argument supplied depends on the value of `modes_method`:

            * If `modes_method` is a string, all solvers and filters will
              share a common type and orders will span a prescribed range.
              In this case `modes_p` must have two elements representing
              this order range (inclusive), e.g. ``modes_p=(2, 6)``.  The
              default value equals ``p_default`` for the given method.
            * If `modes_method` is an array of strings, then each solver
              must have its own prescribed order.  In this case `modes_p`
              must have the same number of elements as `modes_method` and
              `modes_filter`, e.g. ``modes_p=(4, 5, 7, 8)``.
        modes_method : string or array_like[string], optional
            This specifies the kind of solver to use:

            * If `modes_method` is a string, all solvers and filters will have
              the same type (default = 'Adams-Moulton').  See methods.py for
              available types.
            * If `modes_method` is an array of strings, each solver type
              must be specified individually.
        modes_filter : FiltArg or array_like[FiltArg], optional
            This specifies the kind of filter to use:

            * If `modes_method` is a string, all solvers will use the same
              filter (default = `PI3333`).  Supply either a single string
              representing a default filter, or a tuple of ('filter name',
              b) if a bespoke factor is required.  See filters.py for
              available types.
            * If `modes_method` is an array of strings, a separate filter
              must be defined for each solver.  Supply a string or tuple for
              each as described above.
        modes_start_idx : int, optional
            Specify which solver to commence with by index (not by order)
            i.e. if ``modes_p=(2, 6)`` then solver index 0 sets `p` = 2,
            index 1 sets `p` = 3, and so on. The default is the first order
            (typically lowest).
        modes_config : dict[string, Any], optional
            Minor options for fine-tuning and research are passed by
            including them in this optional dict (unknown entries are
            ignored).  Available options are:

            'analytic_fn' : callable
                A function representing a known analytic solution to the
                problem, with a calling signature of ``y = analytic_fn(t)``
                (default = None).
            'analytic_start' : bool
                If True, use the analytic function to generate startup
                points instead of solving for them using a separate one-step
                method (default = False).
            'impl_its' : int
                Number of iterations for the primary implicit solver to
                acheive convergence (default = 12).
            'impl_solver' : string
                Primary solver to use for converging implicit values (default =
                'newton_const').  This can be 'newton_const' or any other
                valid argument accepted by ``scipy.optimize.root()``,
                e.g. ‘hybr’, ‘lm’, ‘broyden1’, etc.
            'impl_backup' : string
                Fallback solver to use if the primary implicit solver fails.
                This can be None, 'newton_const' or any other valid argument
                accepted by ``scipy.optimize.root()`` (as above).

                .. note:: The backup solver always numerically estimates its
                   own Jacobian.  This is done in case the user supplied
                   Jacobian calculation itself is unstable / poorly
                   conditioned.

            'r_min' : float
                Largest legal reduction in step size allowed (default = 0.8).
            'r_max' : float
                Largest legal enlargement in step size allowed (default = 1.2).
            's_rel' : float
                Factor to apply for relative errors (default = 0).
            's_abs' : float
                Factor to apply for absolute errors (default = 1).
            'startup_method' : string
                Type of solver to use when generating startup points
                (default = 'RK45').  Currently supported values are the
                Runge-Kutta solvers supported by
                ``scipy.integrate.solve_ivp()``: 'RK23', 'RK45' or 'DOP853'.
            'unit_errors' : bool
                Apply unit scaling to errors (default = False).
            'verbose' : bool
                Print output to show run progress (default = False).
            'w_mat' : ndarray, shape (ndim, ndim)
                Scaling matrix for errors.  The default is a unit matrix
                i.e. ``w_mat = np.eye(ndim, ndim)``.

        modes_stats: dict[str, array_like], optional
            An empty dictionary may be supplied which will be overwritten to
            allow detailed statistics to be returned.  It will be returned
            with these fields:  't', 'err_norm', 'h', 'p', 'rej_steps'.  If
            an analytic function is supplied it will also include
            'y_analytic' and 'err_norm_analytic'.

            .. note:: The number of points in these arrays may not equal the
               number of points returned by the main method, as they will
               also include startup points.
        """
        # noinspection PyProtectedMember
        sp_int._ivp.common.warn_extraneous(extraneous)

        # Setup specialised configuration options.
        if modes_config is None:
            modes_config = {}

        self.analytic_fn = modes_config.get('analytic_fn', None)
        self.analytic_start = modes_config.get('analytic_start', False)
        self.impl_its = modes_config.get('impl_its', 12)
        self.impl_solver = modes_config.get('impl_solver', 'newton_const')
        self.impl_backup = modes_config.get('impl_backup', 'hybr')
        self.r_min = modes_config.get('r_min', 0.8)
        self.r_max = modes_config.get('r_max', 1.2)
        self.s_rel = modes_config.get('s_rel', 0)
        self.s_abs = modes_config.get('s_abs', 1)
        self.startup_method = modes_config.get('startup_method', 'RK45')
        self.unit_errors = modes_config.get('unit_errors', False)
        self.verbose = modes_config.get('verbose', False)
        self.w_mat = modes_config.get('w_mat', None)

        if not (self.r_min <= 1 <= self.r_max):
            raise ValueError(f"Invalid step size range, r_min <= 1 <= r_max "
                             f"required.")

        # Assemble all orders, methods and filters to be used.
        _ps, _methods, _filters = [], [], []
        if isinstance(modes_method, str):
            # All solvers and filters will share a common type.
            if modes_p is None:
                modes_p = METHODS[modes_method].p_defaults

            for p_i in range(modes_p[0], modes_p[1] + 1):
                _ps.append(p_i)
                _methods.append(METHODS[modes_method])
                _filters.append(_parse_filter_arg(modes_filter))
            pstr = ', '.join(str(_p) for _p in _ps)
            self.vprint(f"MODES Solution -> p = [{pstr}], Method = "
                        f"{modes_method}, Filter = {modes_filter}")
        else:
            # Each order, solver and filter are individually specified.
            self.vprint(f"MODES Solution ->")
            for i, (p_i, m_i, f_i) in enumerate(zip(
                    modes_p, modes_method, modes_filter)):
                _ps.append(p_i)
                _methods.append(METHODS[m_i])
                _filters.append(_parse_filter_arg(f_i))
                self.vprint(f"... Solver {i}: p = {p_i}, Method = {m_i}, "
                            f"Filter = {f_i}")

        # Determine required length of recent history for each solver.
        _q_len = [max(m_i.k(p_i), filter_order(f_i) if f_i is not None else 0)
                  for p_i, m_i, f_i in zip(_ps, _methods, _filters)]

        # Setup solvers.
        self.sol = []  # List of all solvers to be used.
        for p_i, m_i, f_i, q_i in zip(_ps, _methods, _filters, _q_len):
            p_lo, p_hi = m_i.p_limits
            p_hi = np.Inf if p_hi is None else p_hi
            if not (p_lo <= p_i <= p_hi):
                raise ValueError(f"Specified order p = {p_i} out of limits "
                                 f"for solver (p = ({p_lo}, {p_hi})).")
            self.sol.append(SolverInstance(
                p=p_i, method=m_i, filter=f_i, q_len=q_i, h=deque(maxlen=q_i),
                r_ctrl=deque(maxlen=q_i), poly_pred=None, poly_corr=None,
                running=False))

        if self.nsols == 0:
            raise ValueError("At least one solver / order required.")

        if modes_start_idx is not None:
            self.sol_idx = modes_start_idx
        else:
            self.sol_idx = 0  # Start with the lowest order.

        if not (0 <= self.sol_idx < self.nsols):
            raise ValueError(f"Invalid starting solver index.")

        # Recent history of t, y, y' is stored in fixed length deques (FIFO)
        # with the last value corresponds to the 'current' value.  Required
        # length is whichever is longer of `k` or filter order. We also add
        # +1 to cover cases where a high order Implicit + 1 solution may
        # use the explicit equivalent to get predictor polynomials.
        long_q = max(_q_len) + 1
        self._t = deque([t0], maxlen=long_q)  # Reqd before __init__ as
        self._y = deque([y0], maxlen=long_q)  # `t`, `y` overridden.
        # super() is now called, to setup fun().
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self._ydot = deque([self.fun(t0, y0)], maxlen=long_q)

        if len(self._ydot[-1]) != self.n:
            raise ValueError(f"f(t, y) result vector ({len(self._ydot[-1])}) "
                             f"does not match problem dimension ({self.n}).")

        # Setup remaining step size parameters.
        self.tol, self.rejected_steps = tol, 0
        if self.tol <= 0.0:
            raise ValueError(f"Tolerance must be > 0.")

        if self.w_mat is None:
            self.w_mat = np.eye(self.n)

        # Setup Jacobian.
        jac_wrap = None  # Common closure for different Jacobian arguments.
        self.jac_factor = None
        if jac is None:
            # Generate Jacobian from finite differences (if required).
            if any(m_i.solver_type != SolverType.EXPLICIT for m_i in _methods):
                self.vprint(f"... Jacobian computed using finite differences.")

                # noinspection PyPep8Naming
                def jac_wrap(t, y):  # Implementation matches scipy/bdf.py.
                    self.njev += 1
                    f = self.fun_single(t, y)
                    # noinspection PyProtectedMember
                    J, self.jac_factor = sp_int._ivp.common.num_jac(
                        self.fun_vectorized, t, y, f, self.tol,
                        self.jac_factor, sparsity=None)
                    return J

        elif callable(jac):  # Jacobian is a user-supplied function.
            def jac_wrap(t, y):
                self.njev += 1
                return np.asarray(jac(t, y))

        else:  # Jacobian is a constant matrix.
            j_mat = np.asarray(jac)
            if j_mat.shape != (self.n, self.n):
                raise ValueError(f"`jac` is expected to have shape "
                                 f"{(self.n, self.n)}, but is actually "
                                 f"{j_mat.shape}")

            # noinspection PyUnusedLocal
            def jac_wrap(t, y):
                return j_mat

        self.jac = jac_wrap

        # Set starting step size and apply to all solvers.
        if first_step == 'Deterministic':
            first_step = self.auto_stepsize(random_perturb=False)
        elif first_step == 'Random':
            first_step = self.auto_stepsize(random_perturb=True)

        if (not isinstance(first_step, (float, int)) or
                np.abs(first_step) == 0.0):
            raise TypeError(f"Invalid or zero stepsize h0 = {first_step}")

        # All solvers start with same step size.
        for sol_i in self.sol:
            sol_i.h.append(np.abs(first_step) * self.direction)

        self.delta_p = 0  # Initialise accumulated change for order control.
        self.set_running_solvers()

        # Setup temporary holding and generate startup points.
        self._startup_t, self._startup_y, self._startup_ydot = [], [], []
        self._startup_dense = []
        self._last_dense = None
        self.startup()

        # Setup statistics recording (if required).
        if modes_stats is not None:
            self._run_stats = modes_stats
            for field in ('t', 'err_norm', 'h', 'p', 'rej_steps'):
                self._run_stats[field] = []
            if self.analytic_fn is not None:
                self._run_stats['y_analytic'] = []
                self._run_stats['err_norm_analytic'] = []
        else:
            self._run_stats = None

        self.vprint(f"... {self.nfev} RHS evaluations used during startup.")
        self._disp_next = 10  # Next `self.nfev` value to trigger print.

    # Properties -------------------------------------------------------------

    @property
    def h(self):
        """Returns the stepsize `h` associated with the current order."""
        return self.sol[self.sol_idx].h[-1]

    @property
    def nsols(self):
        """Returns the total number of separate solvers / orders used."""
        return len(self.sol)

    @property
    def p(self) -> int:
        """Returns the current order `p`."""
        return self.sol[self.sol_idx].p

    @property
    def t(self):  # Overrides OdeSolver.
        return self._t[-1]

    @t.setter
    def t(self, value):  # Overrides OdeSolver.
        self._t[-1] = value

    @property
    def y(self):  # Overrides OdeSolver.
        return self._y[-1]

    @y.setter
    def y(self, value):  # Overrides OdeSolver.
        self._y[-1] = value

    # Startup Methods  -------------------------------------------------------

    def auto_stepsize(self, random_perturb: bool = False,
                      rms_norm: bool = False) -> float:
        """
        An auto-scaling procedure for computing an initial stepsize for ODE
        IVP linear multistep solvers.  This is called from ``__init__`` if
        required during setup and the stepsize is based on the
        starting order, time direction and last `y` point, which must
        already be defined (these are typically just the single starting
        values).

        Parameters
        ----------
        random_perturb : bool, optional
            If False (default) the initial stepsize is computed using a
            deterministic algorithm.  Otherwise, a small random perturbation
            is used.

        rms_norm : bool, optional
            If True, take the ``sqrt()`` of the norm.

        Returns
        -------
        h : float
            Computed stepsize.
        """
        norm = np.linalg.norm  # Local alias.
        normscale = np.sqrt(self.n) if rms_norm else 1
        t0, y0, p0 = self._t[-1], self._y[-1], self.p  # Shorthand.
        dt = self.t_bound - t0
        t_mag, t_dir = abs(dt), np.sign(dt)

        # Perturb starting point.
        if random_perturb:
            dy0 = (np.random.rand(self.n) * norm(y0) / normscale / 100)
        else:
            # Get signs of initial values, assign +1 for zero initial values.
            signs_y0 = np.sign(y0)
            signs_y0 = np.ones(self.n) - np.abs(signs_y0) + signs_y0
            dy0 = signs_y0 * np.random.rand(self.n)
            dy0 = dy0 / norm(dy0) / normscale  # ||dy0|| wtih sel. signs.
            radius = 1e-6 * norm(y0) / normscale
            dy0 = radius * dy0  # ||dy0|| is scaled to 1e-6 of ||y0||.

        z0 = y0 + dy0
        y0prime = self.fun(t0, y0)
        z0prime = self.fun(t0, z0)
        lips = norm(y0prime - z0prime) / norm(dy0)  # Rough Lipschitz est.

        # Provisional step size used in estimation procedure has a
        # conservative hL ~ 0.05 (due to underetsimation of L), subject to
        # further limitations based on range of integration.
        h = t_dir * min(1e-3 * t_mag, max(1e-8 * t_mag, 0.05 / lips))

        # Estimation Step 1: Fwd Euler step.
        y1 = y0 + h * y0prime
        t1 = t0 + h

        # Estimation Step 2: Reverse Euler step back to _t0.
        y1prime = self.fun(t1, y1)
        y0comp = y1 - h * y1prime

        # Estimation Step 3: Estimate of local error, proportional to
        # h**2 * norm(u'').
        dy = y0comp - y0
        dynorm = norm(dy)  # For use in Lipschitz estimator only.

        # Estimation Step 4: New estimate of Lipschitz constant and log
        # Lipschitz constant.
        y0comprime = self.fun(t0, y0comp)
        l_fac = norm(y0comprime - y0prime) / dynorm
        m_fac = np.dot(dy, (y0comprime - y0prime)) / dynorm ** 2

        # Estimation Step 5: Norm of Fwd Euler error.
        errnorm = dynorm / normscale  # Norm of error.
        h_fe = h * np.sqrt(self.tol / errnorm)  # Correct Fwd Euler stepsize.

        # Estimation Step 6: Construct a refined starting step size.
        if random_perturb:
            tolscale = self.tol ** (1 / (p0 + 1))
            theta1 = tolscale / np.sqrt(errnorm)  # Account for accuracy.
            theta2 = tolscale / abs(h * (l_fac + m_fac / 2))  # Acct for stab.
            h = h * (theta1 + theta2) / 2  # Stab'd avg of both criteria.
        else:
            h = 1e-1 * t_dir * abs(h_fe) ** (2 / (p0 + 1))
            h = h * l_fac / (l_fac + m_fac / 2)  # Account for stability.

        return t_dir * min(3e-3 * t_mag, abs(h))  # Safety net.

    def startup(self):
        """
        Generate the required starting points for the multi-step method at
        of the required order.  This is done by adding new solution points
        until `k` points are present (acknowledging any stored points).
        This is done using either an explicit Runge-Kutta style solver or a
        known analytic solution  at `h` intervals.

        .. note:: This method is essentially a deferred section of
           ``__init___`` that sets up the data history and starting points.
           It is separated for the specific case where multiple solvers may
           be setup that share a common t, y, y' history.

           By the time we reach this point, there must be at least one t, y,
           y' value present in the history.
        """
        k0 = self.sol[self.sol_idx].method.k(self.p)
        h0 = self.sol[self.sol_idx].h[-1]
        if k0 < 2:
            return  # No additional steps required.

        if self.analytic_start and self.analytic_fn is not None:
            # Generate starting points using analytic function.
            self.vprint("... Analytic function used for startup points.")
            t_old = self.t
            while len(self._startup_t) + 1 < k0:
                t = t_old + h0
                y = np.asarray(self.analytic_fn(t))
                ydot = self.fun(t, y)

                self._startup_t.append(t)
                self._startup_y.append(y)
                self._startup_ydot.append(ydot)
                self._startup_dense.append(
                    AnalyticDenseOutput(t_old, t, self.analytic_fn))
                t_old = t
            return

        # All RungeKutta solvers use `self.f` for y' = f(t, y).
        # noinspection PyProtectedMember
        startup_methods: Dict[str, Type[sp_int._ivp.rk.RungeKutta]] = {
            'RK23': sp_int.RK23, 'RK45': sp_int.RK45, 'DOP853': sp_int.DOP853}
        if self.startup_method not in startup_methods:
            raise ValueError(f"Unknown startup method "
                             f"'{self.startup_method}'.")

        startup_type = startup_methods[self.startup_method]
        startup_solver = startup_type(self.fun, self.t, self.y, self.t_bound,
                                      vectorized=self.vectorized,
                                      first_step=h0, max_step=h0,
                                      rtol=self.tol, atol=self.tol * 1e-3)
        # Defaults for rtol / atol inherited from MATLAB implementation.
        while len(self._startup_t) + 1 < k0:
            message = startup_solver.step()
            if startup_solver.status == 'failed':
                raise RuntimeError(f"Startup solver failed: {message}")

            self._startup_t.append(startup_solver.t)
            self._startup_y.append(startup_solver.y)
            self._startup_ydot.append(startup_solver.f)
            self._startup_dense.append(startup_solver.dense_output())

        for li in (self._startup_t, self._startup_y, self._startup_ydot,
                   self._startup_dense):
            li.reverse()  # Make ready for pop() in correct order.

    # Running Methods --------------------------------------------------------

    def _step_impl(self):
        """
        Called automatically by ``solve_ivp()``, this performs one step of the
        ODE solution by doing the following for each order / solver currently
        running:

            * Generates a predictor and corrector point where possible.
            * Compute the error and reject / accept step.
            * Update the current running solvers (normally just different `p`
              values).

        .. note:: For the first few startup steps, the results are drawn
           from pre-computed startup points. This gives the solver the
           appearance of a one-step method to ``solve_ivp()`` which is
           designed with this in mind.
        """

        if len(self._startup_t) > 0:  # Draw from pre-computed points.
            self._t.append(self._startup_t.pop())
            self._y.append(self._startup_y.pop())
            self._ydot.append(self._startup_ydot.pop())
            self._last_dense = self._startup_dense.pop()
            self._record_stats()
            return True, None

        rejected_last = False
        while True:  # Loop until a step is accepted or we fail.
            t_new = self.t + self.h
            if t_new == self.t:  # Check for machine roundoff.
                return False, self.TOO_SMALL_STEP

            r_smooth: List[Optional[float]] = [None] * self.nsols
            y_pred: List[Optional[np.ndarray]] = [None] * self.nsols
            y_corr: List[Optional[np.ndarray]] = [None] * self.nsols
            for idx, sol_i in enumerate(self.sol):
                if not sol_i.running:
                    continue

                # Calculate prediction and correction for y(t) (where avail.)
                p = sol_i.p
                y_pred[idx] = self.predict(idx)
                y_corr[idx] = self.correct(idx)

                if (y_pred[idx] is None or y_corr[idx] is None or
                        sol_i.filter is None):
                    # Remark: I *think* this is the intent of original.
                    if len(sol_i.h) > 1:
                        r_smooth[idx] = sol_i.h[-1] / sol_i.h[-2]
                    else:
                        r_smooth[idx] = 1
                else:
                    # Do step size control. Estimate the 'unscaled' error and
                    # determine error per unit step by scaling error with the
                    # current step size (if required) and apply anti-windup.
                    new_err = y_pred[idx] - y_corr[idx]
                    if self.unit_errors:
                        new_err *= (1 / self.h)  # Step scaling.
                        new_err *= (sol_i.h[-1] / self.h) ** p
                    else:
                        new_err *= (sol_i.h[-1] / self.h) ** (p + 1)

                    # Add 1e-16 to `d` to prevent divide-by-zero and normalise.
                    # Note:  Presently uses infinity norm.
                    d = (self.s_rel * np.abs(y_corr[idx]) + self.s_abs + 1e-16)
                    scl_err = self.w_mat @ new_err / d  # <= OK (CHECKED)

                    sol_i.r_ctrl.append(np.linalg.norm(
                        scl_err, ord=np.inf) / self.tol)  # <= CHK OK

                    try:  # Try intended controller. # <= OK (CHECKED)
                        r_smooth[idx] = filter_r(sol_i.filter, p,
                                                 sol_i.h, sol_i.r_ctrl,
                                                 self.unit_errors)
                    except ValueError:  # Use fallback controller.
                        r_smooth[idx] = filter_r(FILTERS['Elementary'], p,
                                                 sol_i.h, sol_i.r_ctrl,
                                                 self.unit_errors)

            if y_corr[self.sol_idx] is None:
                return False, "Failed to generate a new point."

            # Reject / accept step size.
            if r_smooth[self.sol_idx] >= self.r_min:
                # Step accepted - Update dense output and history.
                main_sol = self.sol[self.sol_idx]
                self._last_dense = MODESDenseOutput(
                    self._t[-1], t_new, main_sol.method.solver_type,
                    main_sol.poly_corr)
                self._t.append(t_new)
                self._y.append(y_corr[self.sol_idx])
                self._ydot.append(self.fun(t_new, y_corr[self.sol_idx]))

                # Update step sizes, capping `r` ratios at r_max.
                rejected_last = False
                for sol_i, r_i in zip(self.sol, r_smooth):
                    if r_i is not None:
                        sol_i.h.append(sol_i.h[-1] * min(r_i, self.r_max))

                    # Shift the prev. corrector to the current predictor poly.
                    sol_i.poly_pred, sol_i.poly_corr = sol_i.poly_corr, None

                self._record_stats()

            else:
                #  Step rejected - Shrink the current step size.
                if not rejected_last:
                    self.sol[self.sol_idx].h[-1] = self.r_min * (
                            self._t[-1] - self._t[-2])  # 1st rejection.
                else:
                    self.sol[self.sol_idx].h[-1] *= 0.95  # Subs. rejns.

                # Clear step data for all orders, except for last value.
                for sol_i in self.sol:
                    if len(sol_i.h) > 0:
                        sol_i.h = deque([sol_i.h[-1]], maxlen=sol_i.q_len)

                self.rejected_steps += 1
                rejected_last = True

            if not rejected_last:
                break

        self.set_running_solvers()

        if self.nfev >= self._disp_next:
            self.vprint(f"... {self.nfev} RHS Evaluations: t = "
                        f"{self._t[-1]:.5G}, h = "
                        f"{self.sol[self.sol_idx].h[-1]:.5G}, "
                        f"p = {self.p} ({self.rejected_steps} Rejected Steps)")

            self._disp_next += int(min(10 ** (np.floor(np.log10(self.nfev))),
                                       1000))

        # Note: The original Matlab code includes an error here.  It doesn't
        # account for the integration direction when checking the endpoint.

        return True, None

    def set_running_solvers(self):
        """
        This is the method that performs order control.  It is called at the
        end of ``__init__()`` and again after each successful step in
        ``_step_impl()``.  It performs two tasks:

        1. Chooses Current Order:  Decides which solver (i.e. order /
           method) is the 'best' to proceed with for the next step.
           `self.sol_idx` is set to the index of the 'best' solver.
        2. Sets Running Orders:  Decides which solvers to advance together
           for the next step.  The flags in `self.solvers[...].running` are
           set either True or False.

        This function can be overridden to use different advancement / order
        control strategies if desired.

        .. note:: This default implementation uses the same method for order
           control as the original MATLAB code, which is to advance the solver
           on each side of the current solver.  The current solver is updated
           based on an internal state `delta_p` that is continuously updated
           by comparing the efficiencies of the solvers running on either
           side.  When the order changes `delta_p` is reset to zero.

        **Differences to MATLAB Implementation**

        - The original MATLAB code monitors orders above and below by
          running an inaccessible / fictitous order on each side; this is
          not done here.
        - If the current solver / order is the top or bottom, `del_p_hi` /
          `_lo` is set to 0 giving no preference to that direction. Also
          if we assume a fictitious higher `sigma` = 1 then +/- `dp` will be
          zero because one of the terms will be zero.
        - From the preprint TOMS paper, instead of Eqn 23 the technique of
          computing `p` +/- 1/2 has been replaced with the average of the
          two adjacent `p` values.  This allows us to use adjacent `p`
          values that are not just +/- 1 steps.
        """
        # Task 1: Choose best solver for next step.

        # Remark: MATLAB code has all workfactors = 1 (constant).  These
        # are left in so they can be included later if desired.
        work_factors = np.ones_like(self.sol)
        h_scld = [sol_i.h[-1] / work_factors[i] if sol_i.running else None
                  for i, sol_i in enumerate(self.sol)]

        # Check performance on both sides of the current order (where
        # possible).  Zero defaults are used when values cannot be computed,
        # i.e. we have reached the low or high `p` limit.
        sigma, del_p, del_p_hilo = [None, None], [0, 0], 0
        p_off = [None, None]
        for i, off_idx, cmp in ((0, self.sol_idx - 1, min),
                                (1, self.sol_idx + 1, max)):
            if (not (0 <= off_idx < self.nsols) or
                    not self.sol[off_idx].running):
                continue
            p_off[i] = self.sol[off_idx].p
            sigma[i] = h_scld[off_idx] / h_scld[self.sol_idx]
            s = (p_off[i] * sigma[i] + self.p) / (sigma[i] + 1)
            d = s - 0.5 * (self.p + p_off[i])  # Changed from MATLAB version.
            del_p[i] = cmp(0, 4 * d)

        if (p_off[0] is not None and p_off[1] is not None and
                (sigma[0] - 1) * (sigma[1] - 1) < 0):  # Changed from MATLAB.
            s0 = ((p_off[1] * sigma[1] + p_off[0] * sigma[0]) /
                  (sigma[1] + sigma[0]))
            del_p_hilo = s0 - self.p

        self.delta_p += (del_p[1] + del_p[0] + del_p_hilo)

        # Compute +/- 1 order change by rounding `delta_p` to nearest
        # choice (i.e. over 0.5). Do not move if the efficiency check on the
        # target order fails.
        old_sol_idx = self.sol_idx
        if self.delta_p > 0.5 and sigma[1] >= 1.1:
            self.sol_idx += 1
        elif self.delta_p < -0.5 and sigma[0] >= 1.1:
            self.sol_idx -= 1

        self.sol_idx = max(min(self.sol_idx, self.nsols - 1), 0)  # Limit.
        if self.sol_idx != old_sol_idx:
            self.delta_p = 0  # Reset accumulated change.

        # Task 2: Set solvers as either running or stopped.
        for idx, sol_i in enumerate(self.sol):
            if self.sol_idx - 1 <= idx <= self.sol_idx + 1:
                sol_i.running = True
                if len(sol_i.h) == 0:  # Inherit step size if reqd.
                    sol_i.h.append(self.h)

            else:
                sol_i.running = False
                sol_i.h.clear()
                sol_i.r_ctrl.clear()
                sol_i.poly_corr, sol_i.poly_pred = None, None

    def predict(self, sol_idx: int) -> Optional[np.ndarray]:
        """Generates a predictor polynomial for the solver at index
        `sol_idx` and prediction point (where possible), for the next
        timepoint.

        .. note:: Original MATLAB implementation includes a ``tril()``
           which has been omitted here.  Refer to source code for more details.

        """
        sol = self.sol[sol_idx]
        sol_type = sol.method.solver_type

        if sol.poly_pred is None:
            # Try to build predictor poly if reqd (i.e. solver startup).

            if sol_type == SolverType.EXPLICIT:
                # Build from steps prior to current.
                sol.poly_pred = self._coeffs_explicit(sol.theta, trim=-1)

            elif sol_type == SolverType.IMPLICIT:
                # Use explicit method over the first p - 1 points.
                sol.poly_pred = self._coeffs_explicit(sol.theta[:-1])

            elif sol_type == SolverType.IMPLICIT_PLUS:
                # Use the relation between explicit/implicit methods to
                # calculate what the old coefficients would have been if
                # this method was previously used.
                p = sol.p

                thetas_expl = np.append(sol.theta, np.pi / 2)
                poly_expl = self._coeffs_explicit(thetas_expl)
                if poly_expl is not None:
                    bin_coeffs_mat = np.fliplr(np.rot90(np.linalg.cholesky(
                        pascal(p + 1))))

                    # Note:  This binomial coefficient matrix in the MATLAB
                    # version however may need review. The matrix values
                    # depend on pmax however it seems that this shouldn't be
                    # the case; pmax should be able to be higher or lower
                    # without affecting the outcome of individual orders.

                    v = np.arange(1, p + 1)[np.newaxis]
                    w = np.arange(0, -p, -1)
                    pwrs_mat = np.tril(v.T + w) + np.triu(np.ones(p), 1)
                    pwrs_mat = np.vstack([np.zeros(p), np.tril(pwrs_mat)])

                    # The polynomial uses a local coordinate system and
                    # needs to be translated, so that it is evaluated in the
                    # correct point later on when the prediction value is
                    # calculated.

                    # Note: The TRIL on the step size here in the MATLAB
                    # code seems very odd; it doesn't seem to do anything.
                    # Maybe it's used to get a matrix as the result?
                    h_prev = self._t[-1] - self._t[-2]
                    last_col = np.asarray([*[0.0] * p, 1])[:, np.newaxis]
                    a_pwrs_mat = np.hstack([np.power(-h_prev, pwrs_mat),
                                            last_col])
                    sol.poly_pred = (bin_coeffs_mat * a_pwrs_mat) @ poly_expl

                # ------------------------------------------------------------
                # Note: The section below is quarantined.  This is an
                # implementation of the MATLAB code for the initial
                # predictor polynomial at startup, however it seems to serve
                # exactly the same function as the 'matrix magic'
                # displacement method that happens at an order change.  As
                # such it seems unecessary to have two different means of
                # accomplishing the same goal.
                #
                # ...
                #
                # Get a starting guess for the implicit poly solver by using
                # Adams-Bashforth of order p-1, then top pad with zeros.
                #
                # theta_expl = adams_bashforth.theta(sol.method.k(p))
                # start_guess = self._coeffs_explicit(theta_expl, trim=-1)
                # if start_guess is not None:
                #     start_guess = pad_coeffs(start_guess, p)
                #     sol.poly_pred = self._coeffs_implicit_plus(
                #         sol.theta, start_guess, self._t[-1], trim=-1)
                # ------------------------------------------------------------

        if sol.poly_pred is not None:
            if sol_type == SolverType.IMPLICIT:
                return polyval_cols(sol.poly_pred, self.h)
            else:
                return polyval_cols(sol.poly_pred, self.h +
                                    (self._t[-1] - self._t[-2]))  # h + h[-1]
        else:
            return None

    def correct(self, sol_idx: int) -> Optional[np.ndarray]:
        """Generates a corrector polynomial for the solver at index
        `sol_idx` and solution point (where possible) for the next timepoint.
        """
        sol = self.sol[sol_idx]
        sol_type = sol.method.solver_type

        if sol_type == SolverType.EXPLICIT:
            if sol.poly_corr is None:
                # Sometimes poly already exists (e.g. after rejection).
                sol.poly_corr = self._coeffs_explicit(sol.theta)

            if sol.poly_corr is not None:
                return polyval_cols(sol.poly_corr, self.h)
            else:
                return None

        start_guess = sol.poly_pred  # Starting guess for implicit solver.
        t_new = self._t[-1] + self.h

        if start_guess is None and sol_type == SolverType.IMPLICIT_PLUS:
            lwr_idx = sol_idx - 1
            if lwr_idx >= 0 and self.sol[lwr_idx].poly_pred is not None:
                # Use next lower order as starting guess.  I believe this is
                # the intent of the MATLAB code (once you tease it all out).
                start_guess = self.sol[lwr_idx].poly_pred
            else:
                # Use an explicit Adams-Bashforth over `k` previous points
                # (including current point) as starting guess.  This is only
                # required if lower order predictor could not be determined.
                start_guess = self._coeffs_explicit(
                    adams_bashforth.theta(sol.method.k(sol.p)))

            if start_guess is not None:
                start_guess = pad_coeffs(start_guess, sol.p)  # 0-pad to `p`.

        # Solve for implicit coefficients where possible.
        if start_guess is not None:
            if sol_type == SolverType.IMPLICIT:
                sol.poly_corr = self._coeffs_implicit(sol.theta,
                                                      sol.poly_pred, t_new)
                if sol.poly_corr is not None:
                    return sol.poly_corr[-1, :]

            else:
                sol.poly_corr = self._coeffs_implicit_plus(
                    sol.theta, start_guess, t_new)
                if sol.poly_corr is not None:
                    return polyval_cols(sol.poly_corr, self.h)

        return None

    def _dense_output_impl(self) -> MODESDenseOutput:
        return self._last_dense

    # Utility Methods --------------------------------------------------------

    def vprint(self, msg):
        if self.verbose:
            print(msg)

    # noinspection PyPep8Naming
    def _coeffs_explicit(self, theta: np.ndarray, trim: int = None
                         ) -> Optional[np.ndarray]:
        r"""Compute the polynomial coefficients for an explicit `k`-step
        methods of order `k`.

        At least ``len(thetas) + 1`` past values from `self.t`. `self.y` and
        `self.y_dot` are required for solution.

        Parameters
        ----------
        theta : ndarray, shape(k-1,)
            :math:`\theta` values defining the method.
        trim : int, optional
            If non-zero then this value is used as the index of the last
            historic value to use.  E.G. Using ``trim = -1`` causes the last
            value to be omitted.

        Returns
        -------
        x : ndarray or None
            Returns the polynomial coefficients found by solving the linear
            system.  If insufficient data was available to find a solution
            it returns None.
        """
        k = len(theta) + 1
        t = list(self._t)[:trim]
        y = list(self._y)[:trim]
        ydot = list(self._ydot)[:trim]

        if len(t) < k:
            return None

        # Sines and cosines of the method parameters.
        cos_th, sin_th = np.cos(theta), np.sin(theta)

        # Write matrix problem A.P = R
        A = np.zeros((k + 1, k + 1))
        A[k - 1, k - 1] = 1
        A[k, k] = 1
        A[0:k - 1, k] = cos_th

        rhs = np.zeros((k + 1, self.n))
        rhs[k - 1, :] = ydot[-1]
        rhs[k, :] = y[-1]

        for i in range(k - 1):
            h_kmi = t[-1 - i] - t[-2 - i]  # Step size rev. order.
            s_i = t[-2 - i] - t[-1]  # Accumulated step size.

            for j in range(k):
                A[i, j] = (cos_th[i] * s_i ** (k - j) +
                           (k - j) * sin_th[i] * h_kmi * s_i ** (k - j - 1))
            rhs[i, :] = (cos_th[i] * y[-2 - i] +
                         sin_th[i] * h_kmi * ydot[-2 - i])

        return np.linalg.solve(A, rhs)

    def _coeffs_implicit(self, thetas: np.ndarray, coeffs_start: np.ndarray,
                         t_new: float) -> Optional[np.ndarray]:
        """
        Compute the polynomial coefficients for an implicit `k`-step methods
        of order `k`.  Operation as per ``_coeffs_explicit()``, except where
        noted.

        Parameters
        ----------
        coeffs_start: ndarray, shape(k+1, n)
            A matrix containing an estimate starting point for the implicit
            solution.  Typically the coefficients for the previous
            polynomials are used.
        t_new : float
            The new time value.

        Returns
        -------
        x : ndarray or None
            Returns the polynomial coefficients found by solving the
            non-linear system.  If insufficient data was available to find
            a solution it returns None.
        """
        ndim = self.n  # Shorthand quantities.
        t, y, ydot = list(self._t), list(self._y), list(self._ydot)
        k = len(thetas)
        if len(t) < k:
            return None

        w = coeffs_start.flatten()  # Reshape coeffs to problem vector.

        # Precompute the constant part of F and some commonly used values.
        cos_th, sin_th = np.cos(thetas), np.sin(thetas)
        h_k1i = np.asarray([t_new - t[-1]] + [t[-i] - t[-1 - i] for i in
                                              range(1, k)])  # Steps rev. ord.
        s_i, s1 = np.empty(k), np.zeros(k * ndim)
        for i in range(k):
            s_i[i] = t[-1 - i] - t_new  # Accum. step size.
            s1[i * ndim: (i + 1) * ndim] = (cos_th[i] * y[-1 - i] + sin_th[i]
                                            * h_k1i[i] * ydot[-1 - i])  # CHKD
        if 'i' in locals():
            del i  # To avoid inner scope errors.

        # noinspection PyPep8Naming
        def resid(w_try: np.ndarray):
            F = np.zeros((k + 1) * ndim)
            for i2 in range(k):
                f_s = (cos_th[i2] * s_i[i2] + sin_th[i2] * h_k1i[i2] *
                       k) * w_try[0:ndim]
                for j in range(1, k):
                    f_s = (s_i[i2] * f_s +
                           (cos_th[i2] * s_i[i2] + sin_th[i2] * h_k1i[i2] *
                            (k - j)) * w_try[j * ndim:(j + 1) * ndim])

                F[i2 * ndim: (i2 + 1) * ndim] = (
                        f_s + cos_th[i2] * w_try[k * ndim: (k + 1) * ndim] -
                        s1[i2 * ndim: (i2 + 1) * ndim])

            F[k * ndim: (k + 1) * ndim] = (
                    np.asarray(self.fun(
                        t_new, w_try[k * ndim: (k + 1) * ndim])) -
                    w_try[(k - 1) * ndim: k * ndim])
            return F  # <= CHKD OK

        # noinspection PyUnusedLocal,PyPep8Naming
        def const_jacobian(*unused_args):
            # def const_jacobian():
            d = self.jac(t_new, y[-1])  # <= CHKD OK
            unit = np.eye(ndim)
            J = np.zeros(((k + 1) * ndim, (k + 1) * ndim))
            for i3 in range(k):
                for j in range(k):
                    J[i3 * ndim: (i3 + 1) * ndim, j * ndim: (j + 1) * ndim] = (
                            s_i[i3] ** (k - j - 1) *
                            (cos_th[i3] * s_i[i3] + sin_th[i3] * h_k1i[i3] *
                             (k - j)) * unit)

                J[i3 * ndim:(i3 + 1) * ndim, k * ndim:(k + 1) * ndim] = (
                        cos_th[i3] * unit)

            J[k * ndim: (k + 1) * ndim, (k - 1) * ndim: k * ndim] = -unit
            J[k * ndim: (k + 1) * ndim, k * ndim: (k + 1) * ndim] = d
            return J  # <= CHKD OK

        res_x = self._solve_implicit_system(resid, w, const_jacobian)

        p = np.reshape(res_x, (k + 1, ndim))  # <= CHKD OK
        return p

    def _coeffs_implicit_plus(self, thetas: np.ndarray,
                              coeffs_start: np.ndarray, t_new: float,
                              trim: int = None) -> Optional[np.ndarray]:
        """
        Compute the polynomial coefficients for an implicit `k`-step methods
        of order `k` + 1.  Operation as per ``_coeffs_implicit()``, except
        where noted.

        **Differences to MATLAB Implementation**

        * There is a subtle change to ``flip()`` when unpacking unknowns.

        Parameters
        ----------
        coeffs_start : ndarray, shape(k+2, n)
            Similar to ``coeffs_implicit()``.
        trim : int
            As per ``_coeffs_explicit()``.
        """
        ndim = self.n  # Shorthand quantities.
        t = list(self._t)[:trim]
        y = list(self._y)[:trim]
        ydot = list(self._ydot)[:trim]
        k = len(thetas) + 1
        if len(t) < k:
            return None

        #  Reshape the coeffecient matrix to a vector and remove the two
        #  last coefficients (due to the structure we already know them).
        w = np.flipud(coeffs_start[0:k, :]).T.flatten(order='F')

        # Note: There is an error in original MATLAB polIp() for case of
        # k = 1 where flip() reverses the order of elements in the
        # individual vector instead of the order of the vectors.  The original
        # MATLAB should instead use flip(_, 1) which enforces flipping of
        # the elements along each column.

        # Compute the constant part of F and some commonly used values.  Note:
        # `y` and `y_dot` appear in these equations because these are actually
        # the last two coefficients defining the polynomial.
        cos_th = np.cos(thetas)
        s1 = np.zeros((k - 1) * ndim)  # <= CHKD
        h_kmi, s_i = np.empty(k - 1), np.empty(k - 1)
        sin_th_h = np.empty(k-1)
        for i in range(k - 1):
            h_kmi[i] = t[-1 - i] - t[-2 - i]  # Steps rev. order.
            s_i[i] = t[-2 - i] - t[-1]  # Accum. step size.
            sin_th_h[i] = np.sin(thetas[i]) * h_kmi[i]
            s1[i * ndim:(i+1) * ndim] = (
                    cos_th[i] * (y[-1] - y[-2 - i] + s_i[i] * ydot[-1]) +
                    sin_th_h[i] * (ydot[-1] - ydot[-2 - i]))  # <= CHKD OK

        if 'i' in locals():
            del i  # To avoid inner scope errors.

        def resid(w_try: np.ndarray):
            f_vec = np.zeros(k * ndim)
            for i2 in range(k - 1):
                fs_vec = ((cos_th[i2] * s_i[i2] + sin_th_h[i2] * (k + 1)) *
                          w_try[(k - 1) * ndim:k * ndim])  # <= CHKD OK
                for j in range(1, k):
                    fs_vec = (s_i[i2] * fs_vec +
                              (cos_th[i2] * s_i[i2] + (k + 1 - j) *
                               sin_th_h[i2]) *
                              w_try[(k - j - 1) * ndim:(k - j) * ndim])

                f_vec[i2 * ndim: (i2 + 1) * ndim] = (
                        s_i[i2] * fs_vec + s1[i2 * ndim:(i2 + 1) * ndim])

            hn = t_new - t[-1]
            sn = w_try[(k - 1) * ndim:k * ndim]
            for n in range(k - 1, 0, -1):
                sn = hn * sn + w_try[(n - 1) * ndim:n * ndim]
            sn = hn * sn + ydot[-1]
            yn = hn * sn + y[-1]
            sn = (k + 1) * w_try[(k - 1) * ndim:k * ndim]
            for n in range(k - 1, 0, -1):
                sn = hn * sn + (n + 1) * w_try[(n - 1) * ndim:n * ndim]
            ypn = hn * sn + ydot[-1]
            f_vec[(k - 1) * ndim: k * ndim] = (np.asarray(self.fun(t_new, yn))
                                               - ypn)
            return f_vec  # <= CHKD OK

        # noinspection PyUnusedLocal, PyPep8Naming
        def const_jacobian(*unused_args):
            # def const_jacobian():
            d = self.jac(t_new, y[-1])
            unit = np.eye(ndim)
            J = np.zeros((k * ndim, k * ndim))
            for i3 in range(k - 1):
                s_1 = s_i[i3]
                for j2 in range(k):
                    J[i3 * ndim: (i3 + 1) * ndim,
                      j2 * ndim: (j2 + 1) * ndim] = (
                            s_1 * (cos_th[i3] * s_1 + (j2 + 2) * sin_th_h[i3])
                            * unit)
                    s_1 *= s_i[i3]

            hk = t_new - t[-1]
            hkj = hk
            for j2 in range(k):
                J[(k - 1) * ndim:k * ndim, j2 * ndim:(j2 + 1) * ndim] = hkj * (
                        hk * d - (j2 + 2) * unit)
                hkj *= hk
            return J  # <= CHKD OK

        res_x = self._solve_implicit_system(resid, w, const_jacobian)

        p = np.flipud(np.reshape(res_x, (ndim, k), order='F').T)
        p = np.vstack([p, ydot[-1], y[-1]])
        return p

    # noinspection PyUnresolvedReferences
    def _solve_implicit_system(self, resid_fn, w, jac_fn):
        """
        Solves a given implicit system using a primary solver or fallback
        solver if the primary solver fails (where requested).

        Parameters
        ----------
        resid_fn : callable
            Function returning the RHS residual of the matrix system.
        w : ndarray
            Solution array.
        jac_fn : callable
            Jacobian function.

        Returns
        -------
        x : ndarray
            Solution to the system.

        Raises
        ------
        RuntimeError
            If a primary (or fallback) solution could not be computed.
        """

        def get_solver_opts(method, its):
            if method == 'newton_const':
                _sol = newton_const
            else:
                _sol = partial(root, method=method)

            if method in ('hybr', 'df-sane'):
                _opts = {'maxfev': its}
            else:
                _opts = {'maxiter': its}

            return _sol, _opts

        sol, opts = get_solver_opts(self.impl_solver, self.impl_its)
        res = sol(resid_fn, w, jac=jac_fn, tol=self.tol / 10, options=opts)
        if res.success:
            return res.x

        if self.impl_backup is not None:
            # If we failed, try the backup with internally-computed Jacobian
            # (as required) and more iterations.
            sol, opts = get_solver_opts(self.impl_backup, 2 * self.impl_its)
            res = sol(resid_fn, w, jac=False, tol=self.tol / 10, options=opts)
            more_its = res.get('nit', None) or res.get('nfev', None)
            self.vprint(f"...Implicit '{self.impl_solver}' solver failed, "
                        f"used fallback '{self.impl_backup}' at t = "
                        f"{self.t:.5G} ({more_its} evaluations).")
            if res.success:
                return res.x

        raise RuntimeError(f"Implicit solver/s failed.")

    def _record_stats(self):
        if self._run_stats is not None:
            self._run_stats['t'].append(self.t)
            self._run_stats['h'].append(self.h)
            self._run_stats['p'].append(self.p)
            if len(self.sol[self.sol_idx].r_ctrl) > 0:
                self._run_stats['err_norm'].append(
                    self.sol[self.sol_idx].r_ctrl[-1] * self.tol)
            else:
                self._run_stats['err_norm'].append(0)
            self._run_stats['rej_steps'].append(self.rejected_steps)

            if self.analytic_fn is not None:
                y_a = np.asarray(self.analytic_fn(self.t))
                self._run_stats['y_analytic'].append(y_a)
                self._run_stats['err_norm_analytic'].append(
                    np.linalg.norm(self.y - y_a, 2))
