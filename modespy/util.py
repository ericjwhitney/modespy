"""
Utility functions used in various places by MODESpy.

**Differences to MATLAB Implementation**

* **(Version 0.9.0)** Convergence check of modified Newton-Raphson method has
  been changed.  Refer to ``newton_const()`` source code for details.

"""

# Last updated: 4 June 2020 by Eric J. Whitney

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import OptimizeResult


# noinspection PyPep8Naming
def newton_const(fun, x0, *, jac, args=(), tol=1e-6, options=None):
    r"""Solve system of equations using a Newton-Raphson iteration with a
    constant Jacobian term computed only on the first iteration.  This
    function is designed to be called in an identical fashion as ``root()``
    in `scipy.optimise`.

    Parameters
    ----------
    fun : callable
        A function ``f(x, *args)`` that takes a vector argument, and
        returns a value of the same length.
    x0 : ndarray
        The starting estimate for the root of ``fun(x) = 0``.
    jac : array_like or callable
        Jacobian matrix of the right-hand side of the system. The Jacobian
        matrix has shape (n, n) and element (i, j) is equal to
        :math:`df_i/dy_j`:

        * If array-like:  It is assumed to be a constant matrix.
        * If callable, the Jacobian is assumed to depend on both `t` and
          `y`; it will be called once-only as ``jac(t, y)``.
    args : tuple, optional
        Any extra arguments to `fun`.
    tol : float
        The calculation will terminate if the infinity norm of the
        last correction step was less than `tol` (default = 1e-6).
    options : dict, optional
        A dictionary of solver options.  Available options are:
            'maxiter' : int
                The maximum number of iterations / calls to the function
                (default = 50).

    Returns
    -------
    Bunched object with the following fields:
    x : ndarray
        Solution vector.
    success : bool
        True if the solution was found within required tolerance and
        number of iterations.
    status : int
        * 0: The solver finished successfully.
        * 1: The number of calls to `f(x)` has reached ``maxfev``.
    message : str
        Verbose description of the status.
    nfev : int
        Number of function (RHS) evaluations.
    njev : int
        Number of Jacobian evaluations.
    nit : int
        Number of iterations of the solver.
    """

    if options is None:
        options = {}

    maxiter = options.get('maxiter', 50)
    x = x0.copy()
    if jac is None:
        raise TypeError("newton_const() cannot be called without Jacobian.")
    elif callable(jac):
        J = jac(x, *args)
    else:
        J = jac
    LUP = lu_factor(J)  # Factorise just once.

    nit, status = 0, 1
    while nit < maxiter:
        f_val = fun(x, *args)
        delta = lu_solve(LUP, -f_val)  # <= CHKD OK
        x += delta
        nit += 1
        if np.linalg.norm(delta, ord=np.inf) < tol:
            status = 0
            break

    # Note:  The original MATLAB implementation of the Newton method has an
    # error where there is no check for proper convergence after 12 passes.
    # This become apparent particularly when when IDC and Milne4 methods are
    # used.

    if status == 0:
        success, message = (True, "The solution converged.")
    else:
        success, message = (False, f"The number of calls to function has "
                                   f"reached maxfev = {maxiter}.")
    return OptimizeResult(x=x, success=success, status=status, message=message,
                          nfev=nit, njev=1, nlu=1, nit=nit)


def pad_coeffs(coeffs: np.ndarray, target_p: int) -> np.ndarray:
    """Return polynomial coefficients `coeffs` with the higher order end
    padded with rows of zero to suit requested higher order `target_p`."""
    add_rows = target_p + 1 - coeffs.shape[0]
    return np.vstack([np.zeros((add_rows, coeffs.shape[1])), coeffs])


def polyval_cols(p: np.ndarray, x: float):
    """Evaluate multiple polynomials as *columns* of `p`, using the same
    value `x` each time.  Returns an array of the results."""
    return np.array([np.polyval(p_i, x) for p_i in p.T])
