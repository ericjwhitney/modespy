"""
This module defines various standard problems from the literature,
commonly used as benchmarks for different ODE methods.

Equations, descriptions, limits, etc for each problem are stored together in a
`Problem` dataclass.  All problems are stored in the module-wide constant
`PROBLEMS`.

At present standard problems include:

* **Non-Stiff:**

    * Lotka-Volterra
    * Pleiades

* **Stiff:**

    * HIRES
    * Pollution

* **Both:**

    * Butterfly

Future
------
* More standard problems can be included.

"""

# Last updated: 4 June 2020 by Eric J. Whitney

from dataclasses import dataclass
import numpy as np
from typing import Callable, Sequence, Optional, Tuple

EQNSystem = Callable[[float, Sequence[float]], Sequence[float]]


@dataclass
class Problem:
    equation: EQNSystem
    jacobian: Optional[EQNSystem] = None
    analytic: Optional[EQNSystem] = None
    default_t: Optional[Tuple[float, float]] = None
    default_y0: Optional[Sequence[float]] = None
    description: Optional[str] = None


# ----------------------------------------------------------------------------
# Non-stiff problems.

# noinspection PyUnusedLocal
def lotka_rhs(t, y):
    alpha, beta = 0.1, 0.3
    delta, gamma = 0.5, 0.5
    return 30 * np.array([alpha * y[0] - beta * y[0] * y[1],
                          delta * y[0] * y[1] - gamma * y[1]])


# noinspection PyUnusedLocal
def lotka_jac(t, y):
    alpha, beta = 0.1, 0.3
    gamma, delta = 0.5, 0.5
    return 30 * np.array([[alpha - beta * y[1], -beta * y[0]],
                          [delta * y[1], -gamma + delta * y[0]]])


# noinspection PyUnusedLocal
def pleiades_rhs(t, y):
    f = np.zeros(28)
    for i in range(7):
        sum_x, sum_y = 0, 0
        for j in range(7):
            rij = (y[i] - y[j]) ** 2 + (y[i + 7] - y[j + 7]) ** 2
            rij32 = rij ** (3 / 2)
            if j != i:
                sum_x += j * (y[j] - y[i]) / rij32
                sum_y += j * (y[j + 7] - y[i + 7]) / rij32

        f[i] = y[i + 14]
        f[i + 14] = sum_x
        f[i + 21] = sum_y

    for i in range(7, 14):
        f[i] = y[i + 14]
    return f


PLEIADES_X_0 = [3, 3, -1, -3, 2, -2, 2]
PLEIADES_Y_0 = [3, -3, 2, 0, 0, -4, 4]
PLEIADES_XD_0 = [0, 0, 0, 0, 0, 1.75, -1.5]
PLEIADES_YD_0 = [0, 0, 0, -1.25, 1, 0, 0]


# ----------------------------------------------------------------------------
# Stiff problems.

# HIRES.

# noinspection PyUnusedLocal
def hires_rhs(t, y):
    dy = np.zeros(8)
    dy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007
    dy[1] = 1.71 * y[0] - 8.75 * y[1]
    dy[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4]
    dy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3]
    dy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6]
    dy[5] = (-280 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] +
             0.69 * y[6])
    dy[6] = 280 * y[5] * y[7] - 1.81 * y[6]
    dy[7] = -280 * y[5] * y[7] + 1.81 * y[6]
    return dy


# noinspection PyUnusedLocal
def hires_jac(t, y):
    jac = np.zeros((8, 8))
    jac[0, :] = [-1.71, 0.43, 8.32, 0, 0, 0, 0, 0]
    jac[1, :] = [1.71, -8.75, 0, 0, 0, 0, 0, 0]
    jac[2, :] = [0, 0, -10.03, 0.43, 0.035, 0, 0, 0]
    jac[3, :] = [0, 8.32, 1.71, -1.12, 0, 0, 0, 0]
    jac[4, :] = [0, 0, 0, 0, -1.745, 0.43, 0.43, 0]
    jac[5, :] = [0, 0, 0, 0.69, 1.71, -280 * y[7] - 0.43, 0.69, -280 * y[5]]
    jac[6, :] = [0, 0, 0, 0, 0, 280 * y[7], -1.81, 280 * y[5]]
    jac[7, :] = [0, 0, 0, 0, 0, -280 * y[7], 1.81, -280 * y[5]]
    return jac


# Pollution.

_POLL_K = np.array([0.350, 0.266e2, 0.123e5, 0.860e-3, 0.820e-3, 0.150e5,
                    0.130e-3, 0.240e5, 0.165e5, 0.900e4, 0.220e-1, 0.120e5,
                    0.188e1, 0.163e5, 0.480e7, 0.350e-3, 0.175e-1, 0.100e9,
                    0.444e12, 0.124e4, 0.210e1, 0.578e1, 0.474e-1, 0.178e4,
                    0.312e1])


# noinspection PyUnusedLocal
def pollution_rhs(t, y):
    r = np.array([_POLL_K[0] * y[0],
                  _POLL_K[1] * y[1] * y[3],
                  _POLL_K[2] * y[4] * y[1],
                  _POLL_K[3] * y[6],
                  _POLL_K[4] * y[6],
                  _POLL_K[5] * y[6] * y[5],
                  _POLL_K[6] * y[8],
                  _POLL_K[7] * y[8] * y[5],
                  _POLL_K[8] * y[10] * y[1],
                  _POLL_K[9] * y[10] * y[0],
                  _POLL_K[10] * y[12],
                  _POLL_K[11] * y[9] * y[1],
                  _POLL_K[12] * y[13],
                  _POLL_K[13] * y[0] * y[5],
                  _POLL_K[14] * y[2],
                  _POLL_K[15] * y[3],
                  _POLL_K[16] * y[3],
                  _POLL_K[17] * y[15],
                  _POLL_K[18] * y[15],
                  _POLL_K[19] * y[16] * y[5],
                  _POLL_K[20] * y[18],
                  _POLL_K[21] * y[18],
                  _POLL_K[22] * y[0] * y[3],
                  _POLL_K[23] * y[18] * y[0],
                  _POLL_K[24] * y[19]])

    dy = np.array([(-r[0] - r[9] - r[13] - r[22] - r[23] + r[1] + r[2] +
                    r[8] + r[10] + r[11] + r[21] + r[24]),
                   -r[1] - r[2] - r[8] - r[11] + r[0] + r[20],
                   -r[14] + r[0] + r[16] + r[18] + r[21],
                   -r[1] - r[15] - r[16] - r[22] + r[14],
                   -r[2] + r[3] + r[3] + r[5] + r[6] + r[12] + r[19],
                   -r[5] - r[7] - r[13] - r[19] + r[2] + r[17] + r[17],
                   -r[3] - r[4] - r[5] + r[12],
                   r[3] + r[4] + r[5] + r[6],
                   -r[6] - r[7],
                   -r[11] + r[6] + r[8],
                   -r[8] - r[9] + r[7] + r[10],
                   r[8],
                   -r[10] + r[9],
                   -r[12] + r[11],
                   r[13],
                   -r[17] - r[18] + r[15],
                   -r[19],
                   r[19],
                   -r[20] - r[21] - r[23] + r[22] + r[24],
                   -r[24] + r[23]])
    return dy


# noinspection PyUnusedLocal
def pollution_jac(t, y):
    jac = np.zeros((20, 20))

    # Row 0.
    jac[0, 0] = (-_POLL_K[0] - _POLL_K[9] * y[10] - _POLL_K[13] * y[5] -
                 _POLL_K[22] * y[3] - _POLL_K[23] * y[18])
    jac[0, 1] = (_POLL_K[11] * y[9] + _POLL_K[1] * y[3] + _POLL_K[2] *
                 y[4] + _POLL_K[8] * y[10])
    jac[0, 3] = _POLL_K[1] * y[1] - _POLL_K[22] * y[0]
    jac[0, 4] = _POLL_K[2] * y[1]
    jac[0, 5] = -_POLL_K[13] * y[0]
    jac[0, 9] = _POLL_K[11] * y[1]
    jac[0, 10] = -_POLL_K[9] * y[0] + _POLL_K[8] * y[1]
    jac[0, 12] = _POLL_K[10]
    jac[0, 18] = _POLL_K[21] - _POLL_K[23] * y[0]
    jac[0, 19] = _POLL_K[24]

    # Row 1.
    jac[1, 0] = _POLL_K[0]
    jac[1, 1] = (-_POLL_K[11] * y[9] - _POLL_K[1] * y[3] - _POLL_K[2] *
                 y[4] - _POLL_K[8] * y[10])
    jac[1, 3] = -_POLL_K[1] * y[1]
    jac[1, 4] = -_POLL_K[2] * y[1]
    jac[1, 9] = -_POLL_K[11] * y[1]
    jac[1, 10] = -_POLL_K[8] * y[1]
    jac[1, 18] = _POLL_K[20]

    # Row 2.
    jac[2, 0] = _POLL_K[0]
    jac[2, 2] = -_POLL_K[14]
    jac[2, 3] = _POLL_K[16]
    jac[2, 15] = _POLL_K[18]
    jac[2, 18] = _POLL_K[21]
    jac[2, 19] = 0

    # Row 3.
    jac[3, 0] = -_POLL_K[22] * y[3]
    jac[3, 1] = -_POLL_K[1] * y[3]
    jac[3, 2] = _POLL_K[14]
    jac[3, 3] = (-_POLL_K[15] - _POLL_K[16] - _POLL_K[1] * y[1] -
                 _POLL_K[22] * y[0])

    # Row 4.
    jac[4, 1] = -_POLL_K[2] * y[4]
    jac[4, 4] = -_POLL_K[2] * y[1]
    jac[4, 5] = _POLL_K[19] * y[16] + _POLL_K[5] * y[6]
    jac[4, 6] = 2 * _POLL_K[3] + _POLL_K[5] * y[5]
    jac[4, 8] = _POLL_K[6]
    jac[4, 13] = _POLL_K[12]
    jac[4, 16] = _POLL_K[19] * y[5]

    # Row 5.
    jac[5, 0] = -_POLL_K[13] * y[5]
    jac[5, 1] = _POLL_K[2] * y[4]
    jac[5, 4] = _POLL_K[2] * y[1]
    jac[5, 5] = (-_POLL_K[13] * y[0] - _POLL_K[19] * y[16] - _POLL_K[5] *
                 y[6] - _POLL_K[7] * y[8])
    jac[5, 6] = -_POLL_K[5] * y[5]
    jac[5, 8] = -_POLL_K[7] * y[5]
    jac[5, 15] = 2 * _POLL_K[17]
    jac[5, 16] = -_POLL_K[19] * y[5]

    # Row 6.
    jac[6, 5] = -_POLL_K[5] * y[6]
    jac[6, 6] = -_POLL_K[3] - _POLL_K[4] - _POLL_K[5] * y[5]
    jac[6, 13] = _POLL_K[12]

    # Row 7.
    jac[7, 5] = _POLL_K[5] * y[6]
    jac[7, 6] = _POLL_K[3] + _POLL_K[4] + _POLL_K[5] * y[5]
    jac[7, 8] = _POLL_K[6]

    # Row 8.
    jac[8, 5] = -_POLL_K[7] * y[8]
    jac[8, 8] = -_POLL_K[6] - _POLL_K[7] * y[5]

    # Row 9.
    jac[9, 1] = -_POLL_K[11] * y[9] + _POLL_K[8] * y[10]
    jac[9, 8] = _POLL_K[6]
    jac[9, 9] = -_POLL_K[11] * y[1]
    jac[9, 10] = _POLL_K[8] * y[1]

    # Row 10.
    jac[10, 0] = -_POLL_K[9] * y[10]
    jac[10, 1] = -_POLL_K[8] * y[10]
    jac[10, 5] = _POLL_K[7] * y[8]
    jac[10, 8] = _POLL_K[7] * y[5]
    jac[10, 10] = -_POLL_K[9] * y[0] - _POLL_K[8] * y[1]
    jac[10, 12] = _POLL_K[10]

    # Row 11.
    jac[11, 1] = _POLL_K[8] * y[10]
    jac[11, 10] = _POLL_K[8] * y[1]

    # Row 12.
    jac[12, 0] = _POLL_K[9] * y[10]
    jac[12, 10] = _POLL_K[9] * y[0]
    jac[12, 12] = -_POLL_K[10]

    # Row 13.
    jac[13, 1] = _POLL_K[11] * y[9]
    jac[13, 9] = _POLL_K[11] * y[1]
    jac[13, 13] = -_POLL_K[12]

    # Row 14.
    jac[14, 0] = _POLL_K[13] * y[5]
    jac[14, 5] = _POLL_K[13] * y[0]

    # Row 15.
    jac[15, 3] = _POLL_K[15]
    jac[15, 15] = -_POLL_K[17] - _POLL_K[18]

    # Row 16.
    jac[16, 5] = -_POLL_K[19] * y[16]
    jac[16, 16] = -_POLL_K[19] * y[5]

    # Row 17.
    jac[17, 5] = _POLL_K[19] * y[16]
    jac[17, 16] = _POLL_K[19] * y[5]

    # Row 18.
    jac[18, 0] = _POLL_K[22] * y[3] - _POLL_K[23] * y[18]
    jac[18, 3] = _POLL_K[22] * y[0]
    jac[18, 18] = -_POLL_K[20] - _POLL_K[21] - _POLL_K[23] * y[0]
    jac[18, 19] = _POLL_K[24]

    # Row 19.
    jac[19, 0] = _POLL_K[23] * y[18]
    jac[19, 18] = _POLL_K[23] * y[0]
    jac[19, 19] = -_POLL_K[24]

    return jac


# ----------------------------------------------------------------------------
# Both stiff and non-stiff problems.

BUTTERFLY_MU = 5
BUTTERFLY_GAMMA = 2


# noinspection PyPep8Naming
def butterfly_rhs(t: float, y: Sequence[float]):
    f_t = np.array([(np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                     2 * np.cos(4 * t)) * np.sin(t),
                    (np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                     2 * np.cos(4 * t)) * np.cos(t)])

    fd_t = np.array([(-np.exp(np.cos(t)) * np.sin(t) - 5 * np.sin(t / 12) ** 4
                      * np.cos(t / 12) / 12 + 8 * np.sin(4 * t)) * np.sin(t) +
                     (np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                      2 * np.cos(4 * t)) * np.cos(t),
                     (-np.exp(np.cos(t)) * np.sin(t) - 5 * np.sin(t / 12) ** 4
                      * np.cos(t / 12) / 12 + 8 * np.sin(4 * t)) * np.cos(t) -
                     (np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                      2 * np.cos(4 * t)) * np.sin(t)])

    A = np.array([[-2, -1],
                  [-3, -2]])

    def non_linear_part(x):
        return np.array([x[0] * x[1], x[1] ** 3 - x[0]])

    def phi(x):
        return BUTTERFLY_MU * A @ x + BUTTERFLY_GAMMA * non_linear_part(x)

    return phi(y - f_t) + fd_t


# noinspection PyPep8Naming
def butterfly_jac(t: float, y: Sequence[float]):
    f_t = np.array([(np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                     2 * np.cos(4 * t)) * np.sin(t),
                    (np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                     2 * np.cos(4 * t)) * np.cos(t)])

    A = np.array([[-2, -1],
                  [-3, -2]])

    def j_phi(x):
        return (A * BUTTERFLY_MU + BUTTERFLY_GAMMA *
                np.array([[x[1], x[0]],
                          [-1, 3 * x[1] ** 2]]))

    return j_phi(y - f_t)


def butterfly_sol(t):
    return np.array([(np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                      2 * np.cos(4 * t)) * np.sin(t),
                     (np.exp(np.cos(t)) - np.sin(t / 12) ** 5 -
                      2 * np.cos(4 * t)) * np.cos(t)])


# ----------------------------------------------------------------------------


PROBLEMS = {
    # Non-stiff problems.
    'Lotka-Volterra': Problem(
        lotka_rhs,
        lotka_jac,
        default_t=(0.0, 2.0),
        default_y0=(1, 1),
        description="This non-stiff problem originates from biology.  The "
                    "model describes the dynamic of the amount of a special "
                    "kind of predator (e.g. foxes) and a special kind of prey"
                    " (e.g. rabbits) at a certain time `t`."),
    'Pleiades': Problem(
        pleiades_rhs,
        default_t=(0, 3),
        default_y0=[*PLEIADES_X_0, *PLEIADES_Y_0, *PLEIADES_XD_0,
                    *PLEIADES_YD_0],
        description="This non-stiff problem has its origin in celestial "
                    "mechanics. It describes the planar movement of seven "
                    "stars in space, where star `i` has coordinates (`x_i`, "
                    "`y_i`) and mass `m`."),

    # Stiff problems.
    'HIRES': Problem(
        hires_rhs,
        hires_jac,
        default_t=(0, 321.8122),
        default_y0=(1, 0, 0, 0, 0, 0, 0, 0.0057),
        description="The name HIRES is short for High Irradiance RESponse.  "
                    "This stiff problem originates from plant physiology, "
                    "modeling the involvement of light in morphogenesis. "
                    "More precisely, it models the high irradiance responses "
                    "of photomorphogenesis on the basis of phytochrome, "
                    "by means of a chemical reaction involving eight "
                    "reactants. This problem consists of 8 non-linear ODEs."),

    'Pollution': Problem(
        pollution_rhs,
        pollution_jac,
        default_t=(0, 60),
        default_y0=[0, 0.2, 0, 0.04, 0, 0, 0.1, 0.3, 0.01, 0, 0, 0, 0, 0, 0,
                    0, 0.007, 0, 0, 0],
        description="This stiff problem is connected to the air pollution "
                    "model developed at The Dutch National Institute of "
                    "Public Health and Environmental Protection (RIVM). More "
                    "precisely it models the chemical reaction part of this "
                    "air pollution model, and consists of 25 chemical "
                    "reactions and 20 reacting compounds. "),

    # Both stiff and non-stiff problems.
    'Butterfly': Problem(
        butterfly_rhs,
        butterfly_jac,
        butterfly_sol,
        default_t=(0, 10),
        default_y0=butterfly_sol(0),
        description="This eqution is not based on a physical model, but is "
                    "constructed to have an analytical solution. The phase "
                    "plot of this equation creates a so called butterfly "
                    "curve."
    )
}
