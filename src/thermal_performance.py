"""
Thermal performance curves (TPCs) — Kremer et al. 2025, Equation (2).

Population-level growth rate for an ectotherm as a function of temperature
and resource availability:

    mu(T, R) = b1 * exp(b2 * T) * g(R)
               - ( d0 + (b1*b2/d2) * exp( (b2 - d2) * T_opt_star ) * exp(d2 * T) )

where
    b1, b2   — birth-rate normalisation and temperature coefficient
    d0, d2   — temperature-independent and temperature-dependent death
    T_opt_star — the species' *idealised* optimum temperature (g = 1).
    g(R)     — resource limitation factor in [0, 1]; for two resources
               we provide both Liebig (min) and substitutable-weighted-sum
               forms.

The trait that evolves is T_opt_star. It enters the model only through
the temperature-dependent death term (via the reparametrisation
d1 = (b1*b2/d2) * exp((b2-d2)*T_opt_star) that makes T_opt_star appear
explicitly; see Kremer et al. 2025 derivation of Eq. 2 from Eq. 1).

A key structural consequence of Kremer Eq. (2): because the trait enters
only the death term, the selection gradient dmu/dT_opt_star is
INDEPENDENT of R. Resource depletion does *not* throttle evolution of
T_opt_star in this formulation (unlike the Gaussian-birth model we used
previously). The Kremer "realised-Topt shift" effect is still present:
the peak of mu(T) at finite R is T_opt_star + ln(g(R))/(d2-b2), which
is lower than T_opt_star when R is limiting — but that is a property of
the *growth curve*, not of evolution on T_opt_star itself.

The required condition for a valid left-skewed TPC is d2 > b2 > 0.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Kremer TPC primitives
# ---------------------------------------------------------------------------

def _d1_from_Topt(T_opt_star, params):
    """
    Reparametrised death-rate coefficient. Derived from Eq. (1) →
    Eq. (2) by requiring that the idealised (g=1) peak of mu(T) lie
    at T = T_opt_star.
        d1 = (b1 * b2 / d2) * exp( (b2 - d2) * T_opt_star )
    """
    b1, b2, d2 = params['b1'], params['b2'], params['d2']
    return (b1 * b2 / d2) * np.exp((b2 - d2) * T_opt_star)


def birth_rate(T, params):
    """
    Temperature-dependent birth rate (pre-resource-limitation).

        b(T) = b1 * exp(b2 * T)

    Notice that in Kremer Eq. (2) the birth rate does NOT depend on
    the trait T_opt_star. The species' thermal optimum is encoded in
    the death term.
    """
    return params['b1'] * np.exp(params['b2'] * T)


def death_rate(T, T_opt_star, params):
    """
    Kremer Eq. (2) death rate:
        d(T, T_opt_star) = d0 + d1(T_opt_star) * exp(d2 * T)
    """
    d0, d2 = params['d0'], params['d2']
    return d0 + _d1_from_Topt(T_opt_star, params) * np.exp(d2 * T)


def mu_kremer(T, T_opt_star, R_factor, params):
    """
    Full Kremer Eq. (2) growth rate.

        mu = b(T) * R_factor  -  d(T, T_opt_star)

    `R_factor` is g(R) for one resource, or the two-resource combiner
    of your choice (substitutable-weighted-sum or Liebig min).
    """
    return birth_rate(T, params) * R_factor - death_rate(T, T_opt_star, params)


def dmu_dTopt_kremer(T, T_opt_star, params):
    """
    Analytic selection gradient d mu / d T_opt_star.

    Since birth rate is independent of T_opt_star, this equals the
    negative derivative of the death term:

        dmu/dT_opt = -(b1*b2/d2) * (b2-d2) * exp((b2-d2)*T_opt) * exp(d2*T)
                   = (b1*b2/d2) * (d2-b2) * exp((b2-d2)*T_opt) * exp(d2*T)

    Positive because d2 > b2. R-independent.
    """
    b1, b2, d2 = params['b1'], params['b2'], params['d2']
    return ((b1 * b2 / d2) * (d2 - b2)
            * np.exp((b2 - d2) * T_opt_star) * np.exp(d2 * T))


# ---------------------------------------------------------------------------
# Resource limitation kernels
# ---------------------------------------------------------------------------

def monod(R, K):
    """Monod resource limitation, g(R) = R/(K+R)."""
    return R / (K + R)


def substitutable_R_factor(R1, R2, K1, K2, w1, w2):
    """
    Substitutable two-resource uptake: weighted sum of Monod terms.
    Preference weights (w1, w2) encode the niche-axis of resource use.
    """
    return w1 * monod(R1, K1) + w2 * monod(R2, K2)


def liebig_R_factor(R1, R2, K1, K2):
    """
    Liebig's law of the minimum (essential resources) — the form used
    in Kremer Eq. (3) for nitrogen + light.
    """
    return np.minimum(monod(R1, K1), monod(R2, K2))


# ---------------------------------------------------------------------------
# Convenience wrappers matching the old API
# ---------------------------------------------------------------------------

def fitness_1R(T, T_opt_star, R, params):
    """Kremer per-capita fitness on one resource."""
    return mu_kremer(T, T_opt_star, monod(R, params['K']), params)


def selection_gradient_1R(T, T_opt_star, R, params):
    """
    Selection gradient for the 1-resource model.

    Note: in the Kremer formulation this is R-INDEPENDENT (we pass R
    in for API compatibility, but it is unused). Contrast with the
    Gaussian-birth model where this would be g(R) * db/dTopt.
    """
    return dmu_dTopt_kremer(T, T_opt_star, params)


def selection_gradient_2R(T, T_opt_star, R1, R2, params):
    """Selection gradient in the 2-resource model — also R-independent."""
    return dmu_dTopt_kremer(T, T_opt_star, params)


def realised_TPC(T_grid, T_opt_star, R, params):
    """
    Realised TPC across temperature at a given standing R.
    The peak of this curve sits at
        T_realised = T_opt_star + ln(g(R)) / (d2 - b2),
    which is BELOW T_opt_star when g(R) < 1 — the Kremer
    realised-Topt shift.
    """
    return mu_kremer(T_grid, T_opt_star, monod(R, params['K']), params)


def realised_TPC_2R(T_grid, T_opt_star, R1, R2, params,
                    mode='substitutable'):
    """Same, for two resources. `mode` is 'substitutable' or 'liebig'."""
    if mode == 'liebig':
        Rf = liebig_R_factor(R1, R2, params['K1'], params['K2'])
    else:
        Rf = substitutable_R_factor(R1, R2, params['K1'], params['K2'],
                                    params['w1'], params['w2'])
    return mu_kremer(T_grid, T_opt_star, Rf, params)
