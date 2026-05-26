"""
Eco-evolutionary ODE systems for thermal adaptation under warming,
built on the Kremer et al. 2025 (Eq. 2) growth-rate model.

Three nested models, all sharing the Kremer TPC machinery in
`thermal_performance.py`:

    1. `rhs_1sp_1R`   — one evolving species on one chemostat resource.
    2. `rhs_2sp_1R`   — focal (evolving) species + non-evolving competitor,
                        one shared chemostat resource.
    3. `rhs_2sp_2R`   — focal (evolving) + non-evolving competitor, two
                        resources (substitutable by default, Liebig via
                        a `resource_mode` flag).

Temperature is an environmental driver T(t) supplied as a callable
(constant, ramp, or ramp + seasonal). Trait evolution uses the
breeder's-equation form
    dT_opt_star / dt = h^2 * V_P * dmu/dT_opt_star .

Because Kremer Eq. (2) places the trait inside the death term, the
selection gradient is R-independent. This differs from a Gaussian-birth
model and has consequences you can see in the simulations.
"""

from __future__ import annotations

import numpy as np

from thermal_performance import (
    birth_rate,
    death_rate,
    mu_kremer,
    dmu_dTopt_kremer,
    monod,
    substitutable_R_factor,
    liebig_R_factor,
)


# ---------------------------------------------------------------------------
# Temperature drivers
# ---------------------------------------------------------------------------

def constant_T(T0):
    """T(t) = T0 (broadcasts to input shape so array-in gives array-out)."""
    def T_of_t(t):
        return np.full_like(np.asarray(t, dtype=float), T0)
    return T_of_t


def warming_ramp(T0, warming_rate, T_final=None):
    """Linear warming T(t) = T0 + warming_rate * t, optionally capped."""
    def T_of_t(t):
        t_arr = np.asarray(t, dtype=float)
        T = T0 + warming_rate * t_arr
        if T_final is not None:
            T = np.minimum(T, T_final)
        return T
    return T_of_t


def seasonal_warming(T0, warming_rate, amplitude, period, T_final=None):
    """Warming ramp with a superimposed sinusoidal seasonal cycle."""
    def T_of_t(t):
        t_arr = np.asarray(t, dtype=float)
        base = T0 + warming_rate * t_arr
        if T_final is not None:
            base = np.minimum(base, T_final)
        return base + amplitude * np.sin(2.0 * np.pi * t_arr / period)
    return T_of_t


# ---------------------------------------------------------------------------
# Model 1: 1 species, 1 resource
# ---------------------------------------------------------------------------

def rhs_1sp_1R(t, y, T_of_t, params):
    """
    State y = [R, N, T_opt_star].

    Resource flux uses the Kremer per-capita birth rate times the Monod
    factor — this is what actually gets consumed.

        dR/dt        = D (R_in - R) - c * b(T) * g(R) * N
        dN/dt        = mu(T, R, T_opt_star) * N
        dT_opt_star/dt = h^2 V_P * dmu/dT_opt_star
    """
    R, N, T_opt = y
    T = T_of_t(t)

    b = birth_rate(T, params)
    gR = monod(R, params['K'])
    mu = mu_kremer(T, T_opt, gR, params)

    dR = params['D'] * (params['R_in'] - R) - params['c'] * b * gR * N
    dN = mu * N
    dTopt = params['h2_VP'] * dmu_dTopt_kremer(T, T_opt, params)

    return [dR, dN, dTopt]


# ---------------------------------------------------------------------------
# Model 2: 2 species, 1 shared resource
# ---------------------------------------------------------------------------

def rhs_2sp_1R(t, y, T_of_t, pF, pC):
    """
    State y = [R, N1, N2, T_opt_star_1].
    Focal species evolves; competitor fixed at pC['T_opt'].
    """
    R, N1, N2, T_opt1 = y
    T = T_of_t(t)

    gR = monod(R, pF['K'])

    bF = birth_rate(T, pF)
    bC = birth_rate(T, pC)

    mu1 = mu_kremer(T, T_opt1, gR, pF)
    mu2 = mu_kremer(T, pC['T_opt'], gR, pC)

    dR = (pF['D'] * (pF['R_in'] - R)
          - pF['c'] * bF * gR * N1
          - pC['c'] * bC * gR * N2)
    dN1 = mu1 * N1
    dN2 = mu2 * N2
    dTopt1 = pF['h2_VP'] * dmu_dTopt_kremer(T, T_opt1, pF)

    return [dR, dN1, dN2, dTopt1]


# ---------------------------------------------------------------------------
# Model 3: 2 species, 2 resources
# ---------------------------------------------------------------------------

def rhs_2sp_2R(t, y, T_of_t, pF, pC, resource_mode='substitutable'):
    """
    State y = [R1, R2, N1, N2, T_opt_star_1].

    `resource_mode`:
        'substitutable' (default): R-factor = w1*g1(R1) + w2*g2(R2)
                                   (niche-partitioning via preferences)
        'liebig'                 : R-factor = min(g1(R1), g2(R2))
                                   (essential resources, Kremer Eq. 3)

    In Liebig mode the w1, w2 are unused for growth but are still
    used for the resource-consumption bookkeeping (they set the per-
    capita uptake ratio). To be explicit about Kremer's light+nutrient
    case where each species draws both resources symmetrically, set
    w1 = w2 = 1 in Liebig mode.
    """
    R1, R2, N1, N2, T_opt1 = y
    T = T_of_t(t)

    g1F = monod(R1, pF['K1']);   g2F = monod(R2, pF['K2'])
    g1C = monod(R1, pC['K1']);   g2C = monod(R2, pC['K2'])

    if resource_mode == 'liebig':
        RfF = np.minimum(g1F, g2F)
        RfC = np.minimum(g1C, g2C)
    else:
        RfF = pF['w1'] * g1F + pF['w2'] * g2F
        RfC = pC['w1'] * g1C + pC['w2'] * g2C

    bF = birth_rate(T, pF)
    bC = birth_rate(T, pC)
    mu1 = mu_kremer(T, T_opt1,        RfF, pF)
    mu2 = mu_kremer(T, pC['T_opt'],   RfC, pC)

    # Per-capita consumption of each resource by each species.
    # Birth rate scales overall consumption; preference weights and
    # Monod factors split it between R1 and R2.
    U1_1 = bF * pF['w1'] * g1F
    U1_2 = bF * pF['w2'] * g2F
    U2_1 = bC * pC['w1'] * g1C
    U2_2 = bC * pC['w2'] * g2C

    dR1 = (pF['D'] * (pF['R1_in'] - R1)
           - pF['c'] * U1_1 * N1
           - pC['c'] * U2_1 * N2)
    dR2 = (pF['D'] * (pF['R2_in'] - R2)
           - pF['c'] * U1_2 * N1
           - pC['c'] * U2_2 * N2)

    dN1 = mu1 * N1
    dN2 = mu2 * N2
    dTopt1 = pF['h2_VP'] * dmu_dTopt_kremer(T, T_opt1, pF)

    return [dR1, dR2, dN1, dN2, dTopt1]
