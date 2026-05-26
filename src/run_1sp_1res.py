"""
Driver: 1 evolving species on 1 chemostat resource under warming.

Question addressed: can the focal species track a warming environment
through evolution of T_opt (subject to its heritable variance and to
resource availability)?

Run:
    python run_1sp_1res.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_1sp_1R, constant_T, warming_ramp
from figdir import fig_path


# Default parameter set — Kremer et al. 2025 Eq. (2) form
# Required: d2 > b2 > 0 for a valid left-skewed TPC.
DEFAULT_PARAMS = dict(
    # Kremer TPC parameters
    b1=0.5,              # birth-rate normalisation (1/day)
    b2=0.05,             # birth-rate temperature coefficient (1/°C)
    d0=0.05,             # temperature-independent death (1/day)
    d2=0.12,             # temperature-dependent death coefficient (1/°C)
    T_opt_ref=20.0,      # initial trait value (°C)
    # Resource (chemostat)
    D=0.10,              # dilution / turnover rate
    R_in=10.0,           # inflow concentration
    K=1.0,               # half-saturation
    c=1.0,               # resource conversion cost per birth
    # Quantitative genetics
    h2_VP=0.1,           # additive genetic variance scaling
)


def run(T_of_t, params=None, y0=None, t_end=1000.0, n_points=4000):
    p = {**DEFAULT_PARAMS, **(params or {})}
    if y0 is None:
        y0 = [p['R_in'] * 0.5, 1.0, p['T_opt_ref']]
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs_1sp_1R, (0.0, t_end), y0,
                    args=(T_of_t, p), t_eval=t_eval,
                    method='LSODA', rtol=1e-8, atol=1e-10)
    return sol, p


def plot_trajectory(sol, T_of_t, title, save_path):
    T_traj = T_of_t(sol.t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(sol.t, T_traj, 'k-', label='environment T(t)')
    axes[0].plot(sol.t, sol.y[2], 'C3-', label=r'trait $\bar{T}_{opt}$')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend(loc='best')
    axes[0].set_title(title)

    axes[1].plot(sol.t, sol.y[1], 'C0-')
    axes[1].set_ylabel('Population N')
    axes[1].set_yscale('log')

    axes[2].plot(sol.t, sol.y[0], 'C2-')
    axes[2].set_ylabel('Resource R')
    axes[2].set_xlabel('time')

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


if __name__ == '__main__':
    # (a) Constant environment: trait should equilibrate at T_env
    T_const = constant_T(20.0)
    sol_c, _ = run(T_const)
    plot_trajectory(sol_c, T_const,
                    'Model 1: constant T = 20°C',
                    '1sp_1R_constant.png')

    # (b) Slow warming: trait tracks environment with a small lag
    T_slow = warming_ramp(T0=20.0, warming_rate=0.01, T_final=30.0)
    sol_s, _ = run(T_slow, t_end=2000.0, n_points=3000)
    plot_trajectory(sol_s, T_slow,
                    'Model 1: slow warming 0.01 °C/day',
                    '1sp_1R_slow_warming.png')

    # (c) Fast warming: trait lags further; population may crash
    T_fast = warming_ramp(T0=20.0, warming_rate=0.05, T_final=30.0)
    sol_f, _ = run(T_fast, t_end=2000.0, n_points=3000)
    plot_trajectory(sol_f, T_fast,
                    'Model 1: fast warming 0.05 °C/day',
                    '1sp_1R_fast_warming.png')

    # (d) Critical-rate scan: vary warming rate, record final N and lag
    rates = np.linspace(0.0, 0.08, 17)
    final_N, final_lag = [], []
    for r in rates:
        T_of_t = warming_ramp(T0=20.0, warming_rate=r, T_final=35.0)
        sol, _ = run(T_of_t, t_end=2000.0, n_points=1500)
        final_N.append(sol.y[1, -1])
        final_lag.append(T_of_t(sol.t[-1]) - sol.y[2, -1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(rates, final_N, 'o-')
    ax1.set_xlabel('warming rate (°C/day)')
    ax1.set_ylabel('final population N')
    ax1.set_yscale('log')
    ax1.set_title('Evolutionary rescue boundary')
    ax1.grid(True, alpha=0.3)
    ax2.plot(rates, final_lag, 's-', color='C3')
    ax2.set_xlabel('warming rate (°C/day)')
    ax2.set_ylabel(r'thermal lag  $T(t) - \bar{T}_{opt}(t)$')
    ax2.set_title('Adaptation lag at end of ramp')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path('1sp_1R_rate_scan.png'), dpi=150)
    plt.close(fig)
    print('wrote 1sp_1R_rate_scan.png')
