"""
Driver: focal (evolving) + non-evolving competitor on 2 substitutable
resources.

This is the "niche partitioning as a coexistence mechanism" scenario
from question 5 of the framework. The depletable resources (R1, R2) are
contrasted with the temperature axis:

    - Resources are depletable, dynamic, and subject to consumer feedback.
    - Temperature is a non-depletable environmental driver set by T(t).

By giving the focal and competitor different resource-preference weights
we can test whether resource-axis partitioning allows the focal species
to persist and evolve its T_opt under warming, even when the competitor
would otherwise exclude it on the thermal axis alone.

Run:
    python run_2sp_2res.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_2sp_2R, warming_ramp
from figdir import fig_path


FOCAL = dict(
    # Kremer Eq. (2) TPC parameters
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    # chemostat
    D=0.10, R1_in=10.0, R2_in=10.0, K1=1.0, K2=1.0, c=1.0,
    # focal prefers R1
    w1=1.0, w2=0.4,
    h2_VP=0.05,
)

COMPETITOR_BASE = dict(
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    K1=1.0, K2=1.0, c=1.0,
    # competitor prefers R2
    w1=0.4, w2=1.0,
)


def run(T_of_t, T_opt_competitor, pF=None, pC=None,
        y0=None, t_end=1500.0, n_points=3000):
    pF = {**FOCAL, **(pF or {})}
    pC = {**COMPETITOR_BASE, 'T_opt': T_opt_competitor, **(pC or {})}
    if y0 is None:
        y0 = [0.5 * pF['R1_in'], 0.5 * pF['R2_in'],
              1.0, 1.0, pF['T_opt_ref']]
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs_2sp_2R, (0.0, t_end), y0,
                    args=(T_of_t, pF, pC), t_eval=t_eval,
                    method='LSODA', rtol=1e-8, atol=1e-10)
    return sol, pF, pC


def plot_trajectory(sol, T_of_t, pC, title, save_path):
    T_traj = T_of_t(sol.t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(sol.t, T_traj, 'k-', label='T(t)')
    axes[0].plot(sol.t, sol.y[4], 'C0-', label=r'focal $\bar{T}_{opt}$')
    axes[0].axhline(pC['T_opt'], color='C3', ls='--',
                    label=r'competitor $T_{opt}$')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].set_title(title)

    axes[1].plot(sol.t, sol.y[2], 'C0-', label='focal N1')
    axes[1].plot(sol.t, sol.y[3], 'C3-', label='competitor N2')
    axes[1].set_ylabel('Population')
    axes[1].set_yscale('log')
    axes[1].legend()

    axes[2].plot(sol.t, sol.y[0], 'C2-', label='R1 (focal-preferred)')
    axes[2].plot(sol.t, sol.y[1], 'C4-', label='R2 (competitor-preferred)')
    axes[2].set_ylabel('Resource')
    axes[2].set_xlabel('time')
    axes[2].legend()

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


if __name__ == '__main__':
    warming = warming_ramp(T0=20.0, warming_rate=0.01, T_final=30.0)

    # (a) Strong resource partitioning — should enable coexistence
    sol, _, pC = run(warming, T_opt_competitor=30.0)
    plot_trajectory(sol, warming, pC,
                    'Two-resource coexistence (preadapted competitor)',
                    '2sp_2R_partitioned.png')

    # (b) Weak partitioning (swap weights toward symmetric)
    pF_sym = {'w1': 0.8, 'w2': 0.6}
    pC_sym = {'w1': 0.6, 'w2': 0.8}
    sol, _, pC = run(warming, T_opt_competitor=30.0,
                     pF=pF_sym, pC=pC_sym)
    plot_trajectory(sol, warming, pC,
                    'Two-resource weak partitioning',
                    '2sp_2R_symmetric.png')

    # (c) Scan: how does resource-preference asymmetry mediate
    #     the focal's ability to evolve and persist under warming?
    asymmetries = np.linspace(0.0, 0.9, 10)  # 0 = identical, 0.9 = strong
    final_N1, final_trait = [], []
    for a in asymmetries:
        pF_a = {'w1': 0.5 + a / 2, 'w2': 0.5 - a / 2}
        pC_a = {'w1': 0.5 - a / 2, 'w2': 0.5 + a / 2}
        sol, _, _ = run(warming, T_opt_competitor=30.0,
                        pF=pF_a, pC=pC_a,
                        t_end=1500.0, n_points=1500)
        final_N1.append(sol.y[2, -1])
        final_trait.append(sol.y[4, -1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(asymmetries, final_N1, 'o-')
    ax1.set_xlabel('resource-preference asymmetry')
    ax1.set_ylabel('focal N1 at end of ramp')
    ax1.set_yscale('log')
    ax1.set_title('Niche partitioning buys persistence')
    ax1.grid(True, alpha=0.3)
    ax2.plot(asymmetries, final_trait, 's-', color='C0')
    ax2.axhline(30.0, color='k', ls=':', label='novel T')
    ax2.set_xlabel('resource-preference asymmetry')
    ax2.set_ylabel(r'focal $\bar{T}_{opt}$ at end')
    ax2.set_title('and frees the trait to track T')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path('2sp_2R_asymmetry_scan.png'), dpi=150)
    plt.close(fig)
    print('wrote 2sp_2R_asymmetry_scan.png')
