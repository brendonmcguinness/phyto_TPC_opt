"""
Driver: focal (evolving) + non-evolving competitor on 1 shared resource.

Explores questions 2–4 from the framework:
    - How does a warm-adapted competitor restrict adaptive capacity?
    - Competitor at novel T vs. intermediate T.
    - Critical distance between competitor T_opt and novel T.

Run:
    python run_2sp_1res.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_2sp_1R, warming_ramp
from figdir import fig_path


# Focal-species parameter set — Kremer Eq. (2) form (same as Model 1)
FOCAL = dict(
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    D=0.10, R_in=10.0, K=1.0, c=1.0,
    h2_VP=0.1,
)

# Template competitor — the caller sets T_opt (the only thing we scan over)
COMPETITOR_BASE = dict(
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    K=1.0, c=1.0,
)


def run(T_of_t, T_opt_competitor, y0=None,
        t_end=3000.0, n_points=6000, pF=None, pC=None):
    pF = {**FOCAL, **(pF or {})}
    pC = {**COMPETITOR_BASE, 'T_opt': T_opt_competitor, **(pC or {})}
    if y0 is None:
        y0 = [pF['R_in'] * 0.5,  # R
              1.0,                # N1 focal
              1.0,                # N2 competitor
              pF['T_opt_ref']]    # focal trait starts at its historical optimum
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs_2sp_1R, (0.0, t_end), y0,
                    args=(T_of_t, pF, pC), t_eval=t_eval,
                    method='LSODA', rtol=1e-8, atol=1e-10)
    return sol, pF, pC


def plot_trajectory(sol, T_of_t, pC, title, save_path):
    T_traj = T_of_t(sol.t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(sol.t, T_traj, 'k-', label='T(t)')
    axes[0].plot(sol.t, sol.y[3], 'C0-', label=r'focal $\bar{T}_{opt}$')
    axes[0].axhline(pC['T_opt'], color='C3', ls='--',
                    label=r'competitor $T_{opt}$')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend(loc='best')
    axes[0].set_title(title)

    axes[1].plot(sol.t, sol.y[1], 'C0-', label='focal N1')
    axes[1].plot(sol.t, sol.y[2], 'C3-', label='competitor N2')
    axes[1].set_ylabel('Population')
    axes[1].set_yscale('log')
    axes[1].legend()

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
    # Warming scenario: 20 → 30 °C over the run.
    warming = warming_ramp(T0=20.0, warming_rate=0.01, T_final=30.0)

    # (a) Competitor at the novel (warm) endpoint of the ramp
    sol, _, pC = run(warming, T_opt_competitor=30.0)
    plot_trajectory(sol, warming, pC,
                    'Competitor preadapted to novel T (T_opt=30°C)',
                    '2sp_1R_competitor_novel.png')

    # (b) Competitor at an intermediate T (fitness-valley scenario)
    sol, _, pC = run(warming, T_opt_competitor=22.0)
    plot_trajectory(sol, warming, pC,
                    'Competitor intermediate (T_opt=22°C)',
                    '2sp_1R_competitor_intermediate.png')

    # (c) Competitor far past the novel T — focal may find a cool refuge
    sol, _, pC = run(warming, T_opt_competitor=35.0)
    plot_trajectory(sol, warming, pC,
                    'Competitor far past novel (T_opt=35°C)',
                    '2sp_1R_competitor_far.png')

    # (d) Scan over competitor T_opt — map the "persistence corridor"
    competitor_T_opts = np.linspace(18.0, 35.0, 25)
    final_N1, final_trait, final_lag = [], [], []
    for Topt_c in competitor_T_opts:
        sol, _, _ = run(warming, T_opt_competitor=Topt_c,
                        t_end=1500.0, n_points=1500)
        final_N1.append(sol.y[1, -1])
        final_trait.append(sol.y[3, -1])
        final_lag.append(warming(sol.t[-1]) - sol.y[3, -1])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(competitor_T_opts, final_N1, 'o-')
    axes[0].set_xlabel(r'competitor $T_{opt}$ (°C)')
    axes[0].set_ylabel('focal N1 at end of ramp')
    axes[0].set_yscale('log')
    axes[0].axvline(30.0, color='k', ls=':', label='novel T')
    axes[0].legend()
    axes[0].set_title('Persistence')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(competitor_T_opts, final_trait, 's-', color='C0')
    axes[1].axhline(30.0, color='k', ls=':', label='novel T')
    axes[1].plot(competitor_T_opts, competitor_T_opts, 'r--',
                 label=r'$y=x$ (competitor)')
    axes[1].set_xlabel(r'competitor $T_{opt}$ (°C)')
    axes[1].set_ylabel(r'focal $\bar{T}_{opt}$ at end')
    axes[1].set_title('Character displacement?')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(competitor_T_opts, final_lag, '^-', color='C3')
    axes[2].set_xlabel(r'competitor $T_{opt}$ (°C)')
    axes[2].set_ylabel('adaptation lag')
    axes[2].set_title('Lag  $T(t)-\\bar{T}_{opt}$')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path('2sp_1R_competitor_scan.png'), dpi=150)
    plt.close(fig)
    print('wrote 2sp_1R_competitor_scan.png')
