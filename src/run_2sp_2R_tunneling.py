"""
Evolutionary tunneling on TWO substitutable resources.
=======================================================

Same thermal scenario as run_2sp_1R_tunneling.py:

    - Warming ramp from T0=20°C to T_final=30°C.
    - Focal (evolving) starts cold-adapted: T_opt,1(0) = 20°C.
    - Competitor is static at intermediate T_opt,2 = 25°C.

The only difference is the resource axis: instead of one shared pool
there are two substitutable resources (R1, R2) with species-specific
preference weights (w1, w2). On a single resource, the focal species
must tunnel through a demographic valley while the competitor drives R
down — a regime we showed earlier can extinguish it.

With two resources, what buffers the focal species? This driver scans
the preference asymmetry. For each asymmetry `a` in [0, 1]:

    focal:      w1 = 0.5 + a/2,  w2 = 0.5 - a/2   (prefers R1)
    competitor: w1 = 0.5 - a/2,  w2 = 0.5 + a/2   (prefers R2)

    - a = 0  —>  identical preferences: effectively the 1R system.
    - a = 1  —>  full resource partitioning: focal uses only R1,
                 competitor uses only R2.

Everything else (TPC, warming rate, h^2 V_P, extinction floor) is held
at the values that gave a clear extinction in the 1R FAILURE scenario,
so any survival here is attributable to the 2nd resource.

Run:
    python run_2sp_2R_tunneling.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_2sp_2R, warming_ramp
from figdir import fig_path


# ---------------------------------------------------------------------------
# Parameters — chosen to be in the 1R FAILURE regime when a = 0
# ---------------------------------------------------------------------------

FOCAL_BASE = dict(
    # Kremer Eq. (2) TPC parameters
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    # chemostat
    D=0.10, R1_in=10.0, R2_in=10.0, K1=1.0, K2=1.0, c=1.0,
    # preferences set per run
    w1=1.0, w2=0.0,
    h2_VP=0.05,        # the small variance that fails in 1R
)

COMPETITOR_BASE = dict(
    # Slightly weaker than the focal at its own optimum, matching the
    # one-resource tunneling setup so asymmetry controls the rescue.
    b1=0.495, b2=0.05,
    d0=0.05, d2=0.12,
    K1=1.0, K2=1.0, c=1.0,
    w1=0.0, w2=1.0,
    T_opt=25.0,
)

WARMING_RATE = 0.010
T0_ENV, T_FINAL_ENV = 20.0, 30.0
T_END = 2500.0

N_FLOOR = 1e-4

Y0_TEMPLATE = lambda pF: [
    0.5 * pF['R1_in'], 0.5 * pF['R2_in'],
    1.0, 1.0, pF['T_opt_ref']
]


# ---------------------------------------------------------------------------
# Solver with extinction floor on focal (state index 2)
# ---------------------------------------------------------------------------

def simulate(asymmetry, h2_VP=None, warming_rate=WARMING_RATE,
             n_points=3000):
    """
    Run the 2sp 2R system with preference asymmetry `a`. Returns
    trajectories plus metadata.
    """
    pF = dict(FOCAL_BASE)
    pF['w1'] = 0.5 + asymmetry / 2.0
    pF['w2'] = 0.5 - asymmetry / 2.0
    if h2_VP is not None:
        pF['h2_VP'] = h2_VP

    pC = dict(COMPETITOR_BASE)
    pC['w1'] = 0.5 - asymmetry / 2.0
    pC['w2'] = 0.5 + asymmetry / 2.0

    T_of_t = warming_ramp(T0=T0_ENV, warming_rate=warming_rate,
                          T_final=T_FINAL_ENV)

    def rhs(t, y):
        return rhs_2sp_2R(t, y, T_of_t, pF, pC)

    def extinction_event(t, y):
        return y[2] - N_FLOOR
    extinction_event.terminal = True
    extinction_event.direction = -1

    t_eval = np.linspace(0.0, T_END, n_points)
    y0 = Y0_TEMPLATE(pF)
    sol = solve_ivp(rhs, (0.0, T_END), y0, t_eval=t_eval,
                    events=extinction_event,
                    method='LSODA', rtol=1e-9, atol=1e-12)

    if sol.status == 1 and len(sol.t_events[0]) > 0:
        t_ext = sol.t_events[0][0]
        y_ext = sol.y_events[0][0].copy()
        y_ext[2] = 0.0

        def rhs_post(t, y):
            dy = rhs_2sp_2R(t, y, T_of_t, pF, pC)
            dy[2] = 0.0   # focal population pinned to 0
            dy[4] = 0.0   # freeze the trait — a mean trait without a
                          # population attached is biologically meaningless,
                          # so we stop evolving it at the extinction event.
            return dy

        t_eval_post = t_eval[t_eval > t_ext]
        sol_post = solve_ivp(rhs_post, (t_ext, T_END), y_ext,
                             t_eval=t_eval_post,
                             method='LSODA', rtol=1e-9, atol=1e-12)
        t_full = np.concatenate([sol.t, sol_post.t])
        y_full = np.concatenate([sol.y, sol_post.y], axis=1)
        extinct = True
    else:
        t_full, y_full = sol.t, sol.y
        extinct = False

    return t_full, y_full, T_of_t, extinct, pF, pC


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_trajectory(t, y, T_of_t, pC, title, save_path):
    """Five-panel look at one run: T, trait, populations, R1, R2."""
    T_env = T_of_t(t)
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(t, T_env, 'k-', label='T(t)')
    axes[0].plot(t, y[4], 'C0-', lw=2, label=r'focal $\bar T_{opt}$')
    axes[0].axhline(pC['T_opt'], color='C3', ls='--',
                    label=r'competitor $T_{opt}$')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, y[2], 'C0-', lw=2, label='focal $N_1$')
    axes[1].plot(t, y[3], 'C3-', lw=2, label='competitor $N_2$')
    axes[1].set_ylabel('population')
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=1e-5)
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, y[0], 'C2-', label='R1 (focal-preferred)')
    axes[2].set_ylabel('R1')
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    axes[3].plot(t, y[1], 'C4-', label='R2 (competitor-preferred)')
    axes[3].set_ylabel('R2')
    axes[3].set_yscale('log')
    axes[3].set_xlabel('time')
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


def plot_asymmetry_grid(results, save_path):
    """Small-multiples panel across the asymmetry scan."""
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.5), sharex='col')

    for col, (a, (t, y, T_of_t, extinct, pF, pC)) in enumerate(results.items()):
        T_env = T_of_t(t)

        ax = axes[0, col]
        ax.plot(t, T_env, 'k-', lw=1)
        ax.plot(t, y[4], 'C0-', lw=2)
        ax.axhline(pC['T_opt'], color='C3', ls='--', lw=1)
        ax.set_title(f'a = {a:.2f}' +
                     ('  ✗ extinct' if extinct else '  ✓ persist'))
        if col == 0:
            ax.set_ylabel('T, trait (°C)')
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(t, y[2], 'C0-', lw=2, label='focal')
        ax.plot(t, y[3], 'C3-', lw=2, label='comp.')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-5)
        if col == 0:
            ax.set_ylabel('N')
            ax.legend(fontsize=7, loc='lower left')
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(t, y[0], 'C2-', lw=1, label='R1')
        ax.plot(t, y[1], 'C4-', lw=1, label='R2')
        ax.set_yscale('log')
        ax.set_xlabel('time')
        if col == 0:
            ax.set_ylabel('R')
            ax.legend(fontsize=7, loc='lower left')
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


def plot_asymmetry_scan(asymmetries, final_N1, final_trait,
                        final_N1_min, extinction_flags, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(asymmetries, final_N1, 'o-', color='C0', label='N1(end)')
    axes[0].plot(asymmetries, final_N1_min, 's--', color='C3',
                 label='N1 minimum')
    axes[0].set_xlabel('preference asymmetry  a')
    axes[0].set_ylabel('focal population')
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=1e-5)
    axes[0].axhline(N_FLOOR, color='gray', ls=':', label='floor')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Valley depth vs asymmetry')
    axes[0].grid(alpha=0.3)

    axes[1].plot(asymmetries, final_trait, '^-', color='C0')
    axes[1].axhline(T_FINAL_ENV, color='k', ls=':', label='novel T')
    axes[1].axhline(COMPETITOR_BASE['T_opt'], color='C3', ls='--',
                    label=r'competitor $T_{opt}$')
    axes[1].set_xlabel('preference asymmetry  a')
    axes[1].set_ylabel(r'focal $\bar T_{opt}$ at end')
    axes[1].set_title('How far the trait evolved')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(asymmetries, extinction_flags, 'o-', color='C3')
    axes[2].set_xlabel('preference asymmetry  a')
    axes[2].set_ylabel('extinct? (1 = yes)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_title('Persistence boundary')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # (1) Two representative trajectories: no partitioning vs strong.
    t0, y0_, Tf0, ext0, pF0, pC0 = simulate(asymmetry=0.0)
    plot_trajectory(t0, y0_, Tf0, pC0,
                    'a = 0  (shared resources — 1R-equivalent)',
                    '2sp_2R_tunneling_a0p00.png')

    t1, y1_, Tf1, ext1, pF1, pC1 = simulate(asymmetry=0.8)
    plot_trajectory(t1, y1_, Tf1, pC1,
                    'a = 0.8  (strong resource partitioning)',
                    '2sp_2R_tunneling_a0p80.png')

    # (2) Small-multiples panel across asymmetry.
    grid_as = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid_results = {}
    for a in grid_as:
        grid_results[a] = simulate(asymmetry=a)
    plot_asymmetry_grid(grid_results, '2sp_2R_tunneling_grid.png')

    # (3) Fine scan summary.
    scan_as = np.linspace(0.0, 1.0, 21)
    final_N1, final_trait, final_N1_min, ext_flags = [], [], [], []
    for a in scan_as:
        t, y, Tf, ext, _, _ = simulate(asymmetry=a, n_points=1500)
        final_N1.append(max(y[2, -1], 1e-20))
        final_trait.append(y[4, -1])
        final_N1_min.append(max(y[2].min(), 1e-20))
        ext_flags.append(1 if ext else 0)
    plot_asymmetry_scan(scan_as, final_N1, final_trait,
                        final_N1_min, ext_flags,
                        '2sp_2R_tunneling_scan.png')

    # (4) Secondary angle: at fixed moderate asymmetry, does a richer
    #     R2 supply (the non-preferred resource of the focal) help?
    a_fixed = 0.4
    R2_ins = np.linspace(2.0, 30.0, 10)
    N1_end, trait_end, ext_R2 = [], [], []
    for R2in in R2_ins:
        pF_over = dict(FOCAL_BASE); pF_over['R2_in'] = R2in
        pF_over['w1'] = 0.5 + a_fixed / 2; pF_over['w2'] = 0.5 - a_fixed / 2
        # push override into module-level FOCAL_BASE via simulate's dict
        saved = FOCAL_BASE['R2_in']
        FOCAL_BASE['R2_in'] = R2in
        t, y, Tf, ext, _, _ = simulate(asymmetry=a_fixed, n_points=1500)
        FOCAL_BASE['R2_in'] = saved
        N1_end.append(max(y[2, -1], 1e-20))
        trait_end.append(y[4, -1])
        ext_R2.append(1 if ext else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(R2_ins, N1_end, 'o-')
    ax1.set_xlabel('R2 supply concentration')
    ax1.set_ylabel('focal N1 at end')
    ax1.set_yscale('log')
    ax1.set_title(f'Secondary resource supply (a = {a_fixed})')
    ax1.grid(alpha=0.3)
    ax2.plot(R2_ins, trait_end, '^-', color='C0')
    ax2.axhline(T_FINAL_ENV, color='k', ls=':', label='novel T')
    ax2.axhline(COMPETITOR_BASE['T_opt'], color='C3', ls='--',
                label=r'competitor $T_{opt}$')
    ax2.set_xlabel('R2 supply concentration')
    ax2.set_ylabel(r'focal $\bar T_{opt}$ at end')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path('2sp_2R_tunneling_R2supply.png'), dpi=150)
    plt.close(fig)
    print('wrote 2sp_2R_tunneling_R2supply.png')

    # Console summary
    print('\nAsymmetry scan:')
    for a, e, N1, Nm, Topt in zip(scan_as, ext_flags, final_N1,
                                  final_N1_min, final_trait):
        mark = 'extinct' if e else 'persist'
        print(f'  a={a:.2f}:  N1_min={Nm:.2e}  N1_end={N1:.2e}  '
              f'T_opt1(end)={Topt:.2f}  {mark}')
