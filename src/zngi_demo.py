"""
ZNGI demo — three figures connecting the eco-evo results to classic
contemporary-niche / Tilman theory.

Figure 1: ZNGI snapshots at three temperatures (20, 25, 30 °C) for the
          initial trait (T_opt,1 = 20). Shows how both species'
          isoclines slide as the environment warms — and when the
          focal's ZNGI leaves the positive quadrant.

Figure 2: Same, but at T = 30 (novel) for three values of the focal
          trait (T_opt,1 = 20, 25, 30). Shows how evolution *pulls the
          focal ZNGI back inward* and reopens the crossing.

Figure 3: (A) rho_i(T) = d_i / b_i across temperature for both species,
          with the w1+w2 ceiling. (B) trajectory of the focal ZNGI
          across the joint (T, T_opt) path from a tunneling simulation.

Figure 4: Resource-preference asymmetry scan. Draws the ZNGIs for
          a = 0, 0.3, 0.6, 0.9 at T = 30 with the evolved trait. Shows
          the birth of a ZNGI crossing as a goes up — the geometric
          root of the persistence threshold at a ≈ 0.6 from the 2R
          tunneling scan.

Run:
    python zngi_demo.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_2sp_2R, warming_ramp
from zngi_plot import (
    plot_zngi_panel,
    plot_trait_trajectory_on_zngi,
    plot_rho_vs_T,
)

from run_2sp_2R_tunneling import FOCAL_BASE, COMPETITOR_BASE
from figdir import fig_path


# ---------------------------------------------------------------------------
# Shared parameter set — use a moderate partitioning so crossings exist
# ---------------------------------------------------------------------------

A_DEFAULT = 0.7
WARMING_RATE = 0.010
T0, T_FINAL = 20.0, 30.0


def make_params(asymmetry=A_DEFAULT, R_in=2.0, K=3.0):
    """
    Build focal/competitor parameter dicts for ZNGI figures.

    `R_in` sets the chemostat supply for *both* resources (R1_in=R2_in),
    which is also the supply-point marker on the (R1, R2) plane. We use
    a low supply here (default 2.0) so the ZNGIs and the supply point
    are all visible inside the same zoomed-in view.

    `K` is the half-saturation for both resources. Raising K from 1.0
    to ~3.0 shifts the typical crossing into the unsaturated regime of
    the Monod curve, where g_i = R_i/(K+R_i) is more responsive to
    R_i. This makes the impact-vector angle visibly track changes in
    the crossing location as the focal trait evolves.
    """
    pF = dict(FOCAL_BASE)
    pF['w1'] = 0.5 + asymmetry / 2.0
    pF['w2'] = 0.5 - asymmetry / 2.0
    pF['R1_in'] = R_in
    pF['R2_in'] = R_in
    pF['K1'] = K; pF['K2'] = K
    pC = dict(COMPETITOR_BASE)
    pC['w1'] = 0.5 - asymmetry / 2.0
    pC['w2'] = 0.5 + asymmetry / 2.0
    pC['K1'] = K; pC['K2'] = K
    return pF, pC


# ---------------------------------------------------------------------------
# FIGURE 1 — warming with fixed (non-evolving) focal trait
# ---------------------------------------------------------------------------

def figure1_temperature_sweep(save_path='zngi_fig1_warming_sweep.png'):
    pF, pC = make_params()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, T in zip(axes, [20.0, 25.0, 30.0]):
        plot_zngi_panel(ax, pF, pC, T=T, T_opt_F=pF['T_opt_ref'],
                        R_max=4.0,
                        title=f'T = {T:.0f}°C  (focal trait frozen at 20°C)')
    fig.suptitle(
        'Figure 1.  Without evolution: the focal ZNGI swells and '
        'eventually leaves the positive quadrant as T climbs past its '
        r'$T_{opt}$.',
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------
# FIGURE 2 — at novel T, evolution pulls the focal ZNGI back inward
# ---------------------------------------------------------------------------

def figure2_evolution_at_novel_T(save_path='zngi_fig2_evolution_rescue.png'):
    pF, pC = make_params()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, Topt in zip(axes, [20.0, 25.0, 30.0]):
        plot_zngi_panel(ax, pF, pC, T=30.0, T_opt_F=Topt,
                        R_max=4.0,
                        title=f'T = 30°C,   focal trait = {Topt:.0f}°C')
    fig.suptitle(
        'At novel T = 30°C, as the trait evolves from 20 → '
        '25 → 30, relative uptake rates are ((0.85, 0.15), (0.15, 0.85)) i.e. a=0.7',
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------
# FIGURE 3 — rho(T) + joint warming/evolution trajectory
# ---------------------------------------------------------------------------

def figure3_rho_and_trajectory(save_path='zngi_fig3_rho_and_trajectory.png'):
    pF, pC = make_params(asymmetry=0.7)

    # run a quick 2R simulation in the persisting regime to get a real
    # (T(t), T_opt(t)) trajectory
    T_of_t = warming_ramp(T0=T0, warming_rate=WARMING_RATE, T_final=T_FINAL)

    def rhs(t, y):
        return rhs_2sp_2R(t, y, T_of_t, pF, pC)

    y0 = [0.5 * pF['R1_in'], 0.5 * pF['R2_in'], 1.0, 1.0, pF['T_opt_ref']]
    sol = solve_ivp(rhs, (0.0, 2500.0), y0,
                    t_eval=np.linspace(0.0, 2500.0, 400),
                    method='LSODA', rtol=1e-9, atol=1e-12)

    # Sample (T, T_opt) pairs across the trajectory
    idx = np.linspace(0, len(sol.t) - 1, 10).astype(int)
    T_trait_pairs = [(float(T_of_t(sol.t[i])), float(sol.y[4, i])) for i in idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel A: rho vs T
    T_grid = np.linspace(15.0, 35.0, 300)
    plot_rho_vs_T(axes[0], pF, pC, T_grid)

    # panel B: focal ZNGI trajectory in (R1, R2) across joint path.
    # Tight zoom so the contracting focal ZNGIs are clearly visible.
    plot_trait_trajectory_on_zngi(axes[1], pF, pC, T_trait_pairs,
                                  R_max=4.0)
    axes[1].set_xlim(0, 2.5)
    axes[1].set_ylim(0, 2.5)

    fig.suptitle(
        'Figure 3.  Left: required g-value rho(T) = d/b. '
        'Right: focal ZNGI motion across the warming+evolution path.',
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------
# FIGURE 4 — asymmetry scan shows the ZNGI crossing being born
# ---------------------------------------------------------------------------

def figure4_asymmetry(save_path='zngi_fig4_asymmetry.png'):
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    for ax, a in zip(axes, [0.0, 0.3, 0.6, 0.9]):
        pF, pC = make_params(asymmetry=a)
        plot_zngi_panel(
            ax, pF, pC, T=30.0, T_opt_F=30.0,
            R_max=4.0,
            title=f'a = {a:.1f}   (T=30°C, focal trait=30°C)'
        )
    fig.suptitle(
        'Figure 4.  Resource-preference asymmetry controls the *angle* '
        'of each ZNGI. Crossings in the positive quadrant appear only '
        'when the angles are sufficiently different — the geometric '
        'basis of the a ≈ 0.6 persistence threshold.',
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    figure1_temperature_sweep()
    figure2_evolution_at_novel_T()
    figure3_rho_and_trajectory()
    figure4_asymmetry()
