"""
ZNGI (zero-net-growth isocline) visualisation for the thermal-evolution
eco model.

For species i the ZNGI in (R1, R2) space is the locus where per-capita
growth equals zero,

    W_i = b_i(T, T_opt_i) * g_i(R1, R2) - d_i(T) = 0 ,

which with substitutable Monod uptake rearranges to

    w_i1 * R1 / (K1 + R1)  +  w_i2 * R2 / (K2 + R2)  =  rho_i(T),

where rho_i(T) = d_i(T) / b_i(T, T_opt_i) is the species' *required
g-value*. Smaller rho = lower R*, stronger competitor at that
temperature. rho > w1 + w2 ⇒ no isocline in the positive quadrant
(species cannot persist at any (R1, R2)).

This module provides:

    required_g(T, T_opt, params)           — rho_i(T)
    zngi_curve(R1_grid, w1, w2, K1, K2, rho)
                                           — (R1, R2) points on the isocline
    consumption_vector(R1, R2, w1, w2, K1, K2)
                                           — species' per-capita resource draw
    supply_point(params)                   — the chemostat supply (R1_in, R2_in)
    plot_zngi_panel(...)                   — one-shot figure builder

Then zngi_demo.py runs the full warming scenario and produces a figure
showing: focal and competitor ZNGIs at three temperatures, the
focal's ZNGI moving as its trait evolves, the crossings, and the
supply point.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from thermal_performance import birth_rate, death_rate


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def required_g(T, T_opt, params):
    """
    rho_i(T, T_opt) = d_i(T, T_opt) / b_i(T)  under Kremer Eq. (2).

    With Kremer's reparametrisation,
        b(T)            = b1 * exp(b2 * T)
        d(T, T_opt*)    = d0 + (b1*b2/d2) * exp((b2-d2)*T_opt*) * exp(d2*T)
    so
        rho(T, T_opt*)  = [ d0 + (b1*b2/d2)*exp((b2-d2)*T_opt*)*exp(d2*T) ]
                          / [ b1*exp(b2*T) ].

    The trait T_opt* enters rho (unlike the Gaussian-birth version where
    it entered only via the birth width). The extinction ceiling
    rho = w1+w2 still carries the same geometric meaning.
    """
    b = birth_rate(T, params)
    d = death_rate(T, T_opt, params)
    if np.any(np.asarray(b) <= 0):
        return np.inf
    return d / b


def zngi_curve(R1_grid, w1, w2, K1, K2, rho):
    """
    For each R1 on `R1_grid`, solve for the R2 that sits on the isocline
    w1 * g1(R1) + w2 * g2(R2) = rho. Returns the matching R2 array; NaN
    where no positive-R2 solution exists.
    """
    g1 = R1_grid / (K1 + R1_grid)
    rhs_over_w2 = (rho - w1 * g1) / w2

    # g2 must lie in (0, 1). Values outside that band -> no isocline
    # intersects this R1 column.
    valid = (rhs_over_w2 > 0) & (rhs_over_w2 < 1)
    R2 = np.full_like(R1_grid, np.nan, dtype=float)
    R2[valid] = K2 * rhs_over_w2[valid] / (1.0 - rhs_over_w2[valid])
    return R2


def isocline_exists_in_quadrant(w1, w2, rho):
    """True if rho <= w1 + w2, i.e. the sum of Monod terms can reach rho."""
    return rho < (w1 + w2)


def zngi_crossing(pF, pC, T, T_opt_F, n_grid=4000, R_max=40.0):
    """
    Numerically locate the intersection of the focal and competitor
    ZNGIs. Returns (R1*, R2*) or (None, None) if no crossing inside
    [0, R_max]^2.
    """
    rho_F = required_g(T, T_opt_F, pF)
    rho_C = required_g(T, pC['T_opt'], pC)
    R1_grid = np.linspace(1e-4, R_max, n_grid)
    R2_F = zngi_curve(R1_grid, pF['w1'], pF['w2'], pF['K1'], pF['K2'], rho_F)
    R2_C = zngi_curve(R1_grid, pC['w1'], pC['w2'], pC['K1'], pC['K2'], rho_C)
    diff = R2_F - R2_C
    mask = np.isfinite(diff)
    if not np.any(mask):
        return None, None
    diff = diff[mask]
    R1 = R1_grid[mask]
    # find sign change
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    # linear interpolation in R1
    x0, x1 = R1[i], R1[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    R1_star = x0 - y0 * (x1 - x0) / (y1 - y0)
    R2_star = zngi_curve(np.array([R1_star]),
                         pF['w1'], pF['w2'],
                         pF['K1'], pF['K2'], rho_F)[0]
    return R1_star, R2_star


def consumption_vector(R1, R2, w1, w2, K1, K2):
    """
    Per-capita consumption of (R1, R2) by a species with preferences
    (w1, w2). Used as the *direction* of the consumption vector in ZNGI
    diagrams — normalised to unit length for plotting.
    """
    c1 = w1 * R1 / (K1 + R1)
    c2 = w2 * R2 / (K2 + R2)
    norm = np.hypot(c1, c2)
    if norm == 0:
        return 0.0, 0.0
    return c1 / norm, c2 / norm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_zngi(ax, R1_grid, R2, color, label, ls='-', lw=2, alpha=1.0):
    ax.plot(R1_grid, R2, color=color, ls=ls, lw=lw, label=label,
            alpha=alpha)


def plot_zngi_panel(ax, pF, pC, T, T_opt_F,
                    R_max=25.0, show_supply=True,
                    show_consumption=True, title=None,
                    focal_color='C0', comp_color='C3'):
    """Draw the two ZNGIs at temperature T with trait T_opt_F."""
    R1_grid = np.linspace(1e-3, R_max, 2000)

    rho_F = required_g(T, T_opt_F, pF)
    rho_C = required_g(T, pC['T_opt'], pC)

    R2_F = zngi_curve(R1_grid, pF['w1'], pF['w2'],
                      pF['K1'], pF['K2'], rho_F)
    R2_C = zngi_curve(R1_grid, pC['w1'], pC['w2'],
                      pC['K1'], pC['K2'], rho_C)

    persist_F = isocline_exists_in_quadrant(pF['w1'], pF['w2'], rho_F)
    persist_C = isocline_exists_in_quadrant(pC['w1'], pC['w2'], rho_C)

    f_label = (f"focal ZNGI (T_opt={T_opt_F:.1f})"
               if persist_F else f"focal cannot persist at T={T:.1f}")
    c_label = (f"competitor ZNGI (T_opt={pC['T_opt']:.1f})"
               if persist_C else f"competitor cannot persist at T={T:.1f}")

    if persist_F:
        _draw_zngi(ax, R1_grid, R2_F, focal_color, f_label)
    if persist_C:
        _draw_zngi(ax, R1_grid, R2_C, comp_color, c_label)

    # crossing
    R1c, R2c = zngi_crossing(pF, pC, T, T_opt_F, R_max=R_max)
    if R1c is not None and np.isfinite(R2c):
        ax.plot(R1c, R2c, 'ko', ms=8, label='ZNGI crossing')

        if show_consumption:
            # Plot consumption vectors for each species, rooted at the
            # crossing, as short arrows showing who draws which resource.
            cF = consumption_vector(R1c, R2c, pF['w1'], pF['w2'],
                                    pF['K1'], pF['K2'])
            cC = consumption_vector(R1c, R2c, pC['w1'], pC['w2'],
                                    pC['K1'], pC['K2'])
            arrow_len = 0.45 * R_max
            head_w = 0.03 * R_max
            ax.arrow(R1c, R2c, cF[0] * arrow_len, cF[1] * arrow_len,
                     head_width=head_w, head_length=2 * head_w,
                     length_includes_head=True,
                     fc=focal_color, ec=focal_color, alpha=0.9,
                     zorder=5)
            ax.arrow(R1c, R2c, cC[0] * arrow_len, cC[1] * arrow_len,
                     head_width=head_w, head_length=2 * head_w,
                     length_includes_head=True,
                     fc=comp_color, ec=comp_color, alpha=0.9,
                     zorder=5)

    if show_supply:
        ax.plot(pF['R1_in'], pF['R2_in'], marker='*',
                color='gold', markeredgecolor='k',
                markersize=15, label='supply point')

    ax.set_xlim(0, R_max)
    ax.set_ylim(0, R_max)
    ax.set_xlabel('R1 (focal-preferred)')
    ax.set_ylabel('R2 (competitor-preferred)')
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def plot_trait_trajectory_on_zngi(ax, pF, pC, T_trait_pairs,
                                  R_max=25.0, cmap_name='viridis'):
    """
    Draw the focal ZNGI at a series of (T, T_opt_F) pairs to visualise
    how it moves under joint warming + evolution. Competitor ZNGI drawn
    at the final T only.
    """
    R1_grid = np.linspace(1e-3, R_max, 1500)
    cmap = plt.get_cmap(cmap_name)

    n = len(T_trait_pairs)
    for i, (T, Topt_F) in enumerate(T_trait_pairs):
        rho_F = required_g(T, Topt_F, pF)
        R2_F = zngi_curve(R1_grid, pF['w1'], pF['w2'],
                          pF['K1'], pF['K2'], rho_F)
        color = cmap(i / max(n - 1, 1))
        label = (f"T={T:.1f}, T_opt={Topt_F:.1f}"
                 if i in (0, n // 2, n - 1) else None)
        ax.plot(R1_grid, R2_F, color=color, lw=2, alpha=0.9, label=label)

    # competitor at final T
    T_final = T_trait_pairs[-1][0]
    rho_C = required_g(T_final, pC['T_opt'], pC)
    R2_C = zngi_curve(R1_grid, pC['w1'], pC['w2'],
                      pC['K1'], pC['K2'], rho_C)
    ax.plot(R1_grid, R2_C, 'C3--', lw=2.5,
            label=f'competitor ZNGI (T={T_final:.1f})')

    ax.plot(pF['R1_in'], pF['R2_in'], marker='*',
            color='gold', markeredgecolor='k', markersize=14,
            label='supply point')
    ax.set_xlim(0, R_max); ax.set_ylim(0, R_max)
    ax.set_xlabel('R1'); ax.set_ylabel('R2')
    ax.set_title('Focal ZNGI under joint warming + evolution')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')


def plot_rho_vs_T(ax, pF, pC, T_grid):
    """
    Plot rho_i(T) = d/b for focal (using its *starting* trait) and
    competitor across a temperature sweep. The horizontal limit
    rho = w1 + w2 marks the absolute extinction ceiling.
    """
    rho_F = [required_g(T, pF['T_opt_ref'], pF) for T in T_grid]
    rho_C = [required_g(T, pC['T_opt'], pC) for T in T_grid]

    ax.plot(T_grid, rho_F, 'C0-', lw=2,
            label=f"focal (T_opt={pF['T_opt_ref']:.1f})")
    ax.plot(T_grid, rho_C, 'C3-', lw=2,
            label=f"competitor (T_opt={pC['T_opt']:.1f})")

    persist_F = pF['w1'] + pF['w2']
    persist_C = pC['w1'] + pC['w2']
    ax.axhline(persist_F, color='C0', ls=':', alpha=0.6,
               label=f"focal ceiling w1+w2={persist_F:.2f}")
    ax.axhline(persist_C, color='C3', ls=':', alpha=0.6,
               label=f"competitor ceiling {persist_C:.2f}")

    ax.set_xlabel('temperature (°C)')
    ax.set_ylabel(r'$\rho_i(T) = d_i / b_i$')
    ax.set_title('Required g-value vs temperature')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
