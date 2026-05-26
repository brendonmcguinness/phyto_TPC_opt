"""
Evolutionary tunneling on one resource.
========================================

Same set-up as run_2sp_1res.py:

    - Environment warms linearly from T0 (cold) to T_final (novel / hot).
    - Focal species (evolving) starts cold-adapted at T_opt = T0.
    - Competitor is STATIC at an intermediate T_opt between T0 and T_final.

On a single depletable resource there is no stable coexistence at fixed
T: the species with lower R* = DK/(mu - D) wins. What CAN happen under
warming is an "evolutionary tunnel":

    (i)  cold phase  —  focal is the better R* competitor, competitor
                        barely persists;
    (ii) valley      —  T sweeps past the competitor's T_opt; for a
                        window the competitor wins the R* race; focal
                        population N1 drops;
    (iii) escape     —  focal trait has evolved past the competitor's
                        T_opt; focal becomes the warm-side R* winner and
                        recovers.

Whether the focal species "tunnels through" or gets extinguished in
the valley is a race between dT/dt (environment) and h^2 V_P * dW/dT_opt
(trait), filtered by how small N1 is allowed to get.

Under the Kremer Eq. (2) form used here, a clean "success vs failure"
contrast emerges only in a fairly narrow, near-threshold regime. We
therefore tune the driver to that boundary:

    - the competitor is weaker than in the older Gaussian-birth draft,
    - it starts relatively rare,
    - its optimum is closer to the cold side of the warming ramp,
    - and the extinction floor is set very low so the focal can
      genuinely tunnel through a severe demographic valley.

The closer competitor optimum lets us use a modest evolutionary rate.
That avoids the visually confusing case where the focal T_opt rockets
far above the 30 C environment before the run ends.

This script then runs two scenarios:

    SUCCESS — warming just slow enough that the focal remains above the
              extinction floor through the valley and recovers by the
              end of the ramp.

    FAILURE — slightly faster warming with the SAME evolutionary
              parameters, pushing the focal just below the extinction
              floor before it can escape.

We enforce a hard extinction floor via a solve_ivp event: once N1 drops
below N_floor, it is pinned to zero for the remainder of the run. That
is what converts a *demographic depression* into an *extinction* and
cleanly separates the two outcomes.

Run:
    python run_2sp_1R_tunneling.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import rhs_2sp_1R, warming_ramp
from figdir import fig_path


# ---------------------------------------------------------------------------
# Parameters shared across scenarios
# ---------------------------------------------------------------------------

FOCAL_BASE = dict(
    # Kremer Eq. (2) TPC parameters
    b1=0.5, b2=0.05,
    d0=0.05, d2=0.12,
    T_opt_ref=20.0,
    D=0.10, R_in=10.0, K=1.0, c=1.0,
    h2_VP=0.3,   # overridden per scenario
)

# Tuned near-threshold competitor. Under the Kremer form there is no
# resource throttling of the selection gradient, so to recover a clear
# tunnel-vs-extinction contrast we use a weaker, initially rarer static
# competitor and a very deep extinction floor.
COMPETITOR = dict(
    b1=0.38, b2=0.05,
    d0=0.05, d2=0.12,
    K=1.0, c=1.0,
    T_opt=23.0,     # <-- intermediate, static
)

# Environment: warm from 20 -> 30 °C; scenarios differ in duration / rate.
T0_ENV, T_FINAL_ENV = 20.0, 30.0

N_FLOOR = 1e-12         # deep demographic extinction threshold for N1
Y0 = [FOCAL_BASE['R_in'] * 0.5, 1.0, 0.05, FOCAL_BASE['T_opt_ref']]


# ---------------------------------------------------------------------------
# Solver with a hard extinction floor on N1
# ---------------------------------------------------------------------------

def simulate(warming_rate, h2_VP, t_end, n_points=1600):
    """
    Integrate the 2sp_1R system; if N1 crosses below N_FLOOR the solver
    is restarted with N1 pinned to 0, giving a clean extinction.
    """
    pF = {**FOCAL_BASE, 'h2_VP': h2_VP}
    pC = dict(COMPETITOR)
    T_of_t = warming_ramp(T0=T0_ENV, warming_rate=warming_rate,
                          T_final=T_FINAL_ENV)

    def rhs(t, y):
        return rhs_2sp_1R(t, y, T_of_t, pF, pC)

    def extinction_event(t, y):
        return y[1] - N_FLOOR
    extinction_event.terminal = True
    extinction_event.direction = -1

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), Y0, t_eval=t_eval,
                    events=extinction_event,
                    method='LSODA', rtol=1e-9, atol=1e-12)

    # If extinction happened, continue the run with N1 pinned to 0.
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        t_ext = sol.t_events[0][0]
        y_ext = sol.y_events[0][0].copy()
        y_ext[1] = 0.0

        def rhs_post(t, y):
            dy = rhs_2sp_1R(t, y, T_of_t, pF, pC)
            dy[1] = 0.0          # focal is extinct
            dy[3] = 0.0          # freeze the trait once the focal is gone
            return dy

        t_eval_post = t_eval[t_eval > t_ext]
        sol_post = solve_ivp(rhs_post, (t_ext, t_end), y_ext,
                             t_eval=t_eval_post,
                             method='LSODA', rtol=1e-9, atol=1e-12)
        # Stitch the two segments back together.
        t_full = np.concatenate([sol.t, sol_post.t])
        y_full = np.concatenate([sol.y, sol_post.y], axis=1)
        extinct = True
    else:
        t_full, y_full = sol.t, sol.y
        extinct = False

    return t_full, y_full, T_of_t, extinct, pF, pC


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_side_by_side(results, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex='col')

    for col, (label, (t, y, T_of_t, extinct, pF, pC)) in enumerate(results.items()):
        T_env = T_of_t(t)

        ax = axes[0, col]
        ax.plot(t, T_env, 'k-', label='T(t)')
        ax.plot(t, y[3], 'C0-', lw=2, label=r'focal $\bar T_{opt}$')
        ax.axhline(pC['T_opt'], color='C3', ls='--',
                   label=r'competitor $T_{opt}$')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{label}'
                     + ('  (focal extinct)' if extinct else '  (focal persists)'))
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        N1_plot = np.maximum(y[1], N_FLOOR)
        N2_plot = np.maximum(y[2], N_FLOOR)
        ax.plot(t, N1_plot, 'C0-', lw=2, label='focal $N_1$')
        ax.plot(t, N2_plot, 'C3-', lw=2, label='competitor $N_2$')
        ax.axhline(N_FLOOR, color='gray', ls=':', label='extinction floor')
        ax.set_ylabel('population')
        ax.set_yscale('log')
        ax.set_ylim(bottom=N_FLOOR)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(t, np.maximum(y[0], N_FLOOR), 'C2-')
        ax.set_ylabel('resource R')
        ax.set_xlabel('time')
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(fig_path(save_path), dpi=150)
    plt.close(fig)
    print(f'wrote {fig_path(save_path)}')


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # SUCCESS: tuned just below the persistence boundary.
    #   - Same h^2 V_P as the failure case, but warming is slightly
    #     slower, so the focal stays barely above the extinction floor
    #     and recovers by the end of the warming window.
    t_s, y_s, Tf_s, ext_s, pF_s, pC_s = simulate(
        warming_rate=0.005,
        h2_VP=0.10,
        t_end=2500.0,
    )

    # FAILURE: same evolutionary capacity, slightly faster warming.
    #   - The focal now crosses the extinction floor during the valley
    #     before the warm-side recovery can occur.
    t_f, y_f, Tf_f, ext_f, pF_f, pC_f = simulate(
        warming_rate=0.007,
        h2_VP=0.10,
        t_end=2500.0,
    )

    plot_side_by_side(
        {
            'SUCCESS — rate=0.005, h²V=0.10':
                (t_s, y_s, Tf_s, ext_s, pF_s, pC_s),
            'FAILURE — rate=0.007, h²V=0.10':
                (t_f, y_f, Tf_f, ext_f, pF_f, pC_f),
        },
        save_path='2sp_1R_tunneling.png',
    )

    # Print a short diagnostic.
    for tag, (t, y, Tf, ext, pF, pC) in [
        ('SUCCESS', (t_s, y_s, Tf_s, ext_s, pF_s, pC_s)),
        ('FAILURE', (t_f, y_f, Tf_f, ext_f, pF_f, pC_f)),
    ]:
        T_env_end = Tf(t[-1])
        print(f'{tag}: T(end)={T_env_end:.2f}°C  '
              f'T_opt1(end)={y[3,-1]:.2f}°C  '
              f'N1(end)={y[1,-1]:.2e}  '
              f'N2(end)={y[2,-1]:.2e}  '
              f'extinct={ext}')
