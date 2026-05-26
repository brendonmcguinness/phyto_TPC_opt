# Eco-evolutionary models of thermal adaptation under warming

A small, self-contained toolkit for the framework built on
Kremer et al. (2025) and Amarasekare & Johnson (2017). It starts at a
1-species / 1-resource chemostat with an evolving thermal optimum and
builds up to a 2-species / 2-resource model where depletable resource
dynamics and a temperature environment can be compared side by side.

## Layout

```
eco_evo_warming/
├── src/                          # all source scripts
│   ├── thermal_performance.py    # TPCs, trait trade-offs, selection gradients
│   ├── models.py                 # ODE systems + temperature drivers
│   ├── figdir.py                 # resolves the figures/ output directory
│   ├── run_1sp_1res.py           # Model 1 driver
│   ├── run_2sp_1res.py           # Model 2 driver
│   ├── run_2sp_2res.py           # Model 3 driver
│   ├── run_2sp_1R_tunneling.py   # evolutionary tunneling, 1 resource
│   ├── run_2sp_2R_tunneling.py   # evolutionary tunneling, 2 resources
│   ├── zngi_plot.py              # ZNGI geometry + plotting helpers
│   └── zngi_demo.py              # ZNGI figure suite
└── figures/                      # all generated PNGs land here
```

## State variables & equations

**Model 1** — one evolving species, one chemostat resource:

```
dR/dt     = D (R_in - R) - c b(T, T_opt) g(R) N
dN/dt     = [ b(T, T_opt) g(R) - d(T) ] N
dT_opt/dt = h² V_P  · ∂W/∂T_opt
```

**Model 2** — focal + non-evolving competitor, shared resource:

```
dR/dt     = D (R_in - R) - c₁ b₁ g(R) N₁ - c₂ b₂ g(R) N₂
dN₁/dt    = [ b₁(T, T_opt,1) g(R) - d₁(T) ] N₁
dN₂/dt    = [ b₂(T, T_opt,2) g(R) - d₂(T) ] N₂
dT_opt,1/dt = h² V_P · ∂W₁/∂T_opt,1
```

**Model 3** — two substitutable, depletable resources with species-specific
preferences `(w₁, w₂)`. Temperature is a non-depletable environmental
driver. Same trait evolution equation, extended selection gradient that
integrates both resource axes.

## Temperature drivers

Provided in `models.py`:

- `constant_T(T0)` — fixed T.
- `warming_ramp(T0, rate, T_final)` — linear ramp, optionally capped.
- `seasonal_warming(T0, rate, amplitude, period, T_final)` — ramp + sine.

Swap any of these into the `T_of_t` argument of the RHS functions.

## What each driver shows

- **`run_1sp_1res.py`** — trait equilibration, adaptation lag, and the
  evolutionary-rescue boundary as a function of warming rate.
- **`run_2sp_1res.py`** — the role of competitor `T_opt`:
  pre-adapted, intermediate (fitness valley / character displacement),
  and past the novel `T` (thermal refuge). Ends with a competitor-`T_opt`
  scan that maps the persistence corridor.
- **`run_2sp_2res.py`** — how much resource-axis partitioning is needed
  for coexistence and for the focal species to continue evolving under
  warming. Includes a preference-asymmetry scan.

## Run

```
cd eco_evo_warming
python src/run_1sp_1res.py
python src/run_2sp_1res.py
python src/run_2sp_2res.py
```

Each script writes its PNGs into the `figures/` folder, regardless of the
working directory you launch it from.

## Where extensions plug in

- **Different TPC shape (Sharpe–Schoolfield):** add it alongside
  `birth_rate` in `thermal_performance.py` and pass a flag / alternative
  callable into the RHS; the selection-gradient functions are the only
  other things to update.
- **Evolving competitor:** replicate the `dT_opt` line for species 2 in
  `rhs_2sp_*` and add a component to the state vector.
- **Essential (Liebig) resources:** replace the substitutable sum with
  `min(U_{i,1}, U_{i,2})` in `rhs_2sp_2R` and adjust the gradient.
- **Evolving niche width:** add `σ_b` as a second evolving trait and use
  a 2-trait `(h² V_P)` matrix; `d_birth_dTopt` already gives the
  template for analytic derivatives.

## Units

Rates are per-day (`b_max`, `d0`, `D`); temperatures are in °C;
resources and populations are dimensionless scaled quantities. `h²V_P`
is the product of heritability and phenotypic variance (in °C²),
scaling the canonical trait-evolution rate to the same time unit.
