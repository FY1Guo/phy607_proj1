# Damped Driven Oscillator & Kepler Orbit Integrals

This project contains two numerical physics problems:

1. **Driven damped harmonic oscillator** (ODE solver).
2. **Kepler orbit period** (evaluation of a definite integral).

It provides multiple numerical solvers, comparison with analytic results, resonance analysis, and convergence studies. Both problems are wrapped in a single `main.py`, and it is easy to swtich between the methods and study various situations

---

## Requirements

- Python
- Dependencies: `numpy`, `matplotlib`, `scipy`

Install with:
```bash
pip install numpy matplotlib scipy
```

---

## Usage

View top-level help:
```bash
python main.py --help
```
Each subcommand (`oscillator`, `orbit`) has detailed sub-level help:
```bash
python main.py oscillator --help
python main.py orbit --help
```

---

## Oscillator

### Example run

```bash
python main.py oscillator --method rk4 --m 1 --k 1 --c 0.2 --omega 1 --F0 1 --x0 0 --v0 1 --dt 1e-3 --tmax 60 --out_prefix osc_rk4
```

This generates:
- `osc_rk4_x.png`: displacement $x(t)$ vs time
- `osc_rk4_E.png`: energy $E(t)$ vs time
for the claimed method.

### Resonance

```bash
python main.py oscillator --method rk4 --resonance --res_min 0.3 --res_max 2 --res_n 30 --res_cycle 60 --res_npc 200 --out_prefix osc_res
```

This generates `osc_res_resonance.png`, the plot of steady state amplitude vs frequency.

### Compare methods

```bash
python main.py oscillator --compare_methods --out_prefix osc
```

This gives a combined plot `osc_compare_E.png` of $x(t)$ and $E(t)$ for Euler, RK4, SciPy and analytic solution.

### Convergence test

```bash
python main.py oscillator --conv_test --dt_min 5e-4 --dt_max 5e-2 --dt_num 8
```

This ouputs a plot `_osc_conv.png` which shows the truncations errors for all the methods scaling with step size `dt`.



## Orbit

### Example run

```bash
python main.py orbit --method simpson --a 3 --e 0.6 --mu 1 --n 1000 --out_prefix orb_simp
```
This computes the orbital period using the claimed method and compare to analytic Kepler's law.

### Divergence check

```bash
python main.py orbit --method simpson --a 3 --e 2 --mu 1 --check_divergence --out_prefix orb_div
```
For $e\ge1$, the orbit is hyperbola/parabola, and the integral diverges. This demo shows how the estimate of this divergent integral changes with increasing sampling.

### Compare methods

```bash
python main.py orbit --compare_methods --a_min 1 --a_max 10 --a_num 12 --out_prefix orb
```
This gives a combined plot `osc_compare_T.png` of $T(a)$ for Riemann, Trapezoid, Simpson, SciPy and analytic solution. 

### Convergence test
```bash
python main.py orbit --conv_test --n_min 2 --n_max 9 --n_num 8
```
This ouputs a plot `_orbit_conv.png` which shows the relative errors for all the numerical methods scaling as a function of number of intervals


## Notes
- The convergence test for the orbit problem does not give the expected behaviors. The default values for `n_min` `n_max` are set small, as larger numbers of intervals will result in extremely high accuracy for the integral estimation, leading to zero error up to the computer resolution. 