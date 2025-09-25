import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.oscillator import euler_solver, rk4_solver, scipy_ivp, steady_amp, amp_phase
from src.orbit import period_integrand, period_kepler, riemann, trapezoid, simpson, scipy_trap, scipy_simp



def plot_x(t, x, title, out):
    plt.figure()
    plt.plot(t, x)
    plt.xlabel(r"$t [s]$")
    plt.ylabel(r"$x(t) [m]$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_energy(t, E, title, out):
    plt.figure()
    plt.plot(t, E)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E(t) [J]$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"saved {out}")


def run_oscillator(args):
    m, k, c = args.m, args.k, args.c
    omega, F0 = args.omega, args.F0
    x0, v0 = args.x0, args.v0
    dt, tmax = args.dt, args.tmax

    if args.method == "euler":
        t, x, v, e = euler_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
    elif args.method == "rk4":
        t, x, v, e = rk4_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
    elif args.method == "scipy":
        t, x, v, e = scipy_ivp(m, k, c, omega, F0, x0, v0, dt, tmax)
    else:
        raise ValueError("Unknown oscillator method")
    
    plot_x(t, x, f"Trajectory of the oscillator ({args.method})", args.out_prefix + "_x.png")
    plot_energy(t, e, f"Energy of the oscillator ({args.method})", args.out_prefix + "_E.png")

    if args.resonance:
        omegas = np.linspace(args.res_min, args.res_max, args.res_n)
        amps_num = []
        amps_ana = []
        for om in omegas:
            """
            To preserve accuracy when scanning over omega, we specify dt and tmax for each frequency, and use frequency-independent quantities as input.
            res_cycles: number of drive cycles for each frequency
            res_npc: number of points per cycle
            """
            period = 2.0 * np.pi / om
            tmax_r = args.res_cycles * period
            dt_r = period / args.res_ppc
            # Since x0, v0 are independent of the resonance, they are set 0 for a better control of the steady state.
            if args.method == "euler":
                tt, xx, vv, ee = euler_solver(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            elif args.method == "rk4":
                tt, xx, vv, ee = rk4_solver(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            else:
                tt, xx, vv, ee = scipy_ivp(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            amps_num.append(steady_amp(tt, xx))
            amps_ana.append(amp_phase(m, k, c, omega, F0)[0])
        plt.figure()
        plt.plot(omegas, amps_num, "o", label="numerical")
        plt.plot(omegas, amps_ana, "-", label="analytic")
        plt.xlabel(r"drive frequency $omega [s^{-1}]$")
        plt.ylabel(r"steady state amplitude $X [m]$")
        plt.title(f"Resonance ({args.method})")
        plt.legend()
        plt.tight_layout()
        out = args.out_prefix + "_resonance.png"
        plt.savefig(out, dpi=160)
        print(f"saved {out}")


def run_orbit(args):
    a, e, mu, m = args.a, args.e, args.mu, args.m
    n = args.n
    T_ana = period_kepler(a, mu)
    f = period_integrand(a, e, mu, m)

    if args.method == "riemann":
        T_sim = riemann(f, 0.0, 2.0*np.pi, n)
    elif args.method == "trapezoid":
        T_sim = trapezoid(f, 0.0, 2.0*np.pi, n)
    elif args.method == "simpson":
        T_sim = simpson(f, 0.0, 2.0*np.pi, n)
    elif args.method == "scipy_trap":
        T_sim = scipy_trap(f, 0.0, 2.0*np.pi, n)
    elif args.method == "scipy_simp":
        T_sim = scipy_simp(f, 0.0, 2.0*np.pi, n)
    else:
        raise ValueError("Unknown orbit method")
    
    rel_err = abs(T_sim - T_ana) / T_ana
    print(r"Kepler's 3rd law via $theta$ integral")
    print(f"method={args.method:12s}   T_sim={T_sim:.9e}   T_ana={T_ana:.9e}   rel_err={rel_err:.3e}")

    if args.check_divergence:
        if e < 1:
            print("Set e > 1 (hyperbola) for the divergence check.")
        ns = np.logspace(2, 5, 10)
        vals = []
        for ni in ns:
            if args.method == "riemann":
                vals.append(riemann(f, 0.0, 2.0*np.pi, ni))
            elif args.method == "trapezoid":
                vals.append(trapezoid(f, 0.0, 2.0*np.pi, ni))
            elif args.method == "simpson":
                vals.append(simpson(f, 0.0, 2.0*np.pi, ni))
            elif args.method == "scipy_trap":
                vals.append(scipy_trap(f, 0.0, 2.0*np.pi, ni))
            elif args.method == "scipy_simp":
                vals.append(scipy_simp(f, 0.0, 2.0*np.pi, ni))
            else:
                raise ValueError("Unknown orbit method")
            print(f"number of samplings={ni:6d}  I_est={vals[-1]:.6e}")

        plt.figure()
        plt.semilogx(ns, vals, "o-")
        plt.xlabel("Number of samplings")
        plt.ylabel("I_est")
        plt.title(f"Divergence for e={e:.2f}")
        plt.tight_layout()
        plt.savefig(args.out_prefix + "_divergence.png", dpi=160)
        print(f"saved {args.out_prefix + '_divergence.png'}")

