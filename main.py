import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.oscillator import euler_solver, rk4_solver, scipy_ivp, steady_amp, amp_phase
from src.orbit import period_integrand, period_kepler, riemann, trapezoid, simpson, scipy_trap, scipy_simp



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
    
    if not args.resonance:
        plt.figure()
        plt.plot(t, x)
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"$x(t)$ [m]")
        plt.title(f"Trajectory of the oscillator ({args.method})")
        plt.tight_layout()
        plt.savefig(args.out_prefix + "_x.png", dpi=160)
        print(f"saved {args.out_prefix + "_x.png"}")

        plt.figure()
        plt.plot(t, e)
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"$E(t)$ [J]")
        plt.title(f"Energy of the oscillator ({args.method})")
        plt.tight_layout()
        plt.savefig(args.out_prefix + "_E.png", dpi=160)
        print(f"saved {args.out_prefix + "_E.png"}")

    if args.resonance:
        omegas = np.linspace(args.res_min, args.res_max, args.res_n)
        amps_num = []
        amps_ana = []
        for om in omegas:
            """
            To preserve accuracy when scanning over omega, we specify dt and tmax for each frequency, and use frequency-independent quantities as input.
            res_cycles: number of cycles for simulation at each frequency
            res_npc: number of points per cycle
            """
            period = 2.0 * np.pi / om
            tmax_r = args.res_cycles * period
            dt_r = period / args.res_npc
            # Since x0, v0 are independent of the resonance, they are set 0 for a better control of the steady state.
            if args.method == "euler":
                tt, xx, vv, ee = euler_solver(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            elif args.method == "rk4":
                tt, xx, vv, ee = rk4_solver(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            else:
                tt, xx, vv, ee = scipy_ivp(m, k, c, om, F0, 0.0, 0.0, dt_r, tmax_r)
            amps_num.append(steady_amp(tt, xx))
            amps_ana.append(amp_phase(m, k, c, om, F0)[0])
        plt.figure()
        plt.plot(omegas, amps_num, "o", label="numerical")
        plt.plot(omegas, amps_ana, "-", label="analytic")
        plt.xlabel(r"Drive frequency $\omega$ [s$^{-1}$]")
        plt.ylabel(r"Steady state amplitude $X$ [m]")
        plt.title(f"Resonance ({args.method})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_prefix + "_resonance.png", dpi=160)
        print(f"saved {args.out_prefix + "_resonance.png"}")


def run_orbit(args):
    a, e, mu = args.a, args.e, args.mu
    n = args.n
    T_ana = period_kepler(a, mu)
    f = period_integrand(a, e, mu)

    if 0.0 <= e < 1.0:
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
        print(r"Kepler's 3rd law via $\theta$ integral")
        print(f"method={args.method:12s}   T_sim={T_sim:.9e}   T_ana={T_ana:.9e}   rel_err={rel_err:.3e}")

        if args.check_divergence:
            print("Divergence demo requires e > 1; skipping because e < 1.")

    elif e >= 1.0:
        print("e >= 1 selected (non-elliptic). Period integral is not finite. Skipping period computation.")
        if not args.check_divergence:
            print("  (use --check_divergence option for a divergence check)")
        if args.check_divergence:
            ns = np.round(np.logspace(2, 5, 10)).astype(int)
            vals = []
            # Define the integrand without other parameters
            g = lambda theta: 1.0 / (1.0 + e * np.cos(theta))**2
            for ni in ns:
                if args.method == "riemann":
                    vals.append(riemann(g, 0.0, 2.0*np.pi, ni))
                elif args.method == "trapezoid":
                    vals.append(trapezoid(g, 0.0, 2.0*np.pi, ni))
                elif args.method == "simpson":
                    vals.append(simpson(g, 0.0, 2.0*np.pi, ni))
                elif args.method == "scipy_trap":
                    vals.append(scipy_trap(g, 0.0, 2.0*np.pi, ni))
                elif args.method == "scipy_simp":
                    vals.append(scipy_simp(g, 0.0, 2.0*np.pi, ni))
                else:
                    raise ValueError("Unknown orbit method")
                print(f"number of samplings={ni:6d}  I_est={vals[-1]:.6e}")

            plt.figure()
            plt.loglog(ns, vals, "o-")
            plt.xlabel("Number of samplings")
            plt.ylabel("Integral value")
            plt.title(f"Divergence check for e={e:.2f}")
            plt.tight_layout()
            plt.savefig(args.out_prefix + "_divergence.png", dpi=160)
            print(f"saved {args.out_prefix + '_divergence.png'}")
        return
    
    else:
        raise ValueError(f"Invalid value for eccentricity: e={e}. Must be non-negative.")


def main():
    p = argparse.ArgumentParser(description="Assemble oscillator and orbit problems")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Oscillator
    po = sub.add_parser("oscillator", help="Driven damped oscillator")
    po.add_argument("--method", choices=["euler", "rk4", "scipy"], default="rk4")
    po.add_argument("--m", type=float, default=1.0, help="mass")
    po.add_argument("--k", type=float, default=1.0, help="spring constant")
    po.add_argument("--c", type=float, default=0.2, help="damping coefficient")
    po.add_argument("--omega", type=float, default=1.0, help="driving frequency")
    po.add_argument("--F0", type=float, default=1.0, help="driving amplitude")
    po.add_argument("--x0", type=float, default=0.0, help="initial position")
    po.add_argument("--v0", type=float, default=1.0, help="initial velocity")
    po.add_argument("--dt", type=float, default=1e-3, help="time step")
    po.add_argument("--tmax", type=float, default=60.0, help="max simulation time")
    po.add_argument("--out_prefix", type=str, default="osc", help="output prefix")
    # Resonance options
    po.add_argument("--resonance", action="store_true", help="Analyze the resonant behavior")
    po.add_argument("--res_min", type=float, default=0.3, help="min simulation frequency")
    po.add_argument("--res_max", type=float, default=2.0, help="max simulation frequency")
    po.add_argument("--res_n", type=int, default=30, help="sampling points for frequency")
    po.add_argument("--res_cycles", type=float, default=60.0, help="drive cycles to simulate")
    po.add_argument("--res_npc", type=int, default=200, help="points per cycle")
    po.set_defaults(func=run_oscillator)

    # Orbit
    pk = sub.add_parser("orbit", help="Kepler period integral")
    pk.add_argument("--method",
                    choices=["riemann", "trapezoid", "simpson", "scipy_trap", "scipy_simp"], default="simpson")
    pk.add_argument("--a", type=float, default=3.0, help="semi-major axis")
    pk.add_argument("--e", type=float, default=0.3, help="eccentricity")
    pk.add_argument("--mu", type=float, default=1.0, help="gravitational parameter GM")
    pk.add_argument("--n", type=int, default=4000)
    pk.add_argument("--check_divergence", action="store_true", help="Check the divergence of the integral for e>=1")
    pk.add_argument("--out_prefix", type=str, default="orbit", help="output prefix")
    pk.set_defaults(func=run_orbit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()