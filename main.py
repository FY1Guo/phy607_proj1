import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.oscillator import (
    euler_solver,
    rk4_solver,
    scipy_ivp,
    steady_amp,
    amp_phase,
    analytic_oscillator,
    analytic_energy,
)
from src.orbit import (
    period_integrand,
    period_kepler,
    riemann,
    trapezoid,
    simpson,
    scipy_trap,
    scipy_simp,
)


"""Use argparse to assemle the parameters and call the functions"""


def run_oscillator(args):
    """Run the oscillator simulation and plot results"""
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

    # if not resonance, plot x(t) and E(t)
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

    # if resonance, scan over omega and plot steady state amplitude vs omega
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
            amps_num.append(
                steady_amp(tt, xx)
            )  # truncate the first half to get steady state
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

    if args.compare_methods:
        compare_oscillator_methods(args)

    if args.conv_test:
        osc_convergence(args)


def compare_oscillator_methods(args):
    """
    This function called when --compare_methods is specified
    Compare E(t) for Euler, RK4, SciPy at the same dt, tmax
    """
    m, k, c = args.m, args.k, args.c
    omega, F0 = args.omega, args.F0
    x0, v0 = args.x0, args.v0
    dt, tmax = args.dt, args.tmax

    tE, xE, vE, eE = euler_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
    tR, xR, vR, eR = rk4_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
    tS, xS, vS, eS = scipy_ivp(m, k, c, omega, F0, x0, v0, dt, tmax)
    # Analytic results
    x_ana, xdot_ana, _ = analytic_oscillator(m, k, c, omega, F0, x0, v0)
    e_ana = analytic_energy(m, k, c, omega, F0, x0, v0)
    x_ana_list = []
    e_ana_list = []
    for t in tR:
        x_ana_list.append(x_ana(t))
        e_ana_list.append(e_ana(t))

    plt.figure()
    plt.plot(tE, xE, label="Euler")
    plt.plot(tR, xR, label="RK4")
    plt.plot(tS, xS, label="SciPy")
    plt.plot(tR, x_ana_list, label="Analytic")
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$x(t)$ [m]")
    plt.title("Oscillator: displacement comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_compare_x.png", dpi=160)
    print(f"saved {args.out_prefix + "_compare_x.png"}")

    plt.figure()
    plt.plot(tE, eE, label="Euler")
    plt.plot(tR, eR, label="RK4")
    plt.plot(tS, eS, label="SciPy")
    plt.plot(tR, e_ana_list, label="Analytic")
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$E(t)$ [J]")
    plt.title("Oscillator: energy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_compare_E.png", dpi=160)
    print(f"saved {args.out_prefix + "_compare_E.png"}")


def osc_convergence(args):
    """
    This function called when --conv_test is specified
    Run truncation error study vs dt
    """
    m, k, c = args.m, args.k, args.c
    omega, F0 = args.omega, args.F0
    x0, v0 = args.x0, args.v0
    tmax = args.tmax

    # Sample dt logarithmically between dt_min and dt_max with dt_num points
    dts = np.logspace(np.log10(args.dt_min), np.log10(args.dt_max), args.dt_num)
    err_x_euler, err_x_rk4 = [], []
    err_e_euler, err_e_rk4 = [], []

    def rms(y):
        return np.sqrt(np.mean(y**2))

    # analytic solution as reference
    x_ana, _, _ = analytic_oscillator(m, k, c, omega, F0, x0, v0)
    e_ana = analytic_energy(m, k, c, omega, F0, x0, v0)

    for dt in dts:
        tE, xE, _, eE = euler_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
        err_x_euler.append(rms(xE - x_ana(tE)))
        err_e_euler.append(rms(eE - e_ana(tE)))

        tR, xR, _, eR = rk4_solver(m, k, c, omega, F0, x0, v0, dt, tmax)
        err_x_rk4.append(rms(xR - x_ana(tR)))
        err_e_rk4.append(rms(eR - e_ana(tR)))

    # fit slopes on log-log
    def fit_slope(x, y):
        a, b = np.polyfit(np.log10(x), np.log10(y), 1)
        return a  # slope

    # displacement errors
    k_x_euler, k_x_rk4 = fit_slope(dts, err_x_euler), fit_slope(dts, err_x_rk4)
    k_x_rk4_trun = fit_slope(dts[3:], err_x_rk4[3:])  # truncate to avoid floor effects

    # energy errors
    k_e_euler, k_e_rk4 = fit_slope(dts, err_e_euler), fit_slope(dts, err_e_rk4)
    k_e_rk4_trun = fit_slope(dts[3:], err_e_rk4[3:])

    # print the fitted slopes
    print(
        f"Orders (displacement): Euler ~{k_x_euler:.2f} (expected ~1), RK4 ~{k_x_rk4_trun:.2f} (expected ~4)"
    )
    print(
        f"Orders (energy): Euler ~{k_e_euler:.2f} (expected ~1), RK4 ~{k_e_rk4_trun:.2f} (expected ~4)"
    )

    plt.figure()
    plt.loglog(dts, err_x_euler, "o-", label=f"Euler (slope={k_x_euler:.2f})")
    plt.loglog(dts, err_x_rk4, "o-", label=f"RK4 (slope={k_x_rk4:.2f})")
    plt.loglog(
        dts[3:], err_x_rk4[3:], "o-", label=f"RK4 truncated (slope={k_x_rk4_trun:.2f})"
    )
    plt.xlabel("dt")
    plt.ylabel("RMS error")
    plt.title("Oscillator: displacement truncation error scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_osc_conv.png", dpi=160)
    print(f"saved {args.out_prefix + "_osc_conv.png"}")

    plt.figure()
    plt.loglog(dts, err_e_euler, "o-", label=f"Euler (slope={k_e_euler:.2f})")
    plt.loglog(dts, err_e_rk4, "o-", label=f"RK4 (slope={k_e_rk4:.2f})")
    plt.loglog(
        dts[3:], err_e_rk4[3:], "o-", label=f"RK4 truncated (slope={k_e_rk4_trun:.2f})"
    )
    plt.xlabel("dt")
    plt.ylabel("RMS error")
    plt.title("Oscillator: energy truncation error scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_osc_E_conv.png", dpi=160)
    print(f"saved {args.out_prefix + "_osc_E_conv.png"}")


def run_orbit(args):
    """Run the orbit period integral and plot results"""
    a, e, mu = args.a, args.e, args.mu
    n = args.n
    T_ana = period_kepler(a, mu)
    f = period_integrand(a, e, mu)

    # only compute period if e < 1
    if 0.0 <= e < 1.0:
        if args.method == "riemann":
            T_sim = riemann(f, 0.0, 2.0 * np.pi, n)
        elif args.method == "trapezoid":
            T_sim = trapezoid(f, 0.0, 2.0 * np.pi, n)
        elif args.method == "simpson":
            T_sim = simpson(f, 0.0, 2.0 * np.pi, n)
        elif args.method == "scipy_trap":
            T_sim = scipy_trap(f, 0.0, 2.0 * np.pi, n)
        elif args.method == "scipy_simp":
            T_sim = scipy_simp(f, 0.0, 2.0 * np.pi, n)
        else:
            raise ValueError("Unknown orbit method")

        rel_err = abs(T_sim - T_ana) / T_ana  # relative error
        # print period integral results
        print(r"Kepler's 3rd law via $\theta$ integral")
        print(
            f"method={args.method:12s}   T_sim={T_sim:.9e}   T_ana={T_ana:.9e}   rel_err={rel_err:.3e}"
        )

        # check divergence not allowed for e < 1
        if args.check_divergence:
            print("Divergence demo requires e > 1; skipping because e < 1.")

    elif e >= 1.0:
        print(
            "e >= 1 selected (non-elliptic). Period integral is not finite. Skipping period computation."
        )
        if not args.check_divergence:
            print("  (use --check_divergence option for a divergence check)")
        if args.check_divergence:
            """Check the divergence of the integral for e >= 1"""
            ns = np.round(np.logspace(2, 5, 10)).astype(
                int
            )  # integers from 10^2 to 10^5
            vals = []
            res_riem = []
            res_trap = []
            res_simp = []
            res_scipy_trap = []
            res_scipy_simp = []
            # Define the integrand without other parameters
            g = lambda theta: 1.0 / (1.0 + e * np.cos(theta)) ** 2
            if args.compare_methods:
                """
                Plot all methods on the same graph if --compare_methods is specified
                """
                for ni in ns:
                    res_riem.append(riemann(g, 0.0, 2.0 * np.pi, ni))
                    res_trap.append(trapezoid(g, 0.0, 2.0 * np.pi, ni))
                    n2 = ni + (ni % 2)  # even for Simpson
                    res_simp.append(simpson(g, 0.0, 2.0 * np.pi, n2))
                    res_scipy_trap.append(scipy_trap(g, 0.0, 2.0 * np.pi, ni))
                    res_scipy_simp.append(scipy_simp(g, 0.0, 2.0 * np.pi, n2))

                plt.figure()
                plt.loglog(ns, res_riem, "o-", label="Riemann")
                plt.loglog(ns, res_trap, "o-", label="Trapezoid")
                plt.loglog(ns, res_simp, "o-", label="Simpson")
                plt.loglog(ns, res_scipy_trap, "o-", label="SciPy Trapezoid")
                plt.loglog(ns, res_scipy_simp, "o-", label="SciPy Simpson")
                plt.legend()
                plt.xlabel("Number of samplings")
                plt.ylabel("Integral value")
                plt.title(f"Divergence check for e={e:.2f}")
                plt.tight_layout()
                plt.savefig(args.out_prefix + "_cmp_divergence.png", dpi=160)
                print(f"saved {args.out_prefix + '_cmp_divergence.png'}")

            elif not args.compare_methods:
                """
                Plot only the selected method if --compare_methods is not specified
                """
                for ni in ns:
                    if args.method == "riemann":
                        vals.append(riemann(g, 0.0, 2.0 * np.pi, ni))
                    elif args.method == "trapezoid":
                        vals.append(trapezoid(g, 0.0, 2.0 * np.pi, ni))
                    elif args.method == "simpson":
                        vals.append(simpson(g, 0.0, 2.0 * np.pi, ni))
                    elif args.method == "scipy_trap":
                        vals.append(scipy_trap(g, 0.0, 2.0 * np.pi, ni))
                    elif args.method == "scipy_simp":
                        vals.append(scipy_simp(g, 0.0, 2.0 * np.pi, ni))
                    else:
                        raise ValueError("Unknown orbit method")

                    # print intermediate results
                    print(f"number of samplings={ni:6d}  I_est={vals[-1]:.6e}")

                plt.figure()
                plt.loglog(ns, res_riem, "o-", label="Riemann")
                plt.loglog(ns, res_trap, "o-", label="Trapezoid")
                plt.loglog(ns, res_simp, "o-", label="Simpson")
                plt.loglog(ns, res_scipy_trap, "o-", label="SciPy Trapezoid")
                plt.loglog(ns, res_scipy_simp, "o-", label="SciPy Simpson")
                plt.legend()
                plt.xlabel("Number of samplings")
                plt.ylabel("Integral value")
                plt.title(f"Divergence check for e={e:.2f}")
                plt.tight_layout()
                plt.savefig(args.out_prefix + "_divergence.png", dpi=160)
                print(f"saved {args.out_prefix + '_divergence.png'}")
        return

    else:
        raise ValueError(
            f"Invalid value for eccentricity: e={e}. Must be non-negative."
        )

    if args.compare_methods:
        compare_orbit_methods(args)

    if args.conv_test:
        orbit_convergence(args)


def compare_orbit_methods(args):
    """
    This function called when --compare_methods is specified
    Compare the integral results for all integral methods at the same N
    """
    mu = args.mu
    method_funcs = [
        ("riemann", riemann),
        ("trapezoid", trapezoid),
        ("simpson", simpson),
        ("scipy_trap", scipy_trap),
        ("scipy_simp", scipy_simp),
    ]

    a, e, mu = args.a, args.e, args.mu
    T_ana = period_kepler(a, mu)
    f = period_integrand(a, e, mu)

    # sample N linearly between n_min and n_max with n_num points
    ns = np.unique(
        np.round(np.linspace(args.n_min, args.n_max, args.n_num)).astype(int)
    )

    # compute T for each method at fixed N
    results = {name: [] for name, _ in method_funcs}
    for ni in ns:
        for name, fun in method_funcs:
            # Simpson needs even N
            N_use = ni + (ni % 2) if name == "simpson" else ni
            results[name].append(fun(f, 0.0, 2.0 * np.pi, N_use))

    # plot
    plt.figure()
    plt.hlines(T_ana, ns[0], ns[-1], colors="k", linestyles="dashed", label="analytic")
    for name in results:
        plt.plot(ns, results[name], "o-", label=name)
    plt.xlabel(r"$N$")
    plt.ylabel(r"period $T$ [s]")
    plt.title(f"Orbit: period vs N (a={a}, e={e})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_compare_T.png", dpi=160)
    print(f"saved {args.out_prefix + "_compare_T.png"}")


def orbit_convergence(args):
    """
    This function called when --conv_test is specified
    Run truncation error study vs N
    """
    a, e, mu = args.a, args.e, args.mu
    T_ana = period_kepler(a, mu)
    f = period_integrand(a, e, mu)

    if e >= 1.0:
        print(
            "e >= 1 selected (non-elliptic). Period integral is not finite. Skipping convergence test."
        )
        return

    # sample N logarithmically between n_min and n_max with n_num points
    ns = np.unique(
        np.round(
            np.logspace(np.log10(args.n_min), np.log10(args.n_max), args.n_num)
        ).astype(int)
    )
    err_riem, err_trap, err_simp = [], [], []
    for ni in ns:
        # compute relative error for each method
        # multiply by 2 because we only integrate from 0 to pi
        err_riem.append(abs(2 * riemann(f, 0.0, np.pi, ni) - T_ana) / T_ana)
        err_trap.append(abs(2 * trapezoid(f, 0.0, np.pi, ni) - T_ana) / T_ana)
        n2 = ni + (ni % 2)  # even for Simpson
        err_simp.append(abs(2 * simpson(f, 0.0, np.pi, n2) - T_ana) / T_ana)

    def fit_slope(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = y > 1e-14  # set a floor to avoid 0 divisor
        a, b = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
        return a

    # fit slopes on log-log
    k_riem = fit_slope(ns, err_riem)
    k_trap = fit_slope(ns, err_trap)
    k_simp = fit_slope(ns, err_simp)

    # print the fitted slopes
    print(
        f"Estimated behavior: Riemann ~ N^{k_riem:.2f}, Trapezoid ~ N^{k_trap:.2f}, Simpson ~ N^{k_simp:.2f}"
    )

    plt.figure()
    plt.loglog(ns, err_riem, "o-", label=f"Riemann (~N^{k_riem:.2f})")
    plt.loglog(ns, err_trap, "o-", label=f"Trapezoid (~N^{k_trap:.2f})")
    plt.loglog(ns, err_simp, "o-", label=f"Simpson (~N^{k_simp:.2f})")
    plt.xlabel("N panels")
    plt.ylabel("Relative error")
    plt.title("Orbit truncation error vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_orbit_conv.png", dpi=160)
    print(f"saved {args.out_prefix + "_orbit_conv.png"}")


def main():
    """Assemble the argument parser and call the appropriate functions"""
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
    po.add_argument(
        "--resonance", action="store_true", help="Analyze the resonant behavior"
    )
    po.add_argument(
        "--res_min", type=float, default=0.3, help="min simulation frequency"
    )
    po.add_argument(
        "--res_max", type=float, default=2.0, help="max simulation frequency"
    )
    po.add_argument(
        "--res_n", type=int, default=30, help="sampling points for frequency"
    )
    po.add_argument(
        "--res_cycles", type=float, default=60.0, help="drive cycles to simulate"
    )
    po.add_argument("--res_npc", type=int, default=200, help="points per cycle")
    po.add_argument(
        "--compare_methods",
        action="store_true",
        help="Compare E(t) for Euler, RK4, SciPy at the same dt, tmax",
    )
    po.add_argument(
        "--conv_test", action="store_true", help="Run truncation error study vs dt"
    )
    po.add_argument(
        "--dt_min", type=float, default=5e-4, help="min step size for convergence test"
    )
    po.add_argument(
        "--dt_max", type=float, default=5e-2, help="max step size for convergence test"
    )
    po.add_argument(
        "--dt_num", type=int, default=8, help="num of steps for convergence test"
    )
    po.set_defaults(func=run_oscillator)

    # Orbit
    pk = sub.add_parser("orbit", help="Kepler period integral")
    pk.add_argument(
        "--method",
        choices=["riemann", "trapezoid", "simpson", "scipy_trap", "scipy_simp"],
        default="simpson",
    )
    pk.add_argument("--a", type=float, default=3.0, help="semi-major axis")
    pk.add_argument("--e", type=float, default=0.6, help="eccentricity")
    pk.add_argument("--mu", type=float, default=1.0, help="gravitational parameter GM")
    pk.add_argument(
        "--n",
        type=int,
        default=1000,
        help="number of intervals for fixed-N period calc",
    )
    pk.add_argument(
        "--check_divergence",
        action="store_true",
        help="Check the divergence of the integral for e>=1",
    )
    pk.add_argument("--out_prefix", type=str, default="orbit", help="output prefix")
    pk.add_argument(
        "--compare_methods",
        action="store_true",
        help="Compare all integral methods at the same N",
    )
    pk.add_argument(
        "--conv_test", action="store_true", help="Run truncation error study vs N"
    )
    pk.add_argument("--n_min", type=int, default=2, help="min number of intervals")
    pk.add_argument("--n_max", type=int, default=15, help="max number of intervals")
    pk.add_argument("--n_num", type=int, default=20, help="num of sampling points")
    pk.set_defaults(func=run_orbit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
