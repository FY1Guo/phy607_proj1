import numpy as np


def period_integrand(a, e, mu, m):
    p = a * (1 - e**2)
    L = m * np.sqrt(mu * p)
    def f(theta):
        return (p**2 * m / L) / (1 + e * np.cos(theta))**2
    return f


def period_kepler(a, mu):
    return 2 * np.pi * np.sqrt(a**3 / mu)


def riemann(f, a0, b0, n):
    x = np.linspace(a0, b0, n+1)
    mid = (x[:-1] + x[1:]) / 2
    h = (b0 - a0)/n
    return np.sum(f(mid) * h)


def trapezoid(f, a0, b0, n):
    x = np.linspace(a0, b0, n+1)
    y = f(x)
    h = (b0 - a0)/n
    return h / 2 * (y[0] + 2 * sum(y[1:-1]) + y[-1])


def simpson(f, a0, b0, n):
    if n % 2 == 1:
        n += 1  # Simpson requires even n
    x = np.linspace(a0, b0, n+1)
    y = f(x)
    h = (b0 - a0)/n
    S = y[0] + 2*np.sum(y[2:-2:2]) + 4*np.sum(y[1:-1:2]) + y[-1]
    return float(h * S / 3)


def scipy_trap(f, a0, b0, n):
    from scipy.integrate import trapezoid
    x = np.linspace(a0, b0, n+1)
    y = f(x)
    return trapezoid(y, x)


def scipy_simp(f, a0, b0, n):
    from scipy.integrate import simpson
    if n % 2 == 1:
        n += 1  # Simpson requires even n
    x = np.linspace(a0, b0, n+1)
    y = f(x)
    return simpson(y, x)

