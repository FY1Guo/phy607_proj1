import numpy as np


def rhs(t, x, v, m, k, c, omega, F0):
    a = (F0 * np.cos(omega * t) - c * v - k * x) / m
    return v, a


def energy(x, v, m, k):
    potential = 0.5 * k * x**2
    kinetic = 0.5 * m * v**2
    total = potential + kinetic
    return total


def euler_solver(m, k, c, omega, F0, x0, v0, dt, tmax):
    e0 = energy(x0, v0, m, k)
    t_hist = [0.0]; x_hist = [x0]; v_hist = [v0]; e_hist = [e0]

    t, x, v = 0.0, x0, v0
    
    while t < tmax:
        dx, dv = rhs(t, x, v, m, k, c, omega, F0)
        x = x + dx * dt
        v = v + dv * dt
        e = energy(x, v, m, k)
        t += dt

        t_hist.append(t)
        x_hist.append(x)
        v_hist.append(v)
        e_hist.append(e)

    return np.array(t_hist), np.array(x_hist), np.array(v_hist), np.array(e_hist)


def rk4_solver(m, k, c, omega, F0, x0, v0, dt, tmax):
    e0 = energy(x0, v0, m, k)
    t_hist = [0.0]; x_hist = [x0]; v_hist = [v0]; e_hist = [e0]

    t, x, v = 0.0, x0, v0

    while t < tmax:
        dx1, dv1 = rhs(t, x, v, m, k, c, omega, F0)
        dx2, dv2 = rhs(t + 0.5*dt, x + 0.5*dt*dx1, v + 0.5*dt*dv1, m, k, c, omega, F0)
        dx3, dv3 = rhs(t + 0.5*dt, x + 0.5*dt*dx2, v + 0.5*dt*dv2, m, k, c, omega, F0)
        dx4, dv4 = rhs(t + dt, x + dt*dx3, v + dt*dv3, m, k, c, omega, F0)
        x += (dt/6.0)*(dx1 + 2*dx2 + 2*dx3 + dx4)
        v += (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)
        e = energy(x, v, m, k)
        t += dt

        t_hist.append(t)
        x_hist.append(x)
        v_hist.append(v)
        e_hist.append(e)
    return np.array(t_hist), np.array(x_hist), np.array(v_hist), np.array(e_hist)


def scipy_ivp(m, k, c, omega, F0, x0, v0, dt, tmax):
    from scipy.integrate import solve_ivp
    n = int(round(tmax / dt))
    t_hist = np.linspace(0.0, tmax, n+1)
    def f(t, y):
        x, v = y
        dx, dv = rhs(t, x, v, m, k, c, omega, F0)
        return [dx, dv]
    sol = solve_ivp(f, (0.0, tmax), [x0, v0], method="RK45", dense_output=True, max_step=dt)

    Y = sol.sol(t_hist)
    x_hist = Y[0]
    v_hist = Y[1]
    e_hist = 1/2 * m * v_hist**2 + 1/2 * k * x_hist**2
    return t_hist, x_hist, v_hist, e_hist


def amp_phase(m, k, c, omega, F0):
    omega0 = np.sqrt(k / m)
    beta = c / (2 * m)
    X = (F0 / m) / np.sqrt((omega0**2 - omega**2) ** 2 + (2 * beta * omega) ** 2)
    phi = np.arctan2(2 * beta * omega, (omega0**2 - omega**2))
    return X, phi


def steady_amp(t, x, fraction=0.5):
    n = int((1.0 - fraction) * len(t))
    xt = x[n:]
    return 0.5 * (np.max(xt) - np.min(xt))


def analytic_oscillator(m, k, c, omega, F0, x0, v0):
    omega0 = np.sqrt(k / m)
    beta = c / (2 * m)
    X, phi = amp_phase(m, k, c, omega, F0)
    xp = lambda t: X * np.cos(omega * t - phi)

    if beta < omega0:  # underdamped
        omega_d = np.sqrt(omega0**2 - beta**2)
        A1 = x0 - X * np.cos(phi)
        A2 = (v0 + beta * A1 - omega * X * np.sin(phi)) / omega_d
        def x(t):
            return np.exp(-beta * t) * (A1 * np.cos(omega_d * t) + A2 * np.sin(omega_d * t)) + xp(t)
        regime = "underdamped"
    elif beta == omega0:  # critical
        B1 = x0 - X * np.cos(phi)
        B2 = v0 + beta * (x0 - X * np.cos(phi)) - omega * X * np.sin(phi)
        def x(t):
            return (B1 + B2 * t) * np.exp(-beta * t) + xp(t)
        regime = "critical"
    else:  # overdamped
        r1 = -beta + np.sqrt(beta**2 - omega0**2)
        r2 = -beta - np.sqrt(beta**2 - omega0**2)
        C1 = ((v0 - omega * X * np.sin(phi)) - r2 * (x0 - X * np.cos(phi))) / (r1 - r2)
        C2 = (r1 * (x0 - X * np.cos(phi)) - (v0 - omega * X * np.sin(phi))) / (r1 - r2)
        def x(t):
            return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + xp(t)
        regime = "overdamped"
    return x, regime
