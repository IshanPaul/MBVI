# src/solvers/ode_euler.py

def euler_integrate(rhs_func, y0, t0, t1, dt):
    """
    Simple forward Euler integrator for baseline profiling.
    """
    t = t0
    y = list(y0)
    traj = []
    times = []

    while t <= t1:
        traj.append(list(y))
        times.append(t)
        dy = rhs_func(t, y)
        y = [yi + dt * dyi for yi, dyi in zip(y, dy)]
        t += dt

    return times, traj
