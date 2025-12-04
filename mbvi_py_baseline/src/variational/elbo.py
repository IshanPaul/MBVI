# src/variational/elbo.py

def dummy_elbo(moment_traj):
    """
    Placeholder: for baseline profiling only.
    """
    return -sum(sum(abs(x) for x in m) for m in moment_traj)
