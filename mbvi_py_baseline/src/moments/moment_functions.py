# src/moments/moment_functions.py

def extract_mean(moment_vector, n_species):
    """
    In baseline version, the moment vector is simply the mean vector.
    """
    return moment_vector[:n_species]
