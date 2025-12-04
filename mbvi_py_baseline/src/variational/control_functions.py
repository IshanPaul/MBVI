# src/variational/control_functions.py

class ConstantControl:
    """
    eta_k(t) = constant scalar.
    """

    def __init__(self, eta_dict):
        # eta_dict: class_id → constant scaling
        self.eta = eta_dict

    def eval(self, t):
        return self.eta
