# src/models/base_model.py

class ReactionNetwork:
    """
    Basic container for an MJP reaction network.
    """

    def __init__(self, species_names, rates, reactants, stoich, classes):
        """
        species_names : list[str]
        rates         : list[float] (k_r)
        reactants     : list[list[int]]  # reactant indices per reaction
        stoich        : list[list[int]]  # stoichiometry changes per reaction
        classes       : list[int]        # reaction → class mapping
        """
        self.species = species_names
        self.n_species = len(species_names)
        self.rates = rates
        self.reactants = reactants
        self.stoich = stoich
        self.classes = classes
        self.R = len(rates)
