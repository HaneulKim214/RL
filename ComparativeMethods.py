

class ComparativeMethodsMAB():
    """
    Class containing variety of comparative methods for MAB problems
    """
    def __init__(self, method, n_prod):
        self.method = method
        self.n_prod = n_prod

    def __str__(self):
        return self.method


class ABn(ComparativeMethodsMAB):
    def __init__(self):
        super(ABN)