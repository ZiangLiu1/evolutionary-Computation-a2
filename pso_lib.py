# pso_lib.py

import numpy as np

class RealValuedOptimizationProblem:
    def __init__(self, dimensions, lower_bound, upper_bound):
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, position):
        """
        Abstract method to be overridden by specific problem implementations.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class AckleyFunction(RealValuedOptimizationProblem):
    def __init__(self, dimensions, lower_bound=-30, upper_bound=30):
        super().__init__(dimensions, lower_bound, upper_bound)

    def evaluate(self, position):
        """
        Evaluate the Ackley function.
        """
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(position)
        
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(position)) / d))
        cos_term = -np.exp(np.sum(np.cos(c * position)) / d)
        result = a + np.exp(1) + sum_sq_term + cos_term
        
        return result
