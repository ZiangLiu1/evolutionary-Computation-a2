import numpy as np

class RealValuedProblem:
    def __init__(self, dimensions, bounds):
        self.dimensions = dimensions
        self.bounds = bounds

    def evaluate(self, solution):
        raise NotImplementedError("This method should be overridden by subclasses.")

class AckleyFunction(RealValuedProblem):
    def __init__(self, dimensions, bounds=(-30, 30)):
        super().__init__(dimensions, bounds)

    def evaluate(self, solution):
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(solution ** 2)))
        term2 = -np.exp(np.mean(np.cos(c * solution)))
        return term1 + term2 + a + np.e

class ParticleSwarmOptimizer:
    def __init__(self, problem, swarm_size=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.problem = problem
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.positions = np.random.uniform(
            problem.bounds[0], problem.bounds[1], (swarm_size, problem.dimensions)
        )
        self.velocities = np.random.uniform(-1, 1, (swarm_size, problem.dimensions))
        self.personal_best = self.positions.copy()
        self.personal_best_scores = np.array(
            [problem.evaluate(p) for p in self.personal_best]
        )
        self.global_best = self.personal_best[np.argmin(self.personal_best_scores)]

    def update_velocity(self, particle_idx):
        inertia_component = self.inertia * self.velocities[particle_idx]
        cognitive_component = (
            self.cognitive * np.random.rand() *
            (self.personal_best[particle_idx] - self.positions[particle_idx])
        )
        social_component = (
            self.social * np.random.rand() *
            (self.global_best - self.positions[particle_idx])
        )
        self.velocities[particle_idx] = inertia_component + cognitive_component + social_component

    def optimize(self, iterations=1000):
        for _ in range(iterations):
            for i in range(self.swarm_size):
                self.update_velocity(i)
                self.positions[i] += self.velocities[i]
                score = self.problem.evaluate(self.positions[i])
                if score < self.personal_best_scores[i]:
                    self.personal_best[i] = self.positions[i]
                    self.personal_best_scores[i] = score
                    if score < self.problem.evaluate(self.global_best):
                        self.global_best = self.positions[i]
        return self.global_best

