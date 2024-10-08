# pso_lib.py (continued)

class Particle:
    def __init__(self, dimensions, problem):
        self.position = np.random.uniform(problem.lower_bound, problem.upper_bound, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, inertia, cognitive_component, social_component, global_best_position):
        cognitive_velocity = cognitive_component * np.random.random() * (self.best_position - self.position)
        social_velocity = social_component * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def move(self):
        self.position += self.velocity

class PSO:
    def __init__(self, problem, swarm_size=30, inertia=0.7, cognitive_component=1.5, social_component=1.5, iterations=1000):
        self.problem = problem
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive_component = cognitive_component
        self.social_component = social_component
        self.iterations = iterations
        self.swarm = [Particle(problem.dimensions, problem) for _ in range(swarm_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def optimize(self):
        for iteration in range(self.iterations):
            for particle in self.swarm:
                fitness = self.problem.evaluate(particle.position)
                
                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # Update velocities and move particles
            for particle in self.swarm:
                particle.update_velocity(self.inertia, self.cognitive_component, self.social_component, self.global_best_position)
                particle.move()

        return self.global_best_position, self.global_best_fitness
