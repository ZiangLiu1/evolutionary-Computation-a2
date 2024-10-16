import numpy as np
from pso_lib import AckleyFunction, ParticleSwarmOptimizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def plot_metrics(metrics, title, filename):
    """
    Plots the metrics collected during optimization.
    """
    num_runs = len(metrics['best_fitness'])  # 获取优化运行的次数
    colors = cm.viridis(np.linspace(0, 1, num_runs))  # 生成可区分的颜色

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(num_runs):
        axs[0, 0].plot(metrics['iterations'][i], metrics['best_fitness'][i], color=colors[i], alpha=0.5)
    axs[0, 0].set_title("Best Fitness Over Iterations")
    axs[0, 0].set_xlabel("Iterations")
    axs[0, 0].set_ylabel("Fitness")

    for i in range(num_runs):
        axs[0, 1].plot(metrics['iterations'][i], metrics['swarm_distance'][i], color=colors[i], alpha=0.5)
    axs[0, 1].set_title("Swarm Center to Global Optimum")
    axs[0, 1].set_xlabel("Iterations")
    axs[0, 1].set_ylabel("Distance")

    for i in range(num_runs):
        axs[1, 0].plot(metrics['iterations'][i], metrics['std_position'][i], color=colors[i], alpha=0.5)
    axs[1, 0].set_title("Standard Deviation of Positions")
    axs[1, 0].set_xlabel("Iterations")
    axs[1, 0].set_ylabel("Std Dev")

    for i in range(num_runs):
        axs[1, 1].plot(metrics['iterations'][i], metrics['velocity_mean'][i], color=colors[i], alpha=0.5)
    axs[1, 1].set_title("Mean Velocity Over Iterations")
    axs[1, 1].set_xlabel("Iterations")
    axs[1, 1].set_ylabel("Velocity Mean")

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(filename)
    plt.show()

# Collect metrics for analysis
metrics = {
    'iterations': [],
    'best_fitness': [],
    'swarm_distance': [],
    'std_position': [],
    'velocity_mean': []
}

# Run PSO on Ackley function 10 times
num_runs = 10
for run in range(num_runs):
    problem = AckleyFunction(10)
    optimizer = ParticleSwarmOptimizer(problem, swarm_size=30)

    # Store best fitness and other metrics for each iteration
    best_fitness_per_run = []
    swarm_distance_per_run = []
    std_position_per_run = []
    velocity_mean_per_run = []

    for _ in range(1000):
        global_best = optimizer.optimize(iterations=1)
        best_fitness_per_run.append(problem.evaluate(global_best))
        swarm_center = np.mean(optimizer.positions, axis=0)
        swarm_distance_per_run.append(np.linalg.norm(swarm_center))
        std_position_per_run.append(np.std(optimizer.positions))
        velocity_mean_per_run.append(np.mean(np.linalg.norm(optimizer.velocities, axis=1)))

    # Store all metrics
    metrics['iterations'].append(list(range(1000)))
    metrics['best_fitness'].append(best_fitness_per_run)
    metrics['swarm_distance'].append(swarm_distance_per_run)
    metrics['std_position'].append(std_position_per_run)
    metrics['velocity_mean'].append(velocity_mean_per_run)


# Plot the collected metrics
plot_metrics(metrics, "PSO on Ackley Function (Swarm Size 30)", "results/metrics_std_pso.png")
