# plot_metrics.py

import matplotlib.pyplot as plt

def plot_metrics(metrics):
    iterations = range(len(metrics['best_fitness']))
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Best fitness
    axs[0, 0].plot(iterations, metrics['best_fitness'])
    axs[0, 0].set_title('Best Fitness Value')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Fitness')

    # Swarm center distance to global optimum
    axs[0, 1].plot(iterations, metrics['swarm_center_dist'])
    axs[0, 1].set_title('Swarm Center Distance to Global Optimum')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Distance')

    # Standard deviation of particle positions
    axs[1, 0].plot(iterations, metrics['position_std_dev'])
    axs[1, 0].set_title('Position Standard Deviation')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Std Dev')

    # Mean velocity length
    axs[1, 1].plot(iterations, metrics['mean_velocity_length'])
    axs[1, 1].set_title('Mean Velocity Length')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Velocity')

    plt.tight_layout()
    plt.show()
