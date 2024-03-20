from simulations.elastic_collisions import Body, HiddenVariables, Variables, ElasticCollisionSimulation
import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np







def plot_timestep(ax, positions, velocities, space_size, max_radius):
    # Normalize the velocity vectors
    normalized_velocities = velocities / np.linalg.norm(velocities, axis=1, keepdims=True) * space_size

    # Set the limits of the plot
    ax.set_xlim([0, space_size])
    ax.set_ylim([0, space_size])

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Add a grid
    ax.grid(True)

    # Convert positions list to numpy array
    positions = np.array(positions)

    # Plot the positions of the bodies
    ax.scatter(positions[:, 0], positions[:, 1], color='b')

    # Add transparent circles at the location of each body
    for pos in positions:
        circle = patches.Circle((pos[0], pos[1]), radius=max_radius, alpha=0.5, edgecolor='none')
        ax.add_patch(circle)

    # Plot the normalized velocities as vectors and display the original velocity values
    for pos, vel, orig_vel in zip(positions, normalized_velocities, velocities):
        ax.quiver(pos[0], pos[1], vel[0], vel[1], color='r')
        ax.text(pos[0] + orig_vel[0]/2, pos[1] + orig_vel[1]/2, f'({orig_vel[0]:.2f}, {orig_vel[1]:.2f})', color='r', fontsize=40)



if __name__ == "__main__":

    total_time = 10.0 
    dt = 0.1
    space_size = 10.0
    max_radius = space_size // 10.0
    acceleration_coefficient_value = 0.0
    constant_mass_value = 1.0
    constant_radius_value = max_radius
    velocity_distribution = torch.distributions.Uniform(low=-5.0, high=5.0)
    position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)

    num_bodies = 4
    VARIABLES = Variables(
        masses = torch.full((num_bodies,), constant_mass_value),
        radii = torch.full((num_bodies,), constant_radius_value),
        starting_positions = None,
        initial_velocities= None,
        acceleration_coefficients = torch.full((num_bodies,), acceleration_coefficient_value),
        num_bodies = num_bodies,
        space_size = torch.tensor([space_size, space_size]),
    )

    initial_positions = ElasticCollisionSimulation.sample_initial_positions_without_overlap(VARIABLES, position_distribution)
    initial_velocities = velocity_distribution.sample(sample_shape=torch.Size([num_bodies, 2]))
    VARIABLES.starting_positions = initial_positions

    HIDDENVARIABLES = HiddenVariables(num_bodies=None, 
                                    masses=None,
                                    radii=None,
                                    acceleration_coefficients=None, 
                                    initial_velocities=initial_velocities,) # the only hidden variable that we investigate in this scenario

    simulation = ElasticCollisionSimulation(variables=VARIABLES, 
                                            enable_logging=False, 
                                            noise=False)

    result = simulation.simulate(hidden_variables=HIDDENVARIABLES, \
                                total_time=total_time, \
                                dt=dt)

    position_history = simulation.get_position_history()
    velocity_history = simulation.get_velocity_history()
    # Transform the history so that each element represents a timestep
    position_history_by_timestep = list(map(list, zip(*position_history)))
    velocity_history_by_timestep = list(map(list, zip(*velocity_history)))

    num_timesteps = len(position_history_by_timestep)

    # Number of timesteps to plot
    num_timesteps = 40

    # Create a new figure with a grid of subplots
    fig, axs = plt.subplots(8, 5, figsize=(80, 120))  # 5 rows, 8 columns

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Loop over the timesteps
    for i in range(num_timesteps):
        # Get the positions and velocities for this timestep
        positions = position_history_by_timestep[i]
        velocities = velocity_history_by_timestep[i]

        # Plot this timestep
        plot_timestep(axs[i], positions, velocities, space_size, max_radius)
        axs[i].set_title(f'Timestep {i}', fontsize=46)


    # Adjust the space between subplots to be minimal
    #plt.subplots_adjust(wspace=0.01, hspace=0.1)
    # Display the plot
    plt.tight_layout()
    # Number of timesteps to plot
    plt.savefig('output.png')
    plt.show()