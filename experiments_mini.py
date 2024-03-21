from simulations.elastic_collisions import Body, HiddenVariables, Variables, ElasticCollisionSimulation
from simulations.elastic_collisions_rust_connector import SimulationData, load_rust_simulation_data_from_json
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


def run_simulation(
    total_time: float,
    dt: float,
    space_size: float,
    max_radius: float,
    acceleration_coefficient_value: float,
    constant_mass_value: float,
    constant_radius_value: float,
    velocity_distribution: Distribution,
    position_distribution: Distribution,
    num_bodies: int,
) -> ElasticCollisionSimulation:

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
    
    return simulation

def plot_simulation(
    position_history_by_timestep, 
    velocity_history_by_timestep, 
    space_size, 
    max_radius, 
    num_timesteps_to_plot=40,
    file_path='data/output.png'
):
    num_timesteps = len(position_history_by_timestep)

    # Number of timesteps to plot
    num_timesteps = num_timesteps_to_plot

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
    plt.savefig(file_path)

def get_plot_format_from_python_simulation(simulation):
    position_history = [body.position_history for body in simulation.bodies]
    velocity_history = [body.velocity_history for body in simulation.bodies]
    # position_history / velocity_history 
    # = [body1, body2, body3, ...] 
    # = [[t1, t2, t3, ...], [t1, t2, t3, ...], [t1, t2, t3, ...], ...] 
    # = [[(x1, y1), (x2, y2), (x3, y3), ...], [(x1, y1), (x2, y2), (x3, y3), ...], [(x1, y1), (x2, y2), (x3, y3), ...], ...]
    position_history_by_timestep = list(map(list, zip(*position_history)))
    velocity_history_by_timestep = list(map(list, zip(*velocity_history)))
    # position_history_by_timestep / velocity_history_by_timestep 
    # = [t1, t2, t3, ...]]
    # = [[body1, body2, body3, ...], [body1, body2, body3, ...], [body1, body2, body3, ...], ...
    # = [[(x1, y1), (x2, y2), (x3, y3), ...], [(x1, y1), (x2, y2), (x3, y3), ...], [(x1, y1), (x2, y2), (x3, y3), ...], ...
    space_size = simulation.variables.space_size[0].item()
    max_radius = max(simulation.variables.radii).item()
    return position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius

def get_plot_format_from_python_simulation_json(json_filepath: str):
    simulation = ElasticCollisionSimulation.load_from_json(json_filepath)
    position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius = get_plot_format_from_python_simulation(simulation)
    return position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius


def get_plot_format_from_rust_simulation(simulation_data: SimulationData):
    # Initialize lists to hold the history of positions and velocities by timestep
    position_history_by_timestep = []
    velocity_history_by_timestep = []

    # Iterate through each timestep in the state history
    for timestep in simulation_data.state_history[0]: # for now only look at one simulation (multiple simulations would be in state_history[0] and state_history[1] etc.)
        # Initialize lists to hold the positions and velocities of all bodies at the current timestep
        timestep_positions = []
        timestep_velocities = []
        timestep_radii = []

        # Iterate through the states of each body at the current timestep
        for i in range(simulation_data.num_bodies):
            timestep_positions.append([timestep.positions_x[i], timestep.positions_y[i]])
            timestep_velocities.append([timestep.velocities_x[i], timestep.velocities_y[i]])
            timestep_radii.append([timestep.radii[i]])
            
        # After processing all bodies for the current timestep, add the aggregated data to the history lists
        position_history_by_timestep.append(timestep_positions)
        velocity_history_by_timestep.append(timestep_velocities)
        space_size = simulation_data.space_size_x
        max_radius = max(timestep_radii[0]) # all radii are the same for all bodies currently ! change if needed

    return position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius


def get_plot_format_from_rust_simulation_json(json_filepath: str):
    simulation_data = load_rust_simulation_data_from_json(json_filepath)
    position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius = get_plot_format_from_rust_simulation(simulation_data)
    return position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius


if __name__ == "__main__":

    # total_time = 10.0 
    # dt = 0.1
    # space_size = 10.0
    # max_radius = space_size // 10.0
    # acceleration_coefficient_value = 0.0
    # constant_mass_value = 1.0
    # constant_radius_value = max_radius
    # velocity_distribution = torch.distributions.Uniform(low=-5.0, high=5.0)
    # position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)
    # num_bodies = 4
    # print(velocity_distribution)
    
    # simulation = run_simulation(
    #     total_time=total_time,
    #     dt=dt,
    #     space_size=space_size,
    #     max_radius=max_radius,
    #     acceleration_coefficient_value=acceleration_coefficient_value,
    #     constant_mass_value=constant_mass_value,
    #     constant_radius_value=constant_radius_value,
    #     velocity_distribution=velocity_distribution,
    #     position_distribution=position_distribution,
    #     num_bodies=num_bodies
    # )
    
    # # safe the simulation to a json file
    # simulation.save_to_json('data/simulation_save.json')
    # load the simulation from the json file
    
    # python sym
    # position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius = get_plot_format_from_python_simulation_json('data/simulation_save.json')
    # print(f'shape of position_history_by_timestep: {np.array(position_history_by_timestep).shape}')
    # print(f'shape of velocity_history_by_timestep: {np.array(velocity_history_by_timestep).shape}')
    # print(f'space_size: {space_size}')
    # print(f'max_radius: {max_radius}')
    
    
    # rust sym
    position_history_by_timestep, velocity_history_by_timestep, space_size, max_radius = get_plot_format_from_rust_simulation_json('sbi_algorithms_rust/abc/data/original_simulation_data.json')
    print(f'shape of position_history_by_timestep: {np.array(position_history_by_timestep).shape}')
    print(f'shape of velocity_history_by_timestep: {np.array(velocity_history_by_timestep).shape}')
    print(f'space_size: {space_size}')
    print(f'max_radius: {max_radius}')
    
    
    plot_simulation(
        position_history_by_timestep, 
        velocity_history_by_timestep, 
        space_size, max_radius, 
        num_timesteps_to_plot=40,
        file_path='data/output.png'
    )