import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Callable

import time

from copy import deepcopy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from simulations.elastic_collisions import (
    ElasticCollisionSimulation,
    Variables,
    HiddenVariables,
)


def distance(X: Tensor, X_obs: Tensor) -> Tensor:
    """
    Calculate the distance between the simulated data X and the observed data X_obs using PyTorch.
    :param X: Simulated final position as a **tensor of shape (2,).**
    :param X_obs: Observed final position as a **tensor of shape (2,).**
    :return: The distance between X and X_obs as a tensor **of shape (1,).**
    """
    # return torch.linalg.norm(X - X_obs)
    return torch.sqrt(torch.sum((X - X_obs) ** 2))


def sample_from_prior_velocities(amount: int) -> Tensor:
    """
    Sample a set of velocity vectors from the prior distribution for a given amount of bodies.
    :param amount: The number of velocity vectors to sample.
    :return: A tensor of shape (amount, 2), where each row is a 2D velocity vector [vx, vy].
    """
    prior_distribution = torch.distributions.Uniform(low=-2.0, high=2.0)
    # Sample `amount` times for each component
    vx_samples = prior_distribution.sample(sample_shape=torch.Size([amount]))
    vy_samples = prior_distribution.sample(sample_shape=torch.Size([amount]))
    # Combine into a single tensor of shape (amount, 2)
    velocity_vectors = torch.stack([vx_samples, vy_samples], dim=1)

    return velocity_vectors


def ABC_Algo(
    variables: Variables,
    sample_from_prior: Callable[[int], Tensor],
    X_obs: Tensor,
    epsilon: float,
    T: int,
    total_time: float,
    dt: float,
) -> Tensor:
    simulation = ElasticCollisionSimulation(variables, enable_logging=False)

    accepted_Y: list[Tensor] = []
    accepted_X: list[Tensor] = []

    current_iteration = 0
    start_time = time.time()  # Start timing
    while len(accepted_Y) < T:
        current_iteration += 1

        Y = sample_from_prior(variables.num_bodies)
        hidden_variables = HiddenVariables(
            masses=None,
            radii=None,
            acceleration_coefficients=None,
            initial_velocities=Y,
            num_bodies=None,
        )

        X_sim = simulation.simulate(hidden_variables, total_time, dt)

        if distance(X_sim, X_obs) <= epsilon:
            accepted_Y.append(Y)
            accepted_X.append(X_sim)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"Total iterations: {current_iteration}")
    print(f"time per iteration: {elapsed_time / current_iteration} seconds")
    print(f"Total time: {elapsed_time} seconds")
    return torch.stack(accepted_Y), torch.stack(accepted_X)


if __name__ == "__main__":

    ### 1. Setting simulation parameters
    total_time = 10.0
    dt = 0.1
    space_size = 10.0
    max_radius = space_size // 10.0
    constant_mass_value = 1.0
    constant_radius_value = max_radius
    acceleration_coefficient_value = 0.0
    velocity_distribution = torch.distributions.Uniform(low=-5.0, high=5.0)
    position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)

    num_bodies = 4
    VARIABLES = Variables(
        masses=torch.full((num_bodies,), constant_mass_value),
        radii=torch.full((num_bodies,), constant_radius_value),
        starting_positions=None,
        num_bodies=num_bodies,
        acceleration_coefficients=torch.full(
            (num_bodies,), acceleration_coefficient_value
        ),
        initial_velocities=None,
        space_size=torch.tensor([space_size, space_size]),
    )

    initial_positions = (
        ElasticCollisionSimulation.sample_initial_positions_without_overlap(
            VARIABLES, position_distribution
        )
    )
    VARIABLES.starting_positions = deepcopy(initial_positions)
    true_initial_positions = deepcopy(initial_positions)
    initial_velocities = velocity_distribution.sample(
        sample_shape=torch.Size([num_bodies, 2])
    )

    ### 2. Running the simulation once to create observed data X_obs
    simulation_obs = ElasticCollisionSimulation(
        VARIABLES, enable_logging=False, noise=False
    )
    final_positions = simulation_obs.simulate(
        hidden_variables=HiddenVariables(
            masses=None,
            radii=None,
            acceleration_coefficients=None,
            initial_velocities=initial_velocities,
            num_bodies=None,
        ),
        total_time=total_time,
        dt=dt,
    )

    position_history = simulation_obs.get_position_history()
    velocity_history = simulation_obs.get_velocity_history()

    ### 3. Running the ABC algorithm
    accepted_Y_, accepted_X_ = ABC_Algo(
        variables=VARIABLES,
        sample_from_prior=lambda amount: velocity_distribution.sample(
            sample_shape=([amount, 2])
        ),
        X_obs=final_positions,
        epsilon=2.0,
        T=1,
        total_time=1.0,
        dt=0.1,
    )
    print(f"SIM: accepted_X (final positions): {accepted_X_}")
    print(f"SIM: accepted_Y (initial velocities): {accepted_Y_}")
    print("--------------------------------------------------------")
    print(f"OBS: initial_positions: {initial_positions}")
    print(f"OBS: final_positions: {final_positions}")
    print(f"OBS: initial_velocities: {initial_velocities}")
