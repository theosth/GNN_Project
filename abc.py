# variant1: (simulation1)
# CONSTANTS: velocities, radius (the same for each body)
# HIDDEN VARIABLES: mass (the same for each body) e.g 1 kg for each body

# simulation data
"""
X_obs = [t0: [(x0,y0), (x1,y1) ....],
         t1: [(x0,y0), (x1,y1) ....],
         ...
        ]
"""
"""
Y is sampled from p^s 
nu is sampled randomly 
phi(Y, nu) simulate sort of randomly a new X
"""

# Classical simple ABC (not amortized)
"""
# 1. t = 0, sample {}, -> later becomes {Y_(t) \sim p^s(Y|X_obs)}^T_t=1 
#  - Y = empty array {} 
# 2. repeat until t = T 
    -> sample Y,nu \sim p^s(Y,nu), simulate X \sim phi(Y,nu,S_t)
       - choose e.g. nu = +- normal(0.01) 
       - e.g. for 5 bodies: Y = 1,1,2,2,3
    -> if dist(X,X_obs) <= epsilon
        X_obs_t: we get that by accessing the X_obs array at time t
        we calculate X = [(x0,y0), (x1,y1) ....] from Y that we sampled
        - add Y to samples and increase t 
        - euclidean distance (X and X_obs) should be <= 0.5
       else: 
        - reject

"""

import torch
from torch import Tensor, Size
from torch.distributions import Distribution
from typing import Callable


def simulate(
    phi: Callable[[Tensor, Tensor, float, float], Tensor],
    Y: Tensor,
    eta: Tensor,
    total_time: float,
    dt: float,
) -> Tensor:
    """
    Simulate the data X given parameters Y and noise eta using the simulation function phi.
    :param phi: The simulation function.
    :param Y: Initial velocity of the body as a tensor. **Expected to be of shape (2,).**
    :param eta: Noise affecting the initial position of the body as a tensor. **Expected to be of shape (2,).**
    :param total_time: Total time to simulate.
    :param dt: Time step for simulation.
    :return: The final position of the body as a tensor **of shape (2,).**
    """
    # Assume phi updates the body's position given its velocity and some noise affecting its initial position.
    final_position = phi(Y, eta, total_time, dt)
    return final_position


def distance(X: Tensor, X_obs: Tensor) -> Tensor:
    """
    Calculate the distance between the simulated data X and the observed data X_obs using PyTorch.
    :param X: Simulated final position as a **tensor of shape (2,).**
    :param X_obs: Observed final position as a **tensor of shape (2,).**
    :return: The distance between X and X_obs as a tensor **of shape (1,).**
    """
    return torch.linalg.norm(X - X_obs)


def sample_from_prior(prior_dist: Distribution, size: Size = Size((2,))):
    """
    Sample a velocity vector Y from the prior distribution.
    :param prior_dist: A PyTorch distribution object for the prior.
    :param size: The size of the sample to draw.
    :return: A sample of Y as a tensor **of shape (2,).**
    """
    return prior_dist.sample(sample_shape=size)


def sample_noise(noise_dist: Distribution, size: Size = Size((2,))) -> Tensor:
    """
    Sample noise eta from its distribution.
    :param noise_dist: A PyTorch distribution object for the noise.
    :param size: The size of the sample to draw.
    :return: A sample of noise eta as a tensor **of shape (2,).**
    """
    return noise_dist.sample(sample_shape=size)


def ABC(
    phi: Callable[[Tensor, Tensor, float, float], Tensor],
    prior_dist: Distribution,
    noise_dist: Distribution,
    X_obs: Tensor,
    epsilon: float,
    N: int,
    total_time: float,
    dt: float,
) -> Tensor:
    """
    Approximate Bayesian Computation (ABC) algorithm using PyTorch.
    :param phi: The simulation function.
    :param prior_dist: A PyTorch distribution object for the prior.
    :param noise_dist: A PyTorch distribution object for the noise.
    :param X_obs: Observed final position as a tensor **of shape (2,).**
    :param epsilon: Acceptance threshold distance.
    :param N: Number of accepted samples to generate.
    :param total_time: Total time to simulate.
    :param dt: Time step for simulation.
    :return: Accepted samples of Y as a tensor **of shape (N, 2).**
    """
    accepted_Y: list[Tensor] = []
    while len(accepted_Y) < N:
        Y = sample_from_prior(prior_dist)
        eta = sample_noise(noise_dist)
        X_sim = simulate(phi, Y, eta, total_time, dt)
        if distance(X_sim, X_obs) <= epsilon:
            accepted_Y.append(Y)
    return torch.stack(accepted_Y)


# You'll need to define:
# - The `phi` function based on the provided simulation code.
# - Prior distribution `prior_dist` and noise distribution `noise_dist` as PyTorch distribution objects.
# - `Xobs` as the observed final position tensor.
# - `epsilon`, `N`, `total_time`, and `dt` based on your specific problem setup.


def sample_velocity_noise(mean=0.0, std=1.0):
    """
    Sample Gaussian noise to be added to initial velocities.

    Args:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A tensor representing a 2D noise vector.
    """
    noise_vx = torch.distributions.Normal(mean, std).sample()
    noise_vy = torch.distributions.Normal(mean, std).sample()
    return torch.tensor([noise_vx, noise_vy], dtype=torch.float64)

def sample_position_noise(mean=0.0, std=1.0):
    """
    Sample Gaussian noise to be added to initial positions.

    Args:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A tensor representing a 2D noise vector.
    """
    noise_x = torch.distributions.Normal(mean, std).sample()
    noise_y = torch.distributions.Normal(mean, std).sample()
    return torch.tensor([noise_x, noise_y], dtype=torch.float64)