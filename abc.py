


#Classical simple ABC (not amortized)
"""
# 1. t = 0, sample {}, -> later becomes {Y_(t) \sim p^s(Y|X_obs)}^T_t=1 
# 2. repeat until t = T 
    -> sample Y,nu \sim p^s(Y,nu), simulate X \sim phi(Y,nu)
    -> if dist(X,X_obs) <= epsilon
        - add Y to samples and increase t 
       else: 
        - reject

"""


import numpy as np

def simulate(phi, Y, eta):
    """
    Simulate the data X given parameters Y and noise eta using the simulation function phi.
    
    Args:
    phi: A function that takes parameters Y and noise eta and returns simulated data X.
    Y: The parameters to simulate the data with.
    eta: The noise to simulate the data with.

    Returns:
    Simulated data X.
    """
    # Implement the simulation function according to your model
    return phi(Y, eta)

def distance(X, Xobs):
    """
    Calculate the distance between the simulated data X and the observed data Xobs.
    
    Args:
    X: Simulated data.
    Xobs: Observed data.

    Returns:
    A scalar representing the distance between X and Xobs.
    """
    # Implement the distance measure (e.g., Euclidean, Manhattan, etc.)
    return np.linalg.norm(X - Xobs)

def sample_from_prior(prior):
    """
    Sample a parameter set Y from the prior distribution.
    
    Args:
    prior: A function or distribution to sample from.

    Returns:
    A sample Y from the prior.
    """
    # Implement sampling from the prior distribution of parameters
    return prior()

def sample_noise(eta_distribution):
    """
    Sample noise eta from its distribution.
    
    Args:
    eta_distribution: The distribution to sample noise from.

    Returns:
    A sample eta from the noise distribution.
    """
    # Implement sampling from the noise distribution
    return eta_distribution()

def ABC(phi, prior, eta_distribution, Xobs, epsilon, N):
    """
    Approximate Bayesian Computation (ABC) algorithm.
    
    Args:
    phi: The simulation function.
    prior: The prior distribution to sample parameters from.
    eta_distribution: The noise distribution to sample from.
    Xobs: Observed data to compare against.
    epsilon: The acceptance threshold distance.
    N: The number of accepted samples to generate.

    Returns:
    A list of parameter samples Y that approximate the posterior distribution.
    """
    accepted_Y = []
    while len(accepted_Y) < N:
        Y = sample_from_prior(prior)
        eta = sample_noise(eta_distribution)
        X_sim = simulate(phi, Y, eta)
        if distance(X_sim, Xobs) <= epsilon:
            accepted_Y.append(Y)
    return accepted_Y

# Example usage:
# Define phi, prior, eta_distribution according to your problem.
# Xobs = np.array([...])  # Your observed data
# epsilon = 0.1  # Set an appropriate epsilon value
# N = 100  # Number of samples you want to generate
# accepted_samples = ABC(phi, prior, eta_distribution, Xobs, epsilon, N)
