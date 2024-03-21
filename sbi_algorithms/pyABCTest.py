import numpy as np
import matplotlib.pyplot as plt

from pyabc import *

settings.set_figure_params("pyabc")  # for beautified plots

rng = np.random.default_rng()

from simulations.elastic_collisions import (
    Body,
    Variables,
    ElasticCollisionSimulation,
    HiddenVariables,
)
import torch

# simulation parameters
total_time = 1.0
dt = 0.1
space_size = 10.0
max_radius = space_size // 10.0
constant_mass_value = 1.0
constant_radius_value = max_radius
velocity_distribution = torch.distributions.Uniform(low=-5.0, high=5.0)
position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)
num_bodies = 2
VARIABLES = Variables(
    masses=torch.full((num_bodies,), constant_mass_value),
    radii=torch.full((num_bodies,), constant_radius_value),
    starting_positions=None,
    num_bodies=num_bodies,
    space_size=torch.tensor([space_size, space_size]),
    acceleration_coefficients=torch.Tensor([1.0] * num_bodies),
    initial_velocities=None,
)

initial_positions = ElasticCollisionSimulation.sample_initial_positions_without_overlap(
    VARIABLES, position_distribution
)
VARIABLES.starting_positions = initial_positions
print(f"initial_positions: {initial_positions}")

initial_velocities = velocity_distribution.sample(
    sample_shape=torch.Size([num_bodies, 2])
)
print(f"initial_velocities: {initial_velocities}")

SIMULATION = ElasticCollisionSimulation(VARIABLES, enable_logging=False, noise=False)

hidden_variables = HiddenVariables(
    masses=None,
    radii=None,
    num_bodies=None,
    acceleration_coefficients=None,
    initial_velocities=initial_velocities,
)

result = SIMULATION.simulate(
    hidden_variables=hidden_variables, total_time=total_time, dt=dt
)

print(f"result: {result}")


def model(p):
    starting_velocities = torch.tensor(
        [[p["X0"], p["Y0"]], [p["X1"], p["Y1"]]], dtype=torch.float32
    )
    X_Obs = SIMULATION.simulate(
        hidden_variables=HiddenVariables(
            masses=VARIABLES.masses,
            radii=VARIABLES.radii,
            num_bodies=VARIABLES.num_bodies,
            acceleration_coefficients=None,
            initial_velocities=starting_velocities,
        ),
        total_time=total_time,
        dt=dt,
    )
    # convert to numpy array
    X_Obs = X_Obs.detach().numpy()
    return {"X_Obs": X_Obs}


# true parameters
true_parameters = {
    "X0": initial_velocities[0, 0].item(),
    "Y0": initial_velocities[0, 1].item(),
    "X1": initial_velocities[1, 0].item(),
    "Y1": initial_velocities[1, 1].item(),
}

# prior
prior = Distribution(
    X0=RV("uniform", -5, 5),
    Y0=RV("uniform", -5, 5),
    X1=RV("uniform", -5, 5),
    Y1=RV("uniform", -5, 5),
)

# observed data
# X_Obs = result.detach().numpy()
# without np
X_Obs = {"X_Obs": result.detach().numpy()}


# distance function
def distance(x, y):
    return np.linalg.norm(x["X_Obs"] - y["X_Obs"])


# parameter bounds
prior_bounds = {
    "X0": (-5, 5),
    "Y0": (-5, 5),
    "X1": (-5, 5),
    "Y1": (-5, 5),
}
# ABCSMC
abc = ABCSMC(model, prior, distance, population_size=100)
abc.new(create_sqlite_db_id(), X_Obs)
history = abc.run(minimum_epsilon=0.1, max_nr_populations=5)

# get the best parameter
df, w = history.get_distribution()


visualization.plot_kde_matrix_highlevel(
    history,
    refval=true_parameters,
)
plt.savefig("pyabc.png")