import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Callable

import json

class Body:
    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        mass: torch.float64,
        radius: torch.float64,
    ):
        self.position: torch.Tensor = position
        self.velocity: torch.Tensor = velocity
        self.mass: torch.float64 = mass
        self.radius: torch.float64 = radius
        self.position_history: list[torch.Tensor] = []
        self.velocity_history: list[torch.Tensor] = []

    def update_history(self):
        """Save the current position and velocity to the history."""
        self.position_history.append(self.position.clone())
        self.velocity_history.append(self.velocity.clone())

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, mass={self.mass}, radius={self.radius})"

    def to_dict(self):
        """Convert the Body instance into a dictionary for JSON serialization."""
        return {
            "position": self.position.tolist(),  # Convert tensors to lists
            "velocity": self.velocity.tolist(),
            "mass": self.mass,  # Convert tensor scalar to Python scalar
            "radius": self.radius,
            "position_history": [p.tolist() for p in self.position_history],
            "velocity_history": [v.tolist() for v in self.velocity_history],
        }

    @staticmethod
    def from_dict(data):
        """Reconstruct a Body instance from a dictionary."""
        body = Body(
            position=torch.tensor(data["position"]),
            velocity=torch.tensor(data["velocity"]),
            mass=torch.tensor(data["mass"]),
            radius=torch.tensor(data["radius"]),
        )
        body.position_history = [torch.tensor(p) for p in data["position_history"]]
        body.velocity_history = [torch.tensor(v) for v in data["velocity_history"]]
        return body

    @staticmethod
    def save_bodies_to_json(bodies, file_name="bodies.json"):
        """
        Save the list of bodies, including their histories, to a JSON file.

        :param bodies: List of Body instances to be saved.
        :param file_name: Name of the file where the bodies will be saved.
        """
        with open(file_name, "w") as file:
            # Convert each Body instance to a dictionary and save
            json.dump([body.to_dict() for body in bodies], file, indent=4)

    @staticmethod
    def load_bodies_from_json(file_name="bodies.json"):
        """
        Load the list of bodies from a JSON file.

        :param file_name: Name of the file from which to load the bodies.
        :return: List of Body instances loaded from the file.
        """
        with open(file_name, "r") as file:
            bodies_data = json.load(file)
        return [Body.from_dict(body_dict) for body_dict in bodies_data]


class Variables:
    def __init__(
        self,
        masses: Tensor,
        radii: Tensor,
        starting_positions: Union[Tensor | None],
        # initial_velocities: Tensor,
        num_bodies: int,
        space_size: Tensor,
    ):
        self.masses = masses
        self.radii = radii
        self.num_bodies = num_bodies
        self.space_size = space_size
        self.starting_positions = starting_positions
        
class ElasticCollisionSimulation:
    def __init__(
        self,
        variables: Variables,
        enable_logging: bool = False,
    ):
        self.variables = variables
        self.enable_logging = enable_logging
        self.bodies = None
    
    
    @staticmethod
    def detect_collision(body_a: Body, body_b: Body) -> bool:
        distance = torch.linalg.norm(body_a.position - body_b.position)
        return distance < (body_a.radius + body_b.radius)

    @staticmethod
    def update_velocity_elastic(body_a: Body, body_b: Body):
        distance_vec = body_b.position - body_a.position
        unit_distance_vec = distance_vec / torch.linalg.norm(distance_vec)
        v_rel = body_b.velocity - body_a.velocity
        mass_sum = body_a.mass + body_b.mass

        # Update velocities
        body_a.velocity += (
            2
            * body_b.mass
            / mass_sum
            * (torch.dot(v_rel, unit_distance_vec))
            * unit_distance_vec
        )
        body_b.velocity -= (
            2
            * body_a.mass
            / mass_sum
            * (torch.dot(-v_rel, -unit_distance_vec))
            * unit_distance_vec
        )
        # optional: push bodies apart to avoid error in collision detection due to overlap
        separation_distance = (body_a.radius + body_b.radius) - torch.linalg.norm(
            body_a.position - body_b.position
        )
        if separation_distance > 0:
            # Push bodies apart by a fraction of the overlap
            move_distance = separation_distance * 0.01  # fraction of overlap
            body_a.position -= unit_distance_vec * move_distance
            body_b.position += unit_distance_vec * move_distance
    
    @staticmethod
    def construct_bodies(starting_velocities: torch.Tensor, variables: Variables):
        # construct bodies from variables and starting velocities
        bodies = []
        for i in range(variables.num_bodies):
            bodies.append(
                Body(
                    position=variables.starting_positions[i],
                    velocity=starting_velocities[i],
                    mass=variables.masses[i],
                    radius=variables.radii[i],
                )
            )
        return bodies
    
    @staticmethod
    def sample_initial_positions_without_overlap(variables: Variables, position_distribution: Distribution):
        
        accepted_positions = []
        for i in range(variables.num_bodies):
            
            valid_position = False
            while not valid_position:
                sample_position = position_distribution.sample((2,))
                
                # check boudary conditions
                if (
                    sample_position[0] - variables.radii[i] < 0 # bottom boundary
                    or sample_position[0] + variables.radii[i] > variables.space_size[0] # top boundary
                    or sample_position[1] - variables.radii[i] < 0 # left boundary
                    or sample_position[1] + variables.radii[i] > variables.space_size[1] # right boundary
                ):
                    continue
                
                # check overlap with other bodies
                collision_detected = False
                for j in range(len(accepted_positions)):
                    if torch.linalg.norm(sample_position - accepted_positions[j]) < variables.radii[i] + variables.radii[j]:
                        collision_detected = True
                        break
                
                if collision_detected: continue
                
                # if we reach here, the position is valid
                valid_position = True
                accepted_positions.append(sample_position)
        
        return torch.stack(accepted_positions)

    def logger(self, to_log):
        if self.enable_logging:
            print(to_log)
        else:
            pass
        
    def handle_boundary_collision(self, body: Body, dt: float):
        for dim in range(2):
            # If the body is going to move outside the space boundaries
            if (
                body.position[dim] - body.radius < 0
                or body.position[dim] + body.radius > self.variables.space_size[dim]
            ):
                # log collision with coordinates and time
                self.logger(
                    f"Collision detected between {body} and the boundary at time {dt}"
                )
                body.velocity[dim] = -body.velocity[
                    dim
                ]  # Reverse the velocity component

    def update(self, dt: float):
        # Detect and resolve collisions
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                if self.detect_collision(self.bodies[i], self.bodies[j]):
                    # log collision with coordinates and time
                    self.logger(
                        f"Collision detected between {self.bodies[i]} and {self.bodies[j]} at time {dt}"
                    )
                    self.update_velocity_elastic(self.bodies[i], self.bodies[j])

        # Update positions
        for body in self.bodies:
            self.handle_boundary_collision(body, dt)
            body.position += body.velocity * dt
            # Save the current position and velocity to the history
            body.update_history()
    
    def simulate(self, starting_velocities: torch.Tensor, total_time: float, dt: float):
        self.bodies = self.construct_bodies(starting_velocities, self.variables)
        
        # simulate
        num_steps = int(total_time / dt)
        for i in range(num_steps):
            self.update(dt)
        
        noise_distribution = torch.distributions.Normal(0, 0.5)
        body_positions = torch.stack([body.position for body in self.bodies])
        # apply noise to the final positions
        body_positions_noised = body_positions + noise_distribution.sample(body_positions.shape)
        return body_positions_noised
        
    def get_position_history(self):
        return [body.position_history for body in self.bodies]
    
    def get_velocity_history(self):
        return [body.velocity_history for body in self.bodies]
        

def distance(X: Tensor, X_obs: Tensor) -> Tensor:
    """
    Calculate the distance between the simulated data X and the observed data X_obs using PyTorch.
    :param X: Simulated final position as a **tensor of shape (2,).**
    :param X_obs: Observed final position as a **tensor of shape (2,).**
    :return: The distance between X and X_obs as a tensor **of shape (1,).**
    """
    return torch.linalg.norm(X - X_obs)


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
    while len(accepted_Y) < T:
        current_iteration += 1
        # if current_iteration % 10 == 0: print(f"Current iteration: {current_iteration}")
        Y = sample_from_prior(variables.num_bodies)
        X_sim = simulation.simulate(Y, total_time, dt)
        if distance(X_sim, X_obs) <= epsilon:
            accepted_Y.append(Y)
            accepted_X.append(X_sim)
    print(f"Total iterations: {current_iteration}")
    return torch.stack(accepted_Y), torch.stack(accepted_X)


if __name__ == "__main__":
    
    space_size = 10.0
    max_radius = space_size // 5.0
    constant_mass_value = 1.0
    constant_radius_value = max_radius
    
    
    velocity_distribution = torch.distributions.Uniform(low=-5.0, high=5.0)
    position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)

    # variant 1: sampling mass and radius from uniform distributions
    """
    mass_distribution = torch.distributions.Uniform(low=0.5, high=50.0)
    radius_distribution = torch.distributions.Uniform(low=0.5, high=max_radius)
    masses = mass_distribution.sample(sample_shape=torch.Size([num_bodies])),
    radii = radius_distribution.sample(sample_shape=torch.Size([num_bodies])),
    """

    num_bodies = 2
    VARIABLES = Variables(
        masses = torch.full((num_bodies,), constant_mass_value),
        radii = torch.full((num_bodies,), constant_radius_value),
        starting_positions = None,
        num_bodies = num_bodies,
        space_size = torch.tensor([space_size, space_size]),
    )

    initial_positions = ElasticCollisionSimulation.sample_initial_positions_without_overlap(VARIABLES, position_distribution)
    VARIABLES.starting_positions = initial_positions
    print(f"initial_positions: {initial_positions}")

    initial_velocities = velocity_distribution.sample(sample_shape=torch.Size([num_bodies, 2]))
    print(f"initial_velocities: {initial_velocities}")



    # TODO: final positions instead of initial_velocities
    accepted_Y, accepted_X = ABC_Algo(
        variables=VARIABLES,
        sample_from_prior= lambda amount: velocity_distribution.sample(sample_shape=([amount, 2])),
        X_obs = initial_velocities,
        epsilon = 1.0,
        T = 1,
        total_time = 1.0,
        dt = 0.1,
    )
    print(f"accepted_X: {accepted_X}")
    print(f"accepted_Y: {accepted_Y}")

