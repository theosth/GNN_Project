import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Callable

from copy import deepcopy

import json

class Body:
    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration_coefficient: torch.Tensor,
        mass: torch.float64,
        radius: torch.float64,
    ):
        self.position: torch.Tensor = position
        self.velocity: torch.Tensor = velocity
        self.acceleration_coefficient: torch.Tensor = acceleration_coefficient
        self.mass: torch.float64 = mass
        self.radius: torch.float64 = radius
        self.position_history: list[torch.Tensor] = []
        self.velocity_history: list[torch.Tensor] = []
        self._id = None

    def set_id(self, _id: int):
        self._id = _id

    def get_id(self) -> int | None:
        return self._id

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
            "acceleration_coefficient": self.acceleration_coefficient,
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
            acceleration_coefficient=torch.tensor(data["acceleration_coefficient"]),
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
        masses: Union[Tensor | None],
        radii: Union[Tensor | None],
        acceleration_coefficients: Union[Tensor | None],
        starting_positions: Union[Tensor | None],
        initial_velocities: Union[Tensor | None],
        num_bodies: int | None,
        space_size: Tensor,
    ):
        self.masses = masses
        self.radii = radii
        self.num_bodies = num_bodies
        self.space_size = space_size
        self.starting_positions = starting_positions
        self.acceleration_coefficients = acceleration_coefficients
        self.initial_velocities = initial_velocities
        

class HiddenVariables:
    def __init__(
        self,
        num_bodies: int | None,
        masses: Union[Tensor | None],
        radii: Union[Tensor | None],
        acceleration_coefficients: Union[Tensor | None],
        initial_velocities: Union[Tensor | None],
       
    ):
        self.masses = masses
        self.radii = radii
        self.num_bodies = num_bodies
        self.acceleration_coefficients = acceleration_coefficients
        self.initial_velocities = initial_velocities
        
class ElasticCollisionSimulation:

    SEPARATION_COEFFICIENT = 0.05

    def __init__(
        self,
        variables: Variables,
        enable_logging: bool = False,
        noise: bool = False,
    ):
        self.variables = variables
        self.enable_logging = enable_logging
        self.bodies = None
        self.noise = noise
        self.collision_history_per_timestep = {}
    
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

        # alternative way: ignore one time step for this body pair 
        # optional: push bodies apart to avoid error in collision detection due to overlap
        separation_distance = (body_a.radius + body_b.radius) - torch.linalg.norm(
            body_a.position - body_b.position
        )
        if separation_distance > 0:
            # Push bodies apart by a fraction of the overlap
            move_distance = separation_distance * ElasticCollisionSimulation.SEPARATION_COEFFICIENT  # fraction of overlap
            body_a.position -= unit_distance_vec * move_distance
            body_b.position += unit_distance_vec * move_distance
    
    @staticmethod
    def construct_bodies(hidden_variables: HiddenVariables, variables: Variables):
        # construct bodies from variables and starting velocities

        if not hidden_variables.num_bodies == None:
            num_bodies = hidden_variables.num_bodies
        elif not variables.num_bodies == None:
            num_bodies = variables.num_bodies
        else:
            raise ValueError("Number of bodies not specified")

        if not hidden_variables.masses == None:
            masses = hidden_variables.masses
        elif not variables.masses == None:
            masses = variables.masses
        else:
            raise ValueError("Masses not specified")

        if not hidden_variables.radii == None:
            radii = hidden_variables.radii
        elif not variables.radii == None:
            radii = variables.radii
        else:
            raise ValueError("Radii not specified")

        if not hidden_variables.acceleration_coefficients == None:
            acceleration_coefficients = hidden_variables.acceleration_coefficients
        elif not variables.acceleration_coefficients == None:
            acceleration_coefficients = variables.acceleration_coefficients
        else:
            raise ValueError("Acceleration coefficients not specified")

        if not hidden_variables.initial_velocities == None:
            initial_velocities = hidden_variables.initial_velocities
        elif not variables.initial_velocities == None:
            initial_velocities = variables.initial_velocities
        else:
            raise ValueError("Initial velocities not specified")
        
        if variables.starting_positions == None:
            raise ValueError("Starting positions not specified")

        bodies = []
        for i in range(num_bodies):
            body = Body(
                    position=variables.starting_positions[i],
                    velocity=initial_velocities[i],
                    acceleration_coefficient=acceleration_coefficients[i],
                    mass=masses[i],
                    radius=radii[i],
                   )
            body.set_id(i)
            bodies.append(body)
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
        
        return torch.stack(deepcopy(accepted_positions))

    def logger(self, to_log):
        if self.enable_logging:
            print(to_log)
        else:
            pass
        
    def handle_boundary_collision(self, body: Body, dt: float, dt_i: int):
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
                self.collision_history_per_timestep[int(dt_i)].append({"time": dt*int(dt_i), 
                                                                       "body1": body.get_id(), 
                                                                       "boundary": dim})
                body.velocity[dim] = -body.velocity[
                    dim
                ]  # Reverse the velocity component

    def update(self, dt: float, dt_i: int):
        # Detect and resolve collisions

        self.collision_history_per_timestep[int(dt_i)] = []

        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                if self.detect_collision(self.bodies[i], self.bodies[j]):
                    # log collision with coordinates and time
                    self.logger(
                        f"Collision detected between {self.bodies[i]} and {self.bodies[j]} at time {dt}"
                    )
                    self.collision_history_per_timestep[int(dt_i)].append({"time": dt*int(dt_i), 
                                                                      "body1": self.bodies[i].get_id(), 
                                                                      "body2": self.bodies[j].get_id()})
                    
                    self.update_velocity_elastic(self.bodies[i], self.bodies[j])

        # Update positions
        body_i = 0
        for body in self.bodies:
            self.handle_boundary_collision(body, dt, dt_i)
            body.position += body.velocity * dt
            # Save the current position and velocity to the history
            body.update_history()
            body_i += 1
    
    def simulate(self, hidden_variables: HiddenVariables, total_time: float, dt: float):

        self.bodies = self.construct_bodies(hidden_variables, self.variables)
        # simulate
        num_steps = int(total_time / dt)

        #self.time_steps = num_steps
        #self.dt = dt
        for i in range(num_steps):
            self.update(dt,i)

        body_positions = torch.stack([body.position for body in self.bodies])

        if self.noise:
            noise_distribution = torch.distributions.Normal(0, 0.5)
            body_positions = body_positions + noise_distribution.sample(body_positions.shape)
        
        return body_positions
        
    def get_position_history(self):
        return [body.position_history for body in self.bodies]
    
    def get_velocity_history(self):
        return [body.velocity_history for body in self.bodies]
    
    def get_collision_history_per_timestep(self):
        return deepcopy(self.collision_history_per_timestep)
            
