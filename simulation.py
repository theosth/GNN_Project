from abc import ABC, abstractmethod
import torch
import random


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


class Simulation(ABC):

    def __init__(
        self,
        space_size: torch.Tensor = torch.tensor([100, 100]),
        enable_logging: bool = False,
    ):
        self.space_size: torch.Tensor = space_size
        self.bodies = self.initialize_bodies()
        self.enable_logging: bool = enable_logging

    @abstractmethod
    def initialize_bodies(self) -> list[Body]:
        """
        Initialize the bodies (particles, balls, etc.) involved in the simulation.
        This should return a list of `Body` instances.
        """
        pass

    @abstractmethod
    def handle_boundary_collision(self, body: Body, dt: float):
        """
        Handle the collision of a body with the boundaries of the space.
        """
        pass

    @abstractmethod
    def update(self, dt: float):
        """
        Update the simulation by a time step `dt`.
        This should include moving the bodies and handling collisions.
        """
        pass

    @staticmethod
    @abstractmethod
    def detect_collision(body_a: Body, body_b: Body) -> bool:
        """
        Determine if two bodies collide.
        """
        pass

    @staticmethod
    @abstractmethod
    def update_velocity_elastic(body_a: Body, body_b: Body):
        """
        Update the velocities of two bodies after an elastic collision.
        """
        pass

    @abstractmethod
    def run_simulation(self, total_time: float, dt: float):
        """
        Run the simulation for a total time of `total_time` with time steps of `dt`.
        """
        pass

    @abstractmethod
    def logger(self, to_log):
        """
        Log the simulation
        """
        pass


class ElasticCollisionSimulation(Simulation):

    def initialize_bodies(
        self,
        num_bodies=10,
        min_radius=0.5,
        max_radius=100.0,
        max_initial_speed=5,
    ) -> list[Body]:
        bodies = []
        for _ in range(num_bodies):
            valid_position = False
            while not valid_position:
                radius = random.uniform(
                    min_radius, max_radius
                )  # Random radius within specified range
                position = torch.tensor(
                    [
                        random.uniform(
                            radius, self.space_size[0] - radius
                        ),  # Ensure body is fully within bounds
                        random.uniform(radius, self.space_size[1] - radius),
                    ]
                )
                # Check for overlap with existing bodies
                overlap = any(
                    torch.linalg.norm(position - other_body.position)
                    < (radius + other_body.radius)
                    for other_body in bodies
                )
                if not overlap:
                    valid_position = True
                    velocity = torch.tensor(
                        [
                            random.uniform(-max_initial_speed, max_initial_speed),
                            random.uniform(-max_initial_speed, max_initial_speed),
                        ]
                    )
                    bodies.append(
                        Body(
                            position=position,
                            velocity=velocity,
                            mass=radius,
                            radius=radius,
                        )
                    )
        return bodies

    def handle_boundary_collision(self, body: Body, dt: float):
        for dim in range(2):
            # If the body is going to move outside the space boundaries
            if (
                body.position[dim] - body.radius < 0
                or body.position[dim] + body.radius > self.space_size[dim]
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
        separation_distance = (body_a.radius + body_b.radius) - torch.linalg.norm(body_a.position - body_b.position)
        if separation_distance > 0:
            # Push bodies apart by a fraction of the overlap
            move_distance = separation_distance * 0.01  # fraction of overlap
            body_a.position -= unit_distance_vec * move_distance
            body_b.position += unit_distance_vec * move_distance

    def run_simulation(self, total_time: float, dt: float):
        num_steps = int(total_time / dt)
        for _ in range(num_steps):
            self.update(dt)

    def logger(self, to_log):
        if self.enable_logging:
            print(to_log)
        else:
            pass


if __name__ == "__main__":
    import pygame
    import sys

    # Initialize Pygame
    pygame.init()

    # Fixed window size
    window_size = (800, 800)  # Set a fixed window size

    # Simulation space size
    space_size = (1000, 1000)  # This can vary

    # Dynamically compute the scale factor to fit the simulation space into the window
    scale_factor_x = window_size[0] / space_size[0]
    scale_factor_y = window_size[1] / space_size[1]
    scale_factor = min(scale_factor_x, scale_factor_y)  # Ensure aspect ratio is maintained

    # Set up the display
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Elastic Collision Simulation")

    # Define colors
    background_color = (0, 0, 0)
    ball_color = (255, 255, 255)

    # Main animation loop
    clock = pygame.time.Clock()
    running = True
    simulation_time_step = 1.0  # Corresponds to the dt in your simulation
    frame_rate = 100  # Frames per second

    # Create simulation instance
    simulation = ElasticCollisionSimulation(space_size=torch.tensor(space_size), enable_logging=False)

    # Run the simulation separately to gather all positions before animating
    simulation.run_simulation(total_time=10000.0, dt=simulation_time_step)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(background_color)

        # Draw the balls
        for body in simulation.bodies:
            if len(body.position_history) > 0:
                # Get the next position from the history and remove it
                position = body.position_history.pop(0).numpy()
                # Scale the position for the display
                scaled_position = (int(position[0] * scale_factor), int(position[1] * scale_factor))
                # Draw the ball
                pygame.draw.circle(screen, ball_color, scaled_position, int(body.radius * scale_factor))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(frame_rate)

    pygame.quit()
    sys.exit()

