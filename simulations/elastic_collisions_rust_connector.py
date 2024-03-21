from typing import List, Tuple
from pydantic import BaseModel
import json
from pydantic import ValidationError

class SimState(BaseModel):
    positions_x: List[float]
    positions_y: List[float]
    velocities_x: List[float]
    velocities_y: List[float]
    masses: List[float]
    radii: List[float]
    body_collisions: List[Tuple[int, int]]
    boundary_collisions: List[Tuple[int, int]]

class SimulationData(BaseModel):
    num_bodies: int
    space_size_x: float
    space_size_y: float
    total_time: float
    time_step: float
    state_history: List[List[SimState]]

def parse_rust_simulation_data_from_string(json_str: str) -> SimulationData:
    simulation_data = SimulationData.parse_raw(json_str)
    return simulation_data

def load_rust_simulation_data_from_json(json_filepath: str) -> SimulationData:
    with open(json_filepath, 'r') as json_file:
        json_str = json_file.read()
        return parse_rust_simulation_data_from_string(json_str)


if __name__ == "__main__":
    sim_data: SimulationData = load_rust_simulation_data_from_json("sbi_algorithms_rust/abc/data/original_simulation_data.json")
    print(sim_data.num_bodies)
    