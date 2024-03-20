// use rand::{Rng, thread_rng};
// use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use serde_json;

fn sample_initial_positions(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    radii: &Vec<f64>,
    positions_distribution: impl Distribution<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();
    let mut positions_x: Vec<f64> = Vec::new();
    let mut positions_y: Vec<f64> = Vec::new();

    for i in 0..num_bodies {
        let mut overlap: bool = true;
        let mut boundary: bool = true;
        let mut position_x: f64 = 0.0;
        let mut position_y: f64 = 0.0;

        while overlap || boundary {
            overlap = false;
            boundary = false;

            // sample new position
            position_x = positions_distribution.sample(&mut rng);
            position_y = positions_distribution.sample(&mut rng);

            for j in 0..positions_x.len() {
                let distance: f64 = ((position_x - positions_x[j]).powi(2)
                    + (position_y - positions_y[j]).powi(2))
                .sqrt();

                if distance < radii[i] + radii[j] {
                    overlap = true;
                    break;
                }
            }

            if position_x - radii[i] < 0.0 // bottom boundary
                || position_x + radii[i] > space_size_x // top boundary
                || position_y - radii[i] < 0.0 // left boundary
                || position_y + radii[i] > space_size_y
            // right boundary
            {
                boundary = true;
            }
        }
        positions_x.push(position_x);
        positions_y.push(position_y);
    }
    return (positions_x, positions_y);
}

// for now elastic collisions
fn update_velocities(num_bodies: usize, current_state: &mut SimState) {
    for i in 0..num_bodies {
        for j in i + 1..num_bodies {
            let distance: f64 = ((current_state.positions_x[i] - current_state.positions_x[j])
                .powi(2)
                + (current_state.positions_y[i] - current_state.positions_y[j]).powi(2))
            .sqrt();

            if distance <= current_state.radii[i] + current_state.radii[j] {
                current_state.body_collisions.push((i, j));
                let normal_x: f64 =
                    (current_state.positions_x[i] - current_state.positions_x[j]) / distance;
                let normal_y: f64 =
                    (current_state.positions_y[i] - current_state.positions_y[j]) / distance;

                // let tangent_x: f64 = -normal_y; // not needed for elastic collisions
                // let tangent_y: f64 = normal_x; // not needed for elastic collisions

                let relative_velocity_x: f64 =
                    current_state.velocities_x[i] - current_state.velocities_x[j];
                let relative_velocity_y: f64 =
                    current_state.velocities_y[i] - current_state.velocities_y[j];

                let dot_product: f64 =
                    relative_velocity_x * normal_x + relative_velocity_y * normal_y;

                if dot_product < 0.0 {
                    let mass_sum: f64 = current_state.masses[i] + current_state.masses[j];
                    let mass_difference: f64 = current_state.masses[i] - current_state.masses[j];

                    let new_velocity_i_x: f64 = (mass_difference * current_state.velocities_x[i]
                        + 2.0 * current_state.masses[j] * current_state.velocities_x[j])
                        / mass_sum;
                    let new_velocity_i_y: f64 = (mass_difference * current_state.velocities_y[i]
                        + 2.0 * current_state.masses[j] * current_state.velocities_y[j])
                        / mass_sum;

                    let new_velocity_j_x: f64 =
                        (2.0 * current_state.masses[i] * current_state.velocities_x[i]
                            - mass_difference * current_state.velocities_x[j])
                            / mass_sum;
                    let new_velocity_j_y: f64 =
                        (2.0 * current_state.masses[i] * current_state.velocities_y[i]
                            - mass_difference * current_state.velocities_y[j])
                            / mass_sum;

                    current_state.velocities_x[i] = new_velocity_i_x;
                    current_state.velocities_y[i] = new_velocity_i_y;
                    current_state.velocities_x[j] = new_velocity_j_x;
                    current_state.velocities_y[j] = new_velocity_j_y;
                }
            }
        }
    }
}

fn handle_boundary_collisions(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    current_state: &mut SimState,
) {
    for i in 0..num_bodies {
        // left boundary
        if current_state.positions_x[i] - current_state.radii[i] < 0.0 {
            current_state.positions_x[i] = current_state.radii[i];
            current_state.velocities_x[i] = -current_state.velocities_x[i];
            current_state.boundary_collisions.push((i, 2));
        // right boundary
        } else if current_state.positions_x[i] + current_state.radii[i] > space_size_x {
            current_state.positions_x[i] = space_size_x - current_state.radii[i];
            current_state.velocities_x[i] = -current_state.velocities_x[i];
            current_state.boundary_collisions.push((i, 3));
        }
        // bottom boundary
        if current_state.positions_y[i] - current_state.radii[i] < 0.0 {
            current_state.positions_y[i] = current_state.radii[i];
            current_state.velocities_y[i] = -current_state.velocities_y[i];
            current_state.boundary_collisions.push((i, 0));
        // top boundary
        } else if current_state.positions_y[i] + current_state.radii[i] > space_size_y {
            current_state.positions_y[i] = space_size_y - current_state.radii[i];
            current_state.velocities_y[i] = -current_state.velocities_y[i];
            current_state.boundary_collisions.push((i, 1));
        }
    }
}

fn update(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    time_step: f64,
    current_state: &mut SimState,
) {
    // update positions based on velocities
    for i in 0..num_bodies {
        current_state.positions_x[i] += current_state.velocities_x[i] * time_step;
        current_state.positions_y[i] += current_state.velocities_y[i] * time_step;
    }

    // handle collisions between bodies
    update_velocities(num_bodies, current_state);

    // handle collisions with boundaries
    handle_boundary_collisions(num_bodies, space_size_x, space_size_y, current_state);
}

// currently without noise
#[derive(Clone, Serialize, Deserialize)]
struct SimState {
    positions_x: Vec<f64>,
    positions_y: Vec<f64>,
    velocities_x: Vec<f64>,
    velocities_y: Vec<f64>,
    masses: Vec<f64>,
    radii: Vec<f64>,
    body_collisions: Vec<(usize, usize)>,
    boundary_collisions: Vec<(usize, usize)>, // collisions[0] = body index, collisions[1] = corresponding boundary index (0: bottom, 1: top, 2: left, 3: right)
}

fn simulate(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    total_time: f64,
    time_step: f64,
    initial_state: SimState,
) -> Vec<SimState> {
    let mut sim_state_history: Vec<SimState> = Vec::new();
    sim_state_history.push(initial_state.clone());
    let mut current_state: SimState = initial_state.clone();
    let mut current_time: f64 = 0.0;
    while current_time < total_time {
        update(
            num_bodies,
            space_size_x,
            space_size_y,
            time_step,
            &mut current_state,
        );
        current_time += time_step;
        sim_state_history.push(current_state.clone());
    }
    return sim_state_history;
}

// return values of ABC function
struct SimulationData {
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    total_time: f64,
    time_step: f64,
    state_history: Vec<Vec<SimState>>,
}
struct AbcData {
    errors_values: Vec<f64>,
    num_iterations: usize,
}

// ABC algorithm
// modify to take parameters as input
fn run_abc(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    total_time: f64,
    time_step: f64,
    velocity_distribution: &impl Distribution<f64>,
    radius_distribution: &impl Distribution<f64>,
    mass_distribution: &impl Distribution<f64>,
    position_distribution: &impl Distribution<f64>,
    n: usize,
    epsilon: f64,
    include_velocities_in_error: bool,
    initial_state: SimState,
) -> (SimulationData, SimulationData, AbcData) {
    let mut rng = rand::thread_rng();

    let goal_states: Vec<SimState> = simulate(
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        initial_state.clone(),
    );

    let mut accepted_states: Vec<Vec<SimState>> = Vec::new();
    let mut num_accepted_states: usize = 0;
    let mut accepted_error_values: Vec<f64> = Vec::new();

    let mut current_iteration: usize = 0;

    println!("starting abc loop");
    while num_accepted_states < n {
        // build new state
        let mut current_state: SimState = initial_state.clone();

        /* for sampling velocities
        // sample new velocities
        current_state.velocities_x = velocity_distribution
            .sample_iter(&mut rng)
            .take(num_bodies)
            .collect();
        current_state.velocities_y = velocity_distribution
            .sample_iter(&mut rng)
            .take(num_bodies)
            .collect();
        */

        // for sampling masses
        current_state.masses = mass_distribution
            .sample_iter(&mut rng)
            .take(num_bodies)
            .collect();
        // set first mass to original mass
        current_state.masses[0] = initial_state.masses[0];

        let simulated_states: Vec<SimState> = simulate(
            num_bodies,
            space_size_x,
            space_size_y,
            total_time,
            time_step,
            current_state,
        );

        // calculate error
        let mut error: f64 = 0.0;

        let mut positions_error: f64 = 0.0;
        for i in 0..num_bodies {
            positions_error += (goal_states.last().unwrap().positions_x[i]
                - simulated_states.last().unwrap().positions_x[i])
                .powi(2);
            positions_error += (goal_states.last().unwrap().positions_y[i]
                - simulated_states.last().unwrap().positions_y[i])
                .powi(2);
        }
        positions_error /= num_bodies as f64;
        positions_error = positions_error.sqrt();
    

        let mut velocities_error: f64 = 0.0;
        if include_velocities_in_error {
            for i in 0..num_bodies {
                velocities_error += (goal_states.last().unwrap().velocities_x[i]
                    - simulated_states.last().unwrap().velocities_x[i])
                    .powi(2);
                velocities_error += (goal_states.last().unwrap().velocities_y[i]
                    - simulated_states.last().unwrap().velocities_y[i])
                    .powi(2);
            }
            velocities_error /= num_bodies as f64;
            velocities_error = velocities_error.sqrt();
        }

        if include_velocities_in_error {
            // average of positions and velocities error
            error = 0.5 * (positions_error + velocities_error);
        } else {
            // only positions error
            error = positions_error;
        }

        // if the error is less than epsilon, accept the state
        if error < epsilon {
            num_accepted_states += 1;
            accepted_states.push(simulated_states);
            // append error value
            accepted_error_values.push(error);
        }
        current_iteration += 1;
    }

    let original_states_simulation_data = SimulationData {
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        state_history: vec![goal_states],
    };

    let accepted_states_simulation_data = SimulationData {
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        state_history: accepted_states,
    };

    let general_abc_data = AbcData {
        errors_values: accepted_error_values,
        num_iterations: current_iteration,
    };

    return (
        original_states_simulation_data,
        accepted_states_simulation_data,
        general_abc_data,
    );
}

fn get_initial_state(
    velocity_distribution: &impl Distribution<f64>,
    radius_distribution: &impl Distribution<f64>,
    mass_distribution: &impl Distribution<f64>,
    position_distribution: &impl Distribution<f64>,
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
) -> SimState {
    let mut rng = rand::thread_rng();
    let initial_velocities_x: Vec<f64> = velocity_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let initial_velocities_y: Vec<f64> = velocity_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let initial_radii: Vec<f64> = radius_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let initial_masses: Vec<f64> = mass_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();

    // initialize positions
    let (initial_positions_x, initial_positions_y) = sample_initial_positions(
        num_bodies,
        space_size_x,
        space_size_y,
        &initial_radii,
        position_distribution,
    );

    return SimState {
        positions_x: initial_positions_x,
        positions_y: initial_positions_y,
        velocities_x: initial_velocities_x,
        velocities_y: initial_velocities_y,
        masses: initial_masses,
        radii: initial_radii,
        body_collisions: Vec::new(),
        boundary_collisions: Vec::new(),
    };
}

// use struct simulation data
fn write_simulation_data_to_json(file_name: &str, simulation_data: &SimulationData) {
    let data = serde_json::json!({
        "num_bodies": simulation_data.num_bodies,
        "space_size_x": simulation_data.space_size_x,
        "space_size_y": simulation_data.space_size_y,
        "total_time": simulation_data.total_time,
        "time_step": simulation_data.time_step,
        "state_history": simulation_data.state_history,
    });

    // pretty
    let data_string = serde_json::to_string_pretty(&data).unwrap();
    // not pretty
    // let data_string = serde_json::to_string(&data_string).unwrap();
    std::fs::write(file_name, data_string).unwrap();
}

fn main() {
    let num_bodies = 4;
    let space_size_x = 10.0;
    let space_size_y = 10.0;
    let total_time = 40.0;
    let time_step = 0.01;
    // let velocity_distribution = Normal::new(0.0, 5.0).unwrap();
    let velocity_distribution = Uniform::new(-5.0, 5.0);
    let radius_distribution = Uniform::new(1.5, 1.6);
    let mass_distribution = Uniform::new(1.0, 50.0);
    // let position_distribution = Normal::new(space_size_x/2.0, space_size_x/2.0).unwrap();
    let position_distribution = Uniform::new(0.0, space_size_x);
    let n = 1;
    let epsilon = 1.0;
    let include_velocities_in_error = false;

    let initial_state = get_initial_state(
        &velocity_distribution,
        &radius_distribution,
        &mass_distribution,
        &position_distribution,
        num_bodies,
        space_size_x,
        space_size_y,
    );

    let time = std::time::Instant::now();
    let (original_states_simulation_data, accepted_states_simulation_data, abc_data) = run_abc(
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        &velocity_distribution,
        &radius_distribution,
        &mass_distribution,
        &position_distribution,
        n,
        epsilon,
        include_velocities_in_error,
        initial_state,
    );
    let elapsed = time.elapsed();

    write_simulation_data_to_json(
        "data/original_simulation_data.json",
        &original_states_simulation_data,
    );
    write_simulation_data_to_json(
        "data/accepted_simulation_data.json",
        &accepted_states_simulation_data,
    );

    println!("total iterations: {}", abc_data.num_iterations);
    println!("time: {:?}", elapsed);
    // time per iteration
    println!(
        "time per iteration: {:?}",
        elapsed / abc_data.num_iterations as u32
    );
    // in seconds
    println!(
        "time per iteration in seconds: {:?}",
        elapsed.as_secs_f64() / abc_data.num_iterations as f64
    );

    // some print statements to check the results
    println!(
        "original states - initial positions x: {:?}",
        original_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .positions_x
    );
    println!(
        "original states - initial positions y: {:?}",
        original_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .positions_y
    );
    println!(
        "accepted states - initial positions x: {:?}",
        accepted_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .positions_x
    );
    println!(
        "accepted states - initial positions y: {:?}",
        accepted_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .positions_y
    );
    println!(
        "original states - initial velocities x: {:?}",
        original_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .velocities_x
    );
    println!(
        "original states - initial velocities y: {:?}",
        original_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .velocities_y
    );
    println!(
        "accepted states - initial velocities x: {:?}",
        accepted_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .velocities_x
    );
    println!(
        "accepted states - initial velocities y: {:?}",
        accepted_states_simulation_data.state_history[0]
            .first()
            .unwrap()
            .velocities_y
    );
    println!(
        "original states - final positions x: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .positions_x
    );
    println!(
        "original states - final positions y: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .positions_y
    );
    println!(
        "accepted states - final positions x: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .positions_x
    );
    println!(
        "accepted states - final positions y: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .positions_y
    );
    println!(
        "original states - final velocities x: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .velocities_x
    );
    println!(
        "original states - final velocities y: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .velocities_y
    );
    println!(
        "accepted states - final velocities x: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .velocities_x
    );
    println!(
        "accepted states - final velocities y: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .velocities_y
    );
    println!(
        "original states - masses: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .masses
    );
    println!(
        "accepted states - masses: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .masses
    );
    // print normalized masses original states
    let mut normalized_masses: Vec<f64> = original_states_simulation_data.state_history[0]
        .last()
        .unwrap()
        .masses
        .clone();
    // let sum: f64 = normalized_masses.iter().sum();
    let first_mass: f64 = normalized_masses[0];
    normalized_masses.iter_mut().for_each(|x| *x /= first_mass);
    println!(
        "original states - div by first, masses: {:?}",
        normalized_masses
    );
    // print normalized masses accepted states\
    normalized_masses = accepted_states_simulation_data.state_history[0]
        .last()
        .unwrap()
        .masses
        .clone();
    // let sum: f64 = normalized_masses.iter().sum();
    let first_mass: f64 = normalized_masses[0];
    normalized_masses.iter_mut().for_each(|x| *x /= first_mass);
    println!(
        "accepted states - div by first, masses: {:?}",
        normalized_masses
    );

    println!("accepted error values: {:?}", abc_data.errors_values);
    // to get the amount of collisions the simulations
    println!(
        "original states - body collisions: {:?}",
        original_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .body_collisions
            .len()
    );
    println!(
        "accepted states - body collisions: {:?}",
        accepted_states_simulation_data.state_history[0]
            .last()
            .unwrap()
            .body_collisions
            .len()
    );
}
