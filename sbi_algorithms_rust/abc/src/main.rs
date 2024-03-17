// use rand::{Rng, thread_rng};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use serde_json;

fn sample_initial_positions(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    radii: &Vec<f64>,
    positions_distribution: &Uniform<f64>,
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
fn update_velocities(
    num_bodies: usize,
    velocities_x: &mut Vec<f64>,
    velocities_y: &mut Vec<f64>,
    positions_x: &Vec<f64>,
    positions_y: &Vec<f64>,
    masses: &Vec<f64>,
    radii: &Vec<f64>,
    collisions: &mut Vec<(usize, usize)>,
) {
    for i in 0..num_bodies {
        for j in i + 1..num_bodies {
            let distance: f64 = ((positions_x[i] - positions_x[j]).powi(2)
                + (positions_y[i] - positions_y[j]).powi(2))
            .sqrt();

            if distance <= radii[i] + radii[j] {
                collisions.push((i, j));
                let normal_x: f64 = (positions_x[i] - positions_x[j]) / distance;
                let normal_y: f64 = (positions_y[i] - positions_y[j]) / distance;

                // let tangent_x: f64 = -normal_y; // not needed for elastic collisions
                // let tangent_y: f64 = normal_x; // not needed for elastic collisions

                let relative_velocity_x: f64 = velocities_x[i] - velocities_x[j];
                let relative_velocity_y: f64 = velocities_y[i] - velocities_y[j];

                let dot_product: f64 =
                    relative_velocity_x * normal_x + relative_velocity_y * normal_y;

                if dot_product < 0.0 {
                    let mass_sum: f64 = masses[i] + masses[j];
                    let mass_difference: f64 = masses[i] - masses[j];

                    let new_velocity_i_x: f64 = (mass_difference * velocities_x[i]
                        + 2.0 * masses[j] * velocities_x[j])
                        / mass_sum;
                    let new_velocity_i_y: f64 = (mass_difference * velocities_y[i]
                        + 2.0 * masses[j] * velocities_y[j])
                        / mass_sum;

                    let new_velocity_j_x: f64 = (2.0 * masses[i] * velocities_x[i]
                        - mass_difference * velocities_x[j])
                        / mass_sum;
                    let new_velocity_j_y: f64 = (2.0 * masses[i] * velocities_y[i]
                        - mass_difference * velocities_y[j])
                        / mass_sum;

                    velocities_x[i] = new_velocity_i_x;
                    velocities_y[i] = new_velocity_i_y;
                    velocities_x[j] = new_velocity_j_x;
                    velocities_y[j] = new_velocity_j_y;
                }
            }
        }
    }
}

fn handle_boundary_collisions(
    num_bodies: usize,
    positions_x: &mut Vec<f64>,
    positions_y: &mut Vec<f64>,
    velocities_x: &mut Vec<f64>,
    velocities_y: &mut Vec<f64>,
    radii: &Vec<f64>,
    space_size_x: f64,
    space_size_y: f64,
    boundary_collisions: &mut Vec<(usize, usize)>, // collisions[0] = body index, collisions[1] = corresponding boundary index (0: bottom, 1: top, 2: left, 3: right)
) {
    for i in 0..num_bodies {
        // left boundary
        if positions_x[i] - radii[i] < 0.0 {
            positions_x[i] = radii[i];
            velocities_x[i] = -velocities_x[i];
            boundary_collisions.push((i, 2));
        // right boundary
        } else if positions_x[i] + radii[i] > space_size_x {
            positions_x[i] = space_size_x - radii[i];
            velocities_x[i] = -velocities_x[i];
            boundary_collisions.push((i, 3));
        }
        // bottom boundary
        if positions_y[i] - radii[i] < 0.0 {
            positions_y[i] = radii[i];
            velocities_y[i] = -velocities_y[i];
            boundary_collisions.push((i, 0));
        // top boundary
        } else if positions_y[i] + radii[i] > space_size_y {
            positions_y[i] = space_size_y - radii[i];
            velocities_y[i] = -velocities_y[i];
            boundary_collisions.push((i, 1));
        }
    }
}

fn update(
    num_bodies: usize,
    positions_x: &mut Vec<f64>,
    positions_y: &mut Vec<f64>,
    velocities_x: &mut Vec<f64>,
    velocities_y: &mut Vec<f64>,
    masses: &Vec<f64>,
    radii: &Vec<f64>,
    space_size_x: f64,
    space_size_y: f64,
    time_step: f64,
    velocity_history_x: &mut Vec<Vec<f64>>,
    velocity_history_y: &mut Vec<Vec<f64>>,
    position_history_x: &mut Vec<Vec<f64>>,
    position_history_y: &mut Vec<Vec<f64>>,
    body_collision_history: &mut Vec<Vec<(usize, usize)>>,
    boundary_collision_history: &mut Vec<Vec<(usize, usize)>>,
) {
    // update positions based on velocities
    for i in 0..num_bodies {
        positions_x[i] += velocities_x[i] * time_step;
        positions_y[i] += velocities_y[i] * time_step;
    }

    // handle collisions between bodies
    let mut body_collisions: Vec<(usize, usize)> = Vec::new();
    update_velocities(
        num_bodies,
        velocities_x,
        velocities_y,
        positions_x,
        positions_y,
        masses,
        radii,
        &mut body_collisions,
    );

    // handle collisions with boundaries
    let mut boundary_collisions: Vec<(usize, usize)> = Vec::new();
    handle_boundary_collisions(
        num_bodies,
        positions_x,
        positions_y,
        velocities_x,
        velocities_y,
        radii,
        space_size_x,
        space_size_y,
        &mut boundary_collisions,
    );
    // Append current velocities and positions to their respective histories
    velocity_history_x.push(velocities_x.clone());
    velocity_history_y.push(velocities_y.clone());
    position_history_x.push(positions_x.clone());
    position_history_y.push(positions_y.clone());
    body_collision_history.push(body_collisions);
    boundary_collision_history.push(boundary_collisions);
}

// currently without noise
fn simulate(
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    total_time: f64,
    time_step: f64,
    velocities_x: &mut Vec<f64>,
    velocities_y: &mut Vec<f64>,
    radii: &Vec<f64>,
    masses: &Vec<f64>,
    positions_x: &mut Vec<f64>,
    positions_y: &mut Vec<f64>,
    velocity_history_x: &mut Vec<Vec<f64>>,
    velocity_history_y: &mut Vec<Vec<f64>>,
    position_history_x: &mut Vec<Vec<f64>>,
    position_history_y: &mut Vec<Vec<f64>>,
    body_collision_history: &mut Vec<Vec<(usize, usize)>>,
    boundary_collision_history: &mut Vec<Vec<(usize, usize)>>,
) {
    let mut current_time: f64 = 0.0;

    while current_time < total_time {
        update(
            num_bodies,
            positions_x,
            positions_y,
            velocities_x,
            velocities_y,
            masses,
            radii,
            space_size_x,
            space_size_y,
            time_step,
            velocity_history_x,
            velocity_history_y,
            position_history_x,
            position_history_y,
            body_collision_history,
            boundary_collision_history,
        );
        current_time += time_step;
    }
}

fn run_abc() {
    let num_bodies = 4;
    let space_size_x = 10.0;
    let space_size_y = 10.0;
    let total_time = 1.0;
    let time_step = 0.01;

    let mut rng = rand::thread_rng();
    let velocity_distribution = Uniform::new(-5.0, 5.0).unwrap();
    let radius_distribution = Uniform::new(1.0, 2.0).unwrap();
    let mass_distribution = Uniform::new(1.0, 50.0).unwrap();
    let position_distribution = Uniform::new(0.0, space_size_x).unwrap();

    // first initialize the bodies
    let mut velocities_x: Vec<f64> = velocity_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let mut velocities_y: Vec<f64> = velocity_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let radii: Vec<f64> = radius_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();
    let masses: Vec<f64> = mass_distribution
        .sample_iter(&mut rng)
        .take(num_bodies)
        .collect();

    // initialize positions
    let (positions_x, positions_y) = sample_initial_positions(
        num_bodies,
        space_size_x,
        space_size_y,
        &radii,
        &position_distribution,
    );

    // get final positions
    let mut final_positions_x: Vec<f64> = positions_x.clone();
    let mut final_positions_y: Vec<f64> = positions_y.clone();
    // get positions and velocities history
    let mut velocity_history_x: Vec<Vec<f64>> = Vec::new();
    let mut velocity_history_y: Vec<Vec<f64>> = Vec::new();
    let mut position_history_x: Vec<Vec<f64>> = Vec::new();
    let mut position_history_y: Vec<Vec<f64>> = Vec::new();
    // get collision history
    let mut body_collision_history: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut boundary_collision_history: Vec<Vec<(usize, usize)>> = Vec::new();

    // append initial velocities and positions to their respective histories
    velocity_history_x.push(velocities_x.clone());
    velocity_history_y.push(velocities_y.clone());
    position_history_x.push(positions_x.clone());
    position_history_y.push(positions_y.clone());

    // simulate to get final positions
    simulate(
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        &mut velocities_x,
        &mut velocities_y,
        &radii,
        &masses,
        &mut final_positions_x,
        &mut final_positions_y,
        &mut velocity_history_x,
        &mut velocity_history_y,
        &mut position_history_x,
        &mut position_history_y,
        &mut body_collision_history,
        &mut boundary_collision_history,
    );

    // abc algorithm
    let n = 10;
    let epsilon = 1.0;

    let mut accepted_states = 0;
    let mut accepted_velocities_x: Vec<Vec<f64>> = Vec::new();
    let mut accepted_velocities_y: Vec<Vec<f64>> = Vec::new();
    let mut accepted_positions_x: Vec<Vec<f64>> = Vec::new();
    let mut accepted_positions_y: Vec<Vec<f64>> = Vec::new();
    // velocity and position history of accepted states
    let mut accepted_velocity_history_x: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut accepted_velocity_history_y: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut accepted_position_history_x: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut accepted_position_history_y: Vec<Vec<Vec<f64>>> = Vec::new();
    // get collision history of accepted states
    let mut accepted_body_collision_history: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();
    let mut accepted_boundary_collision_history: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();

    let mut current_positions_x: Vec<f64>;
    let mut current_positions_y: Vec<f64>;
    let mut current_velocities_x: Vec<f64>;
    let mut current_velocities_y: Vec<f64>;
    // velocity and position history of current state
    let mut current_velocity_history_x: Vec<Vec<f64>>;
    let mut current_velocity_history_y: Vec<Vec<f64>>;
    let mut current_position_history_x: Vec<Vec<f64>>;
    let mut current_position_history_y: Vec<Vec<f64>>;
    // get collision history of current state
    let mut current_body_collision_history: Vec<Vec<(usize, usize)>>;
    let mut current_boundary_collision_history: Vec<Vec<(usize, usize)>>;

    let mut current_iteration: usize = 0;

    let start = std::time::Instant::now(); // measure time

    while accepted_states < n {
        // clone initial states so that we can revert to them if the state is not accepted
        current_positions_x = positions_x.clone();
        current_positions_y = positions_y.clone();

        // sample new velocities
        current_velocities_x = velocity_distribution
            .sample_iter(&mut rng)
            .take(num_bodies)
            .collect();
        current_velocities_y = velocity_distribution
            .sample_iter(&mut rng)
            .take(num_bodies)
            .collect();

        // reset history
        current_velocity_history_x = Vec::new();
        current_velocity_history_y = Vec::new();
        current_position_history_x = Vec::new();
        current_position_history_y = Vec::new();
        current_body_collision_history = Vec::new();
        current_boundary_collision_history = Vec::new();
        // append initial velocities and positions to their respective histories
        current_velocity_history_x.push(current_velocities_x.clone());
        current_velocity_history_y.push(current_velocities_y.clone());
        current_position_history_x.push(current_positions_x.clone());
        current_position_history_y.push(current_positions_y.clone());

        // simulate with sampled velocities
        simulate(
            num_bodies,
            space_size_x,
            space_size_y,
            total_time,
            time_step,
            &mut current_velocities_x,
            &mut current_velocities_y,
            &radii,
            &masses,
            &mut current_positions_x,
            &mut current_positions_y,
            &mut current_velocity_history_x,
            &mut current_velocity_history_y,
            &mut current_position_history_x,
            &mut current_position_history_y,
            &mut current_body_collision_history,
            &mut current_boundary_collision_history,
        );

        // calculate the error
        // ! TODO: this is probably not the best way to calculate the error
        let mut error: f64 = 0.0;
        for i in 0..num_bodies {
            error += (final_positions_x[i] - current_positions_x[i]).powi(2);
            error += (final_positions_y[i] - current_positions_y[i]).powi(2);
        }
        error = error.sqrt();

        // if the error is less than epsilon, accept the state
        if error < epsilon {
            accepted_states += 1;
            println!("accepted states: {}", accepted_states);
            accepted_velocities_x.push(current_velocities_x.clone());
            accepted_velocities_y.push(current_velocities_y.clone());
            accepted_positions_x.push(current_positions_x.clone());
            accepted_positions_y.push(current_positions_y.clone());
            // append history
            accepted_velocity_history_x.push(current_velocity_history_x.clone());
            accepted_velocity_history_y.push(current_velocity_history_y.clone());
            accepted_position_history_x.push(current_position_history_x.clone());
            accepted_position_history_y.push(current_position_history_y.clone());
            // append collision history
            accepted_body_collision_history.push(current_body_collision_history.clone());
            accepted_boundary_collision_history.push(current_boundary_collision_history.clone());
        }

        current_iteration += 1;
    }
    println!("total iterations: {}", current_iteration);
    println!("time: {:?}", start.elapsed());
    // time per iteration
    println!(
        "time per iteration: {:?}",
        start.elapsed() / current_iteration as u32
    );
    // in seconds
    println!("time per iteration in seconds: {:?}", start.elapsed().as_secs_f64()/current_iteration as f64);

    // println!("final original positions x: {:?}", final_positions_x);
    // println!("final original positions y: {:?}", final_positions_y);
    // println!("accepted positions x: {:?}", accepted_positions_x);
    // println!("accepted positions y: {:?}", accepted_positions_y);
    // println!("original velocities x: {:?}", velocities_x);
    // println!("original velocities y: {:?}", velocities_y);
    // println!("accepted velocities x: {:?}", accepted_velocities_x);
    // println!("accepted velocities y: {:?}", accepted_velocities_y);

    write_simulation_data_to_json(
        "data/original_simulation_data.json",
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        &radii,
        &masses,
        // put into vec to match the format of the accepted data
        &vec![velocity_history_x.clone()],
        &vec![velocity_history_y.clone()],
        &vec![position_history_x.clone()],
        &vec![position_history_y.clone()],
        &vec![body_collision_history.clone()],
        &vec![boundary_collision_history.clone()],
    );

    write_simulation_data_to_json(
        "data/accepted_simulation_data.json",
        num_bodies,
        space_size_x,
        space_size_y,
        total_time,
        time_step,
        &radii,
        &masses,
        &accepted_velocity_history_x,
        &accepted_velocity_history_y,
        &accepted_position_history_x,
        &accepted_position_history_y,
        &accepted_body_collision_history,
        &accepted_boundary_collision_history,
    );
}

fn write_simulation_data_to_json(
    file_name: &str,
    num_bodies: usize,
    space_size_x: f64,
    space_size_y: f64,
    total_time: f64,
    time_step: f64,
    radii: &Vec<f64>,
    masses: &Vec<f64>,
    velocity_history_x: &Vec<Vec<Vec<f64>>>,
    velocity_history_y: &Vec<Vec<Vec<f64>>>,
    position_history_x: &Vec<Vec<Vec<f64>>>,
    position_history_y: &Vec<Vec<Vec<f64>>>,
    body_collision_history: &Vec<Vec<Vec<(usize, usize)>>>,
    boundary_collision_history: &Vec<Vec<Vec<(usize, usize)>>>,
) {
    let simulation_data = serde_json::json!({
        "num_bodies": num_bodies,
        "space_size_x": space_size_x,
        "space_size_y": space_size_y,
        "total_time": total_time,
        "time_step": time_step,
        "radii": radii,
        "masses": masses,
        "velocity_history_x": velocity_history_x,
        "velocity_history_y": velocity_history_y,
        "position_history_x": position_history_x,
        "position_history_y": position_history_y,
        "body_collision_history": body_collision_history,
        "boundary_collision_history": boundary_collision_history
    });

    // pretty
    let simulation_data_string = serde_json::to_string_pretty(&simulation_data).unwrap();
    // not pretty
    // let simulation_data_string = serde_json::to_string(&simulation_data).unwrap();
    std::fs::write(file_name, simulation_data_string).unwrap();
}

fn main() {
    run_abc();
}
