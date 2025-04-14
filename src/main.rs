// https://uqworld.org/t/liquid-hydrogen-tank-problem/58

// Modelling failure event of a 5 dim liquid hydrogen tank
// subjected to stress caused by:
// - ullage pressure
// - head pressure
// - axial forces due to acceleration
// - bending
// - shear stresses caused by weight of fuel

use clap::Parser;
use rand_distr::{Distribution, Normal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Parser)]
#[command(about = "Simulate failure probability of LH2 tank", author, version)]
struct Args {
    // Number of iterations to run (default: 1_000_000)
    #[arg(short, long, default_value_t = 1_000_00)]
    iterations: u32,
}

struct HydrogenTank {
    t_plate: f32, // Plate thickness
    t_h: f32,     // Honeycomb thickness
    n_x: f32,     // Load on tank, x component
    n_y: f32,     // Load on tank, y component
    n_xy: f32,    // Load on tank, xy component
}

impl HydrogenTank {
    fn new(t_plate: f32, t_h: f32, n_x: f32, n_y: f32, n_xy: f32) -> Self {
        HydrogenTank {
            t_plate,
            t_h,
            n_x,
            n_y,
            n_xy,
        }
    }

    // von mises stress failure
    fn simulated_pvm(&self) -> f32 {
        let div = (self.n_x * self.n_x) + (self.n_y * self.n_y) - (self.n_x * self.n_y)
            + (3.0 * self.n_xy * self.n_xy);

        return -1.0 + 84000.0 * self.t_plate / div.sqrt();
    }

    // isotropic strength failure
    fn simulated_pis(&self) -> f32 {
        return -1.0 + 84000.0 * self.t_plate / self.n_y.abs();
    }

    // honeycomb buckling failure
    fn simulated_phb(&self) -> f32 {
        let x1 = 4.0 * (self.t_plate - 0.075);
        let x2 = 20.0 * (self.t_h - 0.1);
        let x3 = -6000.0 * (1.0 / self.n_xy + 0.003);

        return 0.847 + 0.96 * x1 + 0.986 * x2 - 0.216 * x3
            + 0.077 * x1 * x1
            + 0.11 * x2 * x2
            + 0.007 * x3 * x3
            + 0.378 * x1 * x2
            - 0.106 * x1 * x3
            - 0.11 * x2 * x3;
    }

    // g(x) = min(PVM, PIS, PHB)
    // fn limit_state_function(&self) -> f32 {
    //     return self
    //         .simulated_pvm()
    //         .min(self.simulated_pis())
    //         .min(self.simulated_phb());
    // }
}

fn main() {
    let args = Args::parse();
    let iterations = args.iterations;

    let start = std::time::Instant::now();

    // Sampling variables from their distributions
    let dist_t_plate = Normal::new(0.07433, 0.005).unwrap();
    let dist_t_h = Normal::new(0.1, 0.01).unwrap();
    let dist_nx = Normal::new(13.0, 60.0).unwrap();
    let dist_ny = Normal::new(4751.0, 48.0).unwrap();
    let dist_nxy = Normal::new(-648.0, 11.0).unwrap();

    // Parallel simulation
    let (total_failures, pvm_failures, pis_failures, phb_failures) = (0..iterations)
        .into_par_iter()
        .map_init(
            || rand::rng(),
            |rng, _| {
                let tank = HydrogenTank::new(
                    dist_t_plate.sample(rng),
                    dist_t_h.sample(rng),
                    dist_nx.sample(rng),
                    dist_ny.sample(rng),
                    dist_nxy.sample(rng),
                );

                let pvm = tank.simulated_pvm() <= 0.0;
                let pis = tank.simulated_pis() <= 0.0;
                let phb = tank.simulated_phb() <= 0.0;

                (
                    (pvm || pis || phb) as usize,
                    pvm as usize,
                    pis as usize,
                    phb as usize,
                )
            },
        )
        .reduce(
            || (0, 0, 0, 0),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        );

    // Calculate probabilities
    let total_prob = total_failures as f32 / iterations as f32;
    let pvm_prob = pvm_failures as f32 / iterations as f32;
    let pis_prob = pis_failures as f32 / iterations as f32;
    let phb_prob = phb_failures as f32 / iterations as f32;

    // Print results with more precision
    println!(
        "\nSimulation completed in {:.1?} for {:e} iterations",
        start.elapsed(),
        iterations
    );
    println!("Failure probability: {:.6}", total_prob);
    println!("Failure by mode:");
    println!("  Von Mises: {:.6} ({} cases)", pvm_prob, pvm_failures);
    println!("  Isotropic: {:.6} ({} cases)", pis_prob, pis_failures);
    println!("  Honeycomb: {:.6} ({} cases)", phb_prob, phb_failures);

    // Identify dominant failure mode
    let dominant = if pvm_failures >= pis_failures && pvm_failures >= phb_failures {
        "Von Mises stress"
    } else if pis_failures >= pvm_failures && pis_failures >= phb_failures {
        "Isotropic strength"
    } else {
        "Honeycomb buckling"
    };

    println!("Dominant failure mode: {}", dominant);
}
