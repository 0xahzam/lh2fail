// https://uqworld.org/t/liquid-hydrogen-tank-problem/58

// Modelling failure event of a 5 dim liquid hydrogen tank
// subjected to stress caused by:
// - ullage pressure
// - head pressure
// - axial forces due to acceleration
// - bending
// - shear stresses caused by weight of fuel

use clap::Parser;
use rand::{Rng, seq::SliceRandom};
use rand_distr::{Distribution, Normal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};

#[derive(Parser)]
#[command(about = "Simulate failure probability of LH2 tank", author, version)]
struct Args {
    // Number of iterations to run (default: 1_000_000)
    #[arg(short, long, default_value_t = 1_000_000)]
    iterations: u32,

    /// Use Latin Hypercube Sampling instead of standard sampling
    #[arg(long, default_value_t = false)]
    lhd: bool,
}

struct HydrogenTank {
    t_plate: f64, // Plate thickness
    t_h: f64,     // Honeycomb thickness
    n_x: f64,     // Load on tank, x component
    n_y: f64,     // Load on tank, y component
    n_xy: f64,    // Load on tank, xy component
}

struct DistributionsType {
    normal: Normal<f64>,
    statnormal: StatNormal,
}

struct DistributionState {
    t_plate: DistributionsType,
    t_h: DistributionsType,
    n_x: DistributionsType,
    n_y: DistributionsType,
    n_xy: DistributionsType,
}

impl HydrogenTank {
    fn new(t_plate: f64, t_h: f64, n_x: f64, n_y: f64, n_xy: f64) -> Self {
        HydrogenTank {
            t_plate,
            t_h,
            n_x,
            n_y,
            n_xy,
        }
    }

    fn get_distributions() -> DistributionState {
        DistributionState {
            t_plate: DistributionsType {
                normal: Normal::new(0.07433, 0.005).unwrap(),
                statnormal: StatNormal::new(0.07433, 0.005).unwrap(),
            },
            t_h: DistributionsType {
                normal: Normal::new(0.1, 0.01).unwrap(),
                statnormal: StatNormal::new(0.1, 0.01).unwrap(),
            },
            n_x: DistributionsType {
                normal: Normal::new(13.0, 60.0).unwrap(),
                statnormal: StatNormal::new(13.0, 60.0).unwrap(),
            },
            n_y: DistributionsType {
                normal: Normal::new(4751.0, 48.0).unwrap(),
                statnormal: StatNormal::new(4751.0, 48.0).unwrap(),
            },
            n_xy: DistributionsType {
                normal: Normal::new(-648.0, 11.0).unwrap(),
                statnormal: StatNormal::new(-648.0, 11.0).unwrap(),
            },
        }
    }

    // Generate Latin Hypercube samples in [0, 1]
    // `n` = number of total samples (rows)
    // `d` = number of input variables (columns)
    fn lhd(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let mut matrix = vec![vec![0.0; d]; n];

        for dim in 0..d {
            // Divide [0,1] into n buckets and pick one value from each
            let mut perm = (0..n).collect::<Vec<_>>();

            // Shuffle the permutation
            perm.shuffle(&mut rng);

            // For each sample, calculate a stratified value
            for i in 0..n {
                // Get which bucket (0 to n-1) this sample uses in this dimension
                let bucket = perm[i];

                // Generate random offset within the bucket
                let offset = rng.random::<f64>();

                // Calculate final value: (bucket + offset) / n
                matrix[i][dim] = (bucket as f64 + offset) / (n as f64);
            }
        }

        matrix
    }

    // von mises stress failure
    fn simulated_pvm(&self) -> f64 {
        let div = (self.n_x * self.n_x) + (self.n_y * self.n_y) - (self.n_x * self.n_y)
            + (3.0 * self.n_xy * self.n_xy);

        return -1.0 + 84000.0 * self.t_plate / div.sqrt();
    }

    // isotropic strength failure
    fn simulated_pis(&self) -> f64 {
        return -1.0 + 84000.0 * self.t_plate / self.n_y.abs();
    }

    // honeycomb buckling failure
    fn simulated_phb(&self) -> f64 {
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
    // fn limit_state_function(&self) -> f64 {
    //     return self
    //         .simulated_pvm()
    //         .min(self.simulated_pis())
    //         .min(self.simulated_phb());
    // }
}

fn main() {
    let args = Args::parse();
    let iterations = args.iterations;
    let lhd_flag = args.lhd;

    let start = std::time::Instant::now();

    // Sampling variables from their distributions
    let distribution_state = HydrogenTank::get_distributions();

    let n_samples = iterations as usize; // Number of samples to generate
    let n_dims = 5; // Number of dimensions (5 variables for the tank)

    let (total_failures, pvm_failures, pis_failures, phb_failures) = if lhd_flag {
        // LHD sampling
        let lhd_samples = HydrogenTank::lhd(n_samples, n_dims);

        lhd_samples
            .into_par_iter()
            .map(|s| {
                let tank = HydrogenTank::new(
                    distribution_state.t_plate.statnormal.inverse_cdf(s[0]),
                    distribution_state.t_h.statnormal.inverse_cdf(s[1]),
                    distribution_state.n_x.statnormal.inverse_cdf(s[2]),
                    distribution_state.n_y.statnormal.inverse_cdf(s[3]),
                    distribution_state.n_xy.statnormal.inverse_cdf(s[4]),
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
            })
            .reduce(
                || (0, 0, 0, 0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
            )
    } else {
        // Standard mode: use uniform sampling
        (0..iterations)
            .into_par_iter()
            .map_init(
                || rand::rng(),
                |rng, _| {
                    let tank = HydrogenTank::new(
                        distribution_state.t_plate.normal.sample(rng),
                        distribution_state.t_h.normal.sample(rng),
                        distribution_state.n_x.normal.sample(rng),
                        distribution_state.n_y.normal.sample(rng),
                        distribution_state.n_xy.normal.sample(rng),
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
            )
    };

    // Calculate probabilities
    let total_prob = total_failures as f64 / iterations as f64;
    let pvm_prob = pvm_failures as f64 / iterations as f64;
    let pis_prob = pis_failures as f64 / iterations as f64;
    let phb_prob = phb_failures as f64 / iterations as f64;

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
