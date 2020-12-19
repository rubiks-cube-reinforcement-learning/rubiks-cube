mod solve2;
extern crate fxhash;
use fxhash::FxHashMap;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::panic;
use std::path::Path;
use std::hash::{BuildHasher, BuildHasherDefault};
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use rand::Rng;
use rayon::prelude::*;
use itertools::iproduct;

pub fn bench() {
    println!("Loading lookup table...");
    let lookup = solve2::load_lookup_table(LOOKUP_FILE_PATH);
    println!("Generating dataset...");
    let nb_per_scramble:i32 = 10000;
    let max_scrambles:i32 = 100;
    let repetitions:i32 = 10;
    let nb_iterations : i128 = (nb_per_scramble as i128) * (max_scrambles as i128) * (repetitions as i128);
    let dataset = solve2::generate_dataset(nb_per_scramble, max_scrambles);

    println!("Test solution to {0}: {1:?}", dataset[0], solve_cube(dataset[0], &lookup));
    println!("Solving the dataset of size {}...", dataset.len());
    let now = Instant::now();
    let mut really_solved:i128 = 0;
    for i in 1..repetitions {
        really_solved += dataset
            .par_iter()
            .map(|state| { solve2::solve_cube(*state, &lookup).len() })
            .reduce(|| 0,
                    |a, b| a + b) as i128;
    }
    println!("{0:?} seconds for solving {1} cubes {2}.", now.elapsed(), nb_iterations, really_solved);
}
