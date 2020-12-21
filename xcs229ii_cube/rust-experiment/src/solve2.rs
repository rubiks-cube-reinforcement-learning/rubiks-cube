mod cube2;
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

static LOOKUP_FILE_PATH: &str = "./results-cubies-fixed.txt";

fn generate_dataset(nb_per_scramble: i32, max_scrambles: i32) -> Vec<i128> {
    iproduct!(0..nb_per_scramble, 0..max_scrambles)
        .par_bridge()
        .map(|(n, scrambles)| { generate_scrambled_state(scrambles) })
        .collect::<Vec<i128>>()
}

pub fn solve_cube<'a>(x: i128, lookup: &FxHashMap<i128, i8>) -> Vec<&'a str> {
    let mut solution: Vec<&str> = vec![];
    let mut state = x;
    if !lookup.contains_key(&state) {
        state = cube2::orient_cube(state);
    }

    let mut distance: i8 = *lookup.get(&state).unwrap();
    let mut updated: bool;
    loop {
        updated = false;
        for (i, op) in cube2::FIXED_CUBIE_OPERATIONS.iter().enumerate() {
            let new_state = op(state);
            let new_distance = *lookup.get(&new_state).unwrap();
            if new_distance < distance {
                solution.push(cube2::FIXED_CUBIE_OPERATIONS_NAMES[i]);
                state = new_state;
                distance = new_distance;
                updated = true;
                break;
            }
        }
        if (distance == 0) {
            break;
        }
        if (!updated) {
            panic!("No move found for state {}", state);
        }
    }

    return solution;
}

fn generate_scrambled_state(scrambles: i32) -> i128 {
    let mut state = cube2::SOLVED_STATE;
    let mut op = 100;
    let mut op_idx = 100;
    for step in 0..scrambles {
        loop {
            if op != op_idx {
                op = op_idx;
                break;
            }
            op_idx = rand::thread_rng().gen_range(0, cube2::ALL_OPERATIONS.len());
        }
        state = cube2::ALL_OPERATIONS[op_idx](state);
    }
    state
}

fn generate_state_lookup_table() {
    let mut stack: VecDeque<(i128, i8)> = VecDeque::new();
    let mut lookup = HashMap::new();

    stack.push_back((cube2::SOLVED_STATE, 0));
    lookup.insert(cube2::SOLVED_STATE, 0);

    let mut i = 0;
    loop {
        let elem = stack.pop_front();
        if elem.is_none() {
            break;
        }
        let (state, distance) = elem.unwrap();
        let results: Vec<i128> = cube2::FIXED_CUBIE_OPERATIONS.par_iter()
            .map(|op| { op(state) })
            .filter(|new_state| { !lookup.contains_key(&new_state) })
            .collect();

        for new_state in results {
            stack.push_back((new_state, distance + 1));
            lookup.insert(new_state, distance + 1);
        }
        i = i + 1;
        if i % 100000 == 0 {
            println!("size={0} state={1} distance={2}", lookup.len(), state, distance);

            if lookup.len() > 50000000 {
                break;
            }
        }
    }

    let data = lookup.iter()
        .map(|(key, &val)| format!("{1} {0:072b}\n", key, val))
        .collect::<String>();
    fs::write(LOOKUP_FILE_PATH, data).expect("Unable to write file");
}

pub fn load_lookup_table(path: &str) -> FxHashMap<i128, i8> {
    let mut lookup = FxHashMap::default();
    if let Ok(lines) = read_lines(path) {
        // Consumes the iterator, returns an (Optional) String
        lookup = lines
            .filter_map(Result::ok)
            .par_bridge()
            .map(|line| {
                let words: Vec<&str> = line.split_whitespace().collect();
                let state = i128::from_str_radix(words[1], 2).unwrap();
                let distance = words[0].parse::<i8>().unwrap();
                return (state, distance);
            })
            .collect::<FxHashMap<i128, i8>>()
        ;
    }
    return lookup;
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
