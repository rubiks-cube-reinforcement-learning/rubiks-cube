use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::panic;
use std::path::Path;
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use rand::Rng;
use rayon::prelude::*;
use itertools::iproduct;

fn main() {
    println!("Loading lookup table...");
    let lookup = load_lookup_table(LOOKUP_FILE_PATH);
    println!("Generating dataset...");
    let nb_per_scramble:i32 = 100000;
    let max_scrambles:i32 = 100;
    let repetitions:i32 = 1000;
    let nb_iterations : i128 = (nb_per_scramble as i128) * (max_scrambles as i128) * (repetitions as i128);
    let dataset = generate_dataset(nb_per_scramble, max_scrambles);

    println!("Test solution to {0}: {1:?}", dataset[0], solve_cube(dataset[0], &lookup));
    println!("Solving the dataset of size {}...", dataset.len());
    let now = Instant::now();
    let mut really_solved = 0;
    for i in 1..repetitions {
        really_solved += dataset
            .par_iter()
            .map(|state| { solve_cube(*state, &lookup) })
            .len();
    }
    println!("{0:?} seconds for solving {1} cubes {2}.", now.elapsed(), nb_iterations, really_solved);
}

fn generate_dataset(nb_per_scramble: i32, max_scrambles: i32) -> Vec<i128> {
    iproduct!(0..nb_per_scramble, 0..max_scrambles)
        .par_bridge()
        .map(|(n, scrambles)| { generate_scrambled_state(scrambles) })
        .collect::<Vec<i128>>()
}

static SOLVED_STATE: i128 = 0x2494926db924b6ddb6;
static LOOKUP_FILE_PATH: &str = "./results-cubies-fixed.txt";

fn solve_cube<'a>(x: i128, lookup: &HashMap<i128, i8>) -> Vec<&'a str> {
    let mut solution: Vec<&str> = vec![];
    let mut state = x;
    if !lookup.contains_key(&state) {
        state = orient_cube(state);
    }

    let mut distance: i8 = *lookup.get(&state).unwrap();
    let mut updated: bool;
    loop {
        updated = false;
        for (i, op) in FIXED_CUBIE_OPERATIONS.iter().enumerate() {
            let new_state = op(state);
            let new_distance = *lookup.get(&new_state).unwrap();
            if new_distance < distance {
                solution.push(FIXED_CUBIE_OPERATIONS_NAMES[i]);
                state = new_state;
                distance = new_distance;
                updated = true;
                break;
            }
        }
        if (!updated) {
            panic!("No move found for state {}", state);
        }
        if (distance == 0) {
            break;
        }
    }

    return solution;
}

fn generate_scrambled_state(scrambles: i32) -> i128 {
    let mut state = SOLVED_STATE;
    let mut op = 100;
    let mut op_idx = 100;
    for step in 0..scrambles {
        loop {
            if op != op_idx {
                op = op_idx;
                break;
            }
            op_idx = rand::thread_rng().gen_range(0, ALL_OPERATIONS.len());
        }
        state = ALL_OPERATIONS[op_idx](state);
    }
    state
}

fn generate_state_lookup_table() {
    let mut stack: VecDeque<(i128, i8)> = VecDeque::new();
    let mut lookup = HashMap::new();

    stack.push_back((SOLVED_STATE, 0));
    lookup.insert(SOLVED_STATE, 0);

    let mut i = 0;
    loop {
        let elem = stack.pop_front();
        if elem.is_none() {
            break;
        }
        let (state, distance) = elem.unwrap();
        let results: Vec<i128> = FIXED_CUBIE_OPERATIONS.par_iter()
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

// @TODO consider simple hasher https://gist.github.com/arthurprs/88eef0b57b9f8341c54e2d82ec775698
fn load_lookup_table(path: &str) -> HashMap<i128, i8> {
    let mut lookup: HashMap<i128, i8> = HashMap::new();
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
            .collect::<HashMap<i128, i8>>()
        ;
    }
    return lookup;
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// 101101011101101011010011100100110110001100010010001001010110100110011001
// 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
static ALL_OPERATIONS: &'static [fn(i128) -> i128] = &[
    // Something is broken here - it goes beyond 29 million states, why?
    lu, ld, ru, rd, fl, fr, ul, ur, dl, dr, bl, br,
];

static ALL_OPERATIONS_NAMES: &'static [&str] = &[
    "lu", "ld", "ru", "rd", "fl", "fr", "ul", "ur", "dl", "dr", "bl", "br"
];

static FIXED_CUBIE_OPERATIONS: &'static [fn(i128) -> i128] = &[
    // This finishes at 3674026 states as it's supposed to
    lu, ld, ul, ur, bl, br,
];

static FIXED_CUBIE_OPERATIONS_NAMES: &'static [&str] = &[
    "lu", "ld", "ul", "ur", "bl", "br"
];


fn orient_cube(x: i128) -> i128 {
    let mut actual_color_pattern: i128;
    actual_color_pattern = ((x & 0x7000000000000000) >> 54) | ((x & 0x7000000) >> 21) | ((x & 0x38) >> 3);
    if actual_color_pattern == 102 {
        return ((x & 0xffffffffffffffffff) << 0);
    }
    if actual_color_pattern == 305 {
        return ((x & 0x1c0) << 63) | ((x & 0x7) << 66) | ((x & 0xe00) << 54) | ((x & 0x38) << 57) | ((x & 0xe07000000000) << 12) | ((x & 0x38000000000) << 15) | ((x & 0x1c0000000000) << 9) | ((x & 0xe07000) << 24) | ((x & 0x38000) << 27) | ((x & 0x1c0000) << 21) | ((x & 0xe07fff000000000000) >> 36) | ((x & 0x38000000000000000) >> 33) | ((x & 0x1c0000000000000000) >> 39) | ((x & 0x1c7000000) >> 21) | ((x & 0xe38000000) >> 27);
    }
    if actual_color_pattern == 396 {
        return ((x & 0xe07fff000) << 36) | ((x & 0x38000000) << 39) | ((x & 0x1c0000000) << 33) | ((x & 0xe07000000000000) >> 12) | ((x & 0x38000000000000) >> 9) | ((x & 0x1c0000000000000) >> 15) | ((x & 0x1c7) << 27) | ((x & 0xe38) << 21) | ((x & 0xe07000000000) >> 24) | ((x & 0x38000000000) >> 21) | ((x & 0x1c0000000000) >> 27) | ((x & 0x38000000000000000) >> 54) | ((x & 0xe00000000000000000) >> 63) | ((x & 0x7000000000000000) >> 57) | ((x & 0x1c0000000000000000) >> 66);
    }
    actual_color_pattern = ((x & 0x1c0000000000000000) >> 60) | ((x & 0x7000000000000) >> 45) | ((x & 0xe00) >> 9);
    if actual_color_pattern == 270 {
        return ((x & 0xfff000fff000000) << 12) | ((x & 0x3f000000000) << 18) | ((x & 0xfc0000038007) << 6) | ((x & 0x3f000000000000000) >> 30) | ((x & 0xfc0000000000000000) >> 42) | ((x & 0xe00038) >> 3) | ((x & 0x71c0) << 3) | ((x & 0x1c0e00) >> 6);
    }
    if actual_color_pattern == 116 {
        return ((x & 0x38000038000000000) << 6) | ((x & 0xe00000e00000000000) >> 3) | ((x & 0x7000007000000000) << 3) | ((x & 0x1c00001c0000000000) >> 6) | ((x & 0x38000) << 42) | ((x & 0xe00007) << 33) | ((x & 0x7000) << 39) | ((x & 0x1c0000) << 30) | ((x & 0x1f8) << 24) | ((x & 0xe00) << 15) | ((x & 0xe07000000) >> 12) | ((x & 0x38000000) >> 9) | ((x & 0x1c0000000) >> 15) | ((x & 0x38000000000000) >> 42) | ((x & 0xe00000000000000) >> 51) | ((x & 0x7000000000000) >> 45) | ((x & 0x1c0000000000000) >> 54);
    }
    if actual_color_pattern == 417 {
        return ((x & 0x7) << 69) | ((x & 0x38) << 63) | ((x & 0x1c0) << 57) | ((x & 0xe00) << 51) | ((x & 0xe07000000) << 24) | ((x & 0x38000000) << 27) | ((x & 0x1c0000000) << 21) | ((x & 0x3f000) << 30) | ((x & 0xfc0000) << 18) | ((x & 0xe07000000000000) >> 24) | ((x & 0x38000000000000) >> 21) | ((x & 0x1c0000000000000) >> 27) | ((x & 0x3f000000000) >> 18) | ((x & 0xfc0000000000) >> 30) | ((x & 0x7000000000000000) >> 51) | ((x & 0x38000000000000000) >> 57) | ((x & 0x1c0000000000000000) >> 63) | ((x & 0xe00000000000000000) >> 69);
    }
    actual_color_pattern = ((x & 0x38000000000000000) >> 57) | ((x & 0x38000000) >> 24) | ((x & 0x7000) >> 12);
    if actual_color_pattern == 417 {
        return ((x & 0xfff000) << 48) | ((x & 0x1c00001c0000000) << 3) | ((x & 0x7000007000000) << 6) | ((x & 0xe00000e00000000) >> 6) | ((x & 0x38000038000000) >> 3) | ((x & 0x1c7) << 39) | ((x & 0xe38) << 33) | ((x & 0x1c7000000000) >> 21) | ((x & 0xe38000000000) >> 27) | ((x & 0xfff000000000000000) >> 60);
    }
    if actual_color_pattern == 116 {
        return ((x & 0x1c00001c0000000000) << 3) | ((x & 0x7000007000000000) << 6) | ((x & 0xe00000e00000000000) >> 6) | ((x & 0x38000038000000000) >> 3) | ((x & 0x1c0) << 51) | ((x & 0x7) << 54) | ((x & 0xe00) << 42) | ((x & 0x38) << 45) | ((x & 0xe07000) << 12) | ((x & 0x38000) << 15) | ((x & 0x1c0000) << 9) | ((x & 0x1c0000e00000000) >> 33) | ((x & 0x7000000000000) >> 30) | ((x & 0xe00000000000000) >> 42) | ((x & 0x38000000000000) >> 39) | ((x & 0x7000000) >> 15) | ((x & 0x1f8000000) >> 24);
    }
    if actual_color_pattern == 270 {
        return ((x & 0x1c7000000) << 39) | ((x & 0xe38000000) << 33) | ((x & 0x1c70000001c0) << 15) | ((x & 0xe38000000038) << 9) | ((x & 0x1c7000000007000) >> 9) | ((x & 0xe38000000e00000) >> 15) | ((x & 0x1c7000000000000000) >> 33) | ((x & 0xe38000000000000000) >> 39) | ((x & 0x7) << 18) | ((x & 0xe00) << 6) | ((x & 0x38000) >> 6) | ((x & 0x1c0000) >> 18);
    }
    actual_color_pattern = ((x & 0x7000000000) >> 30) | ((x & 0x1c0000000) >> 27) | ((x & 0x7) << 0);
    if actual_color_pattern == 270 {
        return ((x & 0x3f000000) << 42) | ((x & 0xfc0000000) << 30) | ((x & 0xfff000fff000000000) >> 12) | ((x & 0x3f000000e001c0) >> 6) | ((x & 0xfc0000000000000) >> 18) | ((x & 0x1c0007) << 3) | ((x & 0x7038) << 6) | ((x & 0x38e00) >> 3);
    }
    if actual_color_pattern == 417 {
        return ((x & 0xfff) << 60) | ((x & 0x38000038000000) << 6) | ((x & 0xe00000e00000000) >> 3) | ((x & 0x7000007000000) << 3) | ((x & 0x1c00001c0000000) >> 6) | ((x & 0x1c7000) << 27) | ((x & 0xe38000) << 21) | ((x & 0xfff000000000000000) >> 48) | ((x & 0x1c7000000000) >> 33) | ((x & 0xe38000000000) >> 39);
    }
    if actual_color_pattern == 116 {
        return ((x & 0xe07000000e07) << 24) | ((x & 0x38000000038) << 27) | ((x & 0x1c00000001c0) << 21) | ((x & 0x1c0000) << 39) | ((x & 0x7000) << 42) | ((x & 0xe00000) << 30) | ((x & 0x38000) << 33) | ((x & 0xe07000000e07000000) >> 24) | ((x & 0x38000000038000000) >> 21) | ((x & 0x1c00000001c0000000) >> 27) | ((x & 0x38000000000000) >> 30) | ((x & 0xe00000000000000) >> 39) | ((x & 0x7000000000000) >> 33) | ((x & 0x1c0000000000000) >> 42);
    }
    actual_color_pattern = ((x & 0x1c0000000000) >> 36) | ((x & 0x1c0000000000000) >> 51) | ((x & 0x1c0) >> 6);
    if actual_color_pattern == 102 {
        return ((x & 0x3f03f000000) << 30) | ((x & 0xfc0fc0000000) << 18) | ((x & 0x3f03f000000000000) >> 18) | ((x & 0xfc0fc0000000000000) >> 30) | ((x & 0x7007) << 9) | ((x & 0x38038) << 3) | ((x & 0x1c01c0) >> 3) | ((x & 0xe00e00) >> 9);
    }
    if actual_color_pattern == 396 {
        return ((x & 0x38000038000fc0) << 18) | ((x & 0xe00000e00000000) << 9) | ((x & 0x7000007000000) << 15) | ((x & 0x1c00001c0000000) << 6) | ((x & 0x7000) << 45) | ((x & 0x38000) << 39) | ((x & 0x1c0000) << 33) | ((x & 0xe00000) << 27) | ((x & 0x3f) << 30) | ((x & 0x38000000000000000) >> 42) | ((x & 0xe00000000000000000) >> 51) | ((x & 0x7000000000000000) >> 45) | ((x & 0x1c0000000000000000) >> 54) | ((x & 0xe07000000000) >> 36) | ((x & 0x38000000000) >> 33) | ((x & 0x1c0000000000) >> 39);
    }
    if actual_color_pattern == 305 {
        return ((x & 0x38) << 66) | ((x & 0xe00) << 57) | ((x & 0x7) << 63) | ((x & 0x1c0) << 54) | ((x & 0x3800003803f000000) >> 6) | ((x & 0xe00000e00000000000) >> 15) | ((x & 0x7000007000000000) >> 9) | ((x & 0x1c00001c0fc0000000) >> 18) | ((x & 0x7000) << 33) | ((x & 0x1f8000) << 24) | ((x & 0xe00000) << 15) | ((x & 0x7000000000000) >> 39) | ((x & 0x38000000000000) >> 45) | ((x & 0x1c0000000000000) >> 51) | ((x & 0xe00000000000000) >> 57);
    }
    actual_color_pattern = ((x & 0xe00000000000000000) >> 63) | ((x & 0x38000000000000) >> 48) | ((x & 0x1c0000) >> 18);
    if actual_color_pattern == 305 {
        return ((x & 0x38000) << 54) | ((x & 0xe00007) << 45) | ((x & 0x7000) << 51) | ((x & 0x1c0000) << 42) | ((x & 0x7000000000) << 21) | ((x & 0x1f8000000000) << 12) | ((x & 0xe00000000000) << 3) | ((x & 0x1f8) << 36) | ((x & 0xe00) << 27) | ((x & 0x7000000000000000) >> 27) | ((x & 0x1f8000000000000000) >> 36) | ((x & 0xe00000000000000000) >> 45) | ((x & 0x1c7000000) >> 9) | ((x & 0xe38000000) >> 15) | ((x & 0xfff000000000000) >> 48);
    }
    if actual_color_pattern == 396 {
        return ((x & 0x1c00001c0000000) << 15) | ((x & 0x700000703f000) << 18) | ((x & 0xe00000e00fc0000) << 6) | ((x & 0x38000038000000) << 9) | ((x & 0x7) << 57) | ((x & 0x38) << 51) | ((x & 0x1c0) << 45) | ((x & 0xe00) << 39) | ((x & 0x7000000000) >> 15) | ((x & 0x1f8000000000) >> 24) | ((x & 0xe00000000000) >> 33) | ((x & 0x1c0000000000000000) >> 57) | ((x & 0x7000000000000000) >> 54) | ((x & 0xe00000000000000000) >> 66) | ((x & 0x38000000000000000) >> 63);
    }
    if actual_color_pattern == 102 {
        return ((x & 0x70000070000001c0) << 9) | ((x & 0x38000038000000e00) << 3) | ((x & 0x1c00001c0000007000) >> 3) | ((x & 0xe00000e00000038000) >> 9) | ((x & 0x1c7000000) << 27) | ((x & 0xe38000007) << 21) | ((x & 0x1c7000000e00000) >> 21) | ((x & 0xe38000000000000) >> 27) | ((x & 0x38) << 15) | ((x & 0x1c0000) >> 15);
    }
    actual_color_pattern = ((x & 0x38000000000) >> 33) | ((x & 0xe00000000) >> 30) | ((x & 0x38000) >> 15);
    if actual_color_pattern == 102 {
        return ((x & 0x1c7000000000) << 27) | ((x & 0xe38000000000) << 21) | ((x & 0x7000007000000) << 9) | ((x & 0x38000038000000) << 3) | ((x & 0x1c00001c0000000) >> 3) | ((x & 0xe00000e00000000) >> 9) | ((x & 0x1c7000000000000000) >> 21) | ((x & 0xe38000000000000000) >> 27) | ((x & 0xfff) << 12) | ((x & 0xfff000) >> 12);
    }
    if actual_color_pattern == 305 {
        return ((x & 0x1c0000) << 51) | ((x & 0x7000) << 54) | ((x & 0xe00000) << 42) | ((x & 0x38000) << 45) | ((x & 0x1c00001c0000000000) >> 9) | ((x & 0x7000007000000000) >> 6) | ((x & 0xe00000e0003f000000) >> 18) | ((x & 0x38000038000000000) >> 15) | ((x & 0xe07) << 36) | ((x & 0x38) << 39) | ((x & 0x1c0) << 33) | ((x & 0x7000000000000) >> 27) | ((x & 0x38000000000000) >> 33) | ((x & 0x1c0000000000000) >> 39) | ((x & 0xe00000000000000) >> 45) | ((x & 0xfc0000000) >> 30);
    }
    if actual_color_pattern == 396 {
        return ((x & 0x7000000) << 45) | ((x & 0x1f8000000) << 36) | ((x & 0xe00000000) << 27) | ((x & 0xfff) << 48) | ((x & 0x7000000000000) >> 3) | ((x & 0x1f8000000000000) >> 12) | ((x & 0xe00000000000000) >> 21) | ((x & 0x1c7000) << 15) | ((x & 0xe38000) << 9) | ((x & 0x1c0000e00000000000) >> 45) | ((x & 0x7000000000000000) >> 42) | ((x & 0xe00000000000000000) >> 54) | ((x & 0x38000000000000000) >> 51) | ((x & 0x7000000000) >> 27) | ((x & 0x1f8000000000) >> 36);
    }
    actual_color_pattern = ((x & 0xe00000000000) >> 39) | ((x & 0xe00000000000000) >> 54) | ((x & 0xe00000) >> 21);
    if actual_color_pattern == 116 {
        return ((x & 0x7000000000) << 33) | ((x & 0x1f8000000000) << 24) | ((x & 0xe00000000000) << 15) | ((x & 0x38) << 54) | ((x & 0xe00) << 45) | ((x & 0x7) << 51) | ((x & 0x1c0) << 42) | ((x & 0x7000000000000000) >> 15) | ((x & 0x1f8000000000000000) >> 24) | ((x & 0xe00000000000000000) >> 33) | ((x & 0x7000) << 21) | ((x & 0x1f8000) << 12) | ((x & 0xe00000) << 3) | ((x & 0x7000000) >> 3) | ((x & 0x1f8000000) >> 12) | ((x & 0xe00000000) >> 21) | ((x & 0x1c0000000000000) >> 45) | ((x & 0x7000000000000) >> 42) | ((x & 0xe00000000000000) >> 54) | ((x & 0x38000000000000) >> 51);
    }
    if actual_color_pattern == 417 {
        return ((x & 0x7000) << 57) | ((x & 0x38000) << 51) | ((x & 0x1c0000) << 45) | ((x & 0xe00000) << 39) | ((x & 0x7000000) << 33) | ((x & 0x1f8000000) << 24) | ((x & 0xe00000000) << 15) | ((x & 0x3f) << 42) | ((x & 0xfc0) << 30) | ((x & 0x7000000000000) >> 15) | ((x & 0x1f8000000000000) >> 24) | ((x & 0xe00000000000000) >> 33) | ((x & 0x7000000000000000) >> 39) | ((x & 0x38000000000000000) >> 45) | ((x & 0x1c0000000000000000) >> 51) | ((x & 0xe00000000000000000) >> 57) | ((x & 0x3f000000000) >> 30) | ((x & 0xfc0000000000) >> 42);
    }
    if actual_color_pattern == 270 {
        return ((x & 0x7000007000000) << 21) | ((x & 0x38000038000007) << 15) | ((x & 0x1c00001c0000e00) << 9) | ((x & 0xe00000e00000000) << 3) | ((x & 0x7000007000000000) >> 3) | ((x & 0x380000380001c0000) >> 9) | ((x & 0x1c00001c0000038000) >> 15) | ((x & 0xe00000e00000000000) >> 21) | ((x & 0x38) << 18) | ((x & 0x1c0) << 6) | ((x & 0x7000) >> 6) | ((x & 0xe00000) >> 18);
    }
    panic!("State was not possible to orient: {}", x);
}

fn lu(x: i128) -> i128 {
    return ((x & 0x38000000) << 42) | ((x & 0x1c71c71c71c7000fff) << 0) | ((x & 0xe00000000) << 30) | ((x & 0xe38000e38000000000) >> 12) | ((x & 0x38000000e00000) >> 6) | ((x & 0xe00000000000000) >> 18) | ((x & 0x1c0000) << 3) | ((x & 0x7000) << 6) | ((x & 0x38000) >> 3);
}

fn ld(x: i128) -> i128 {
    return ((x & 0xe38000e38000000) << 12) | ((x & 0x1c71c71c71c7000fff) << 0) | ((x & 0x38000000000) << 18) | ((x & 0xe00000038000) << 6) | ((x & 0x38000000000000000) >> 30) | ((x & 0xe00000000000000000) >> 42) | ((x & 0xe00000) >> 3) | ((x & 0x7000) << 3) | ((x & 0x1c0000) >> 6);
}

fn ru(x: i128) -> i128 {
    return ((x & 0xe38e38e38e38fff000) << 0) | ((x & 0x7000000) << 42) | ((x & 0x1c0000000) << 30) | ((x & 0x1c70001c7000000000) >> 12) | ((x & 0x70000000001c0) >> 6) | ((x & 0x1c0000000000000) >> 18) | ((x & 0x38) << 6) | ((x & 0xe00) >> 3) | ((x & 0x7) << 3);
}

fn rd(x: i128) -> i128 {
    return ((x & 0xe38e38e38e38fff000) << 0) | ((x & 0x1c70001c7000000) << 12) | ((x & 0x7000000000) << 18) | ((x & 0x1c0000000007) << 6) | ((x & 0x7000000000000000) >> 30) | ((x & 0x1c0000000000000000) >> 42) | ((x & 0x1c0) << 3) | ((x & 0xe00) >> 6) | ((x & 0x38) >> 3);
}

fn fl(x: i128) -> i128 {
    return ((x & 0x1c0000000000000000) << 3) | ((x & 0x7000000000000000) << 6) | ((x & 0xe00000000000000000) >> 6) | ((x & 0x38000000000000000) >> 3) | ((x & 0xfc0ffffc0e381c7) << 0) | ((x & 0xe00) << 42) | ((x & 0x38) << 45) | ((x & 0x1c0000) << 9) | ((x & 0x7000) << 12) | ((x & 0x7000000000000) >> 30) | ((x & 0x38000000000000) >> 39) | ((x & 0x7000000) >> 15) | ((x & 0x38000000) >> 24);
}

fn fr(x: i128) -> i128 {
    return ((x & 0x38000000000000000) << 6) | ((x & 0xe00000000000000000) >> 3) | ((x & 0x7000000000000000) << 3) | ((x & 0x1c0000000000000000) >> 6) | ((x & 0xfc0ffffc0e381c7) << 0) | ((x & 0x7000) << 39) | ((x & 0x1c0000) << 30) | ((x & 0x38) << 24) | ((x & 0xe00) << 15) | ((x & 0x38000000) >> 9) | ((x & 0x7000000) >> 12) | ((x & 0x38000000000000) >> 42) | ((x & 0x7000000000000) >> 45);
}

fn bl(x: i128) -> i128 {
    return ((x & 0xfff03f00003f1c7e38) << 0) | ((x & 0x1c0) << 51) | ((x & 0x7) << 54) | ((x & 0x1c0000000000) << 3) | ((x & 0x7000000000) << 6) | ((x & 0xe00000000000) >> 6) | ((x & 0x38000000000) >> 3) | ((x & 0xe00000) << 12) | ((x & 0x38000) << 15) | ((x & 0x1c0000e00000000) >> 33) | ((x & 0xe00000000000000) >> 42) | ((x & 0x1c0000000) >> 24);
}

fn br(x: i128) -> i128 {
    return ((x & 0xfff03f00003f1c7e38) << 0) | ((x & 0x38000) << 42) | ((x & 0xe00007) << 33) | ((x & 0x38000000000) << 6) | ((x & 0xe00000000000) >> 3) | ((x & 0x7000000000) << 3) | ((x & 0x1c0000000000) >> 6) | ((x & 0x1c0) << 24) | ((x & 0xe00000000) >> 12) | ((x & 0x1c0000000) >> 15) | ((x & 0xe00000000000000) >> 51) | ((x & 0x1c0000000000000) >> 54);
}

fn ul(x: i128) -> i128 {
    return ((x & 0xfc0) << 60) | ((x & 0x3f00003ffff03f03f) << 0) | ((x & 0x38000000000000) << 6) | ((x & 0xe00000000000000) >> 3) | ((x & 0x7000000000000) << 3) | ((x & 0x1c0000000000000) >> 6) | ((x & 0x1c0000) << 27) | ((x & 0xe00000) << 21) | ((x & 0xfc0000000000000000) >> 48) | ((x & 0x1c0000000000) >> 33) | ((x & 0xe00000000000) >> 39);
}

fn ur(x: i128) -> i128 {
    return ((x & 0xfc0000) << 48) | ((x & 0x3f00003ffff03f03f) << 0) | ((x & 0x1c0000000000000) << 3) | ((x & 0x7000000000000) << 6) | ((x & 0xe00000000000000) >> 6) | ((x & 0x38000000000000) >> 3) | ((x & 0x1c0) << 39) | ((x & 0xe00) << 33) | ((x & 0x1c0000000000) >> 21) | ((x & 0xe00000000000) >> 27) | ((x & 0xfc0000000000000000) >> 60);
}

fn dl(x: i128) -> i128 {
    return ((x & 0xfc0ffffc0000fc0fc0) << 0) | ((x & 0x3f) << 60) | ((x & 0x7000) << 27) | ((x & 0x38000) << 21) | ((x & 0x38000000) << 6) | ((x & 0xe00000000) >> 3) | ((x & 0x7000000) << 3) | ((x & 0x1c0000000) >> 6) | ((x & 0x3f000000000000000) >> 48) | ((x & 0x7000000000) >> 33) | ((x & 0x38000000000) >> 39);
}

fn dr(x: i128) -> i128 {
    return ((x & 0xfc0ffffc0000fc0fc0) << 0) | ((x & 0x3f000) << 48) | ((x & 0x7) << 39) | ((x & 0x38) << 33) | ((x & 0x1c0000000) << 3) | ((x & 0x7000000) << 6) | ((x & 0xe00000000) >> 6) | ((x & 0x38000000) >> 3) | ((x & 0x7000000000) >> 21) | ((x & 0x38000000000) >> 27) | ((x & 0x3f000000000000000) >> 60);
}
