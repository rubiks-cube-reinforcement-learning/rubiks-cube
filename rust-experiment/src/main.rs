use std::collections::VecDeque;
use std::collections::HashMap;
use std::fs;
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use rand::Rng;
use rayon::prelude::*;
use itertools::iproduct;

fn main() {
    let dataset = generate_dataset(10000, 100);
    println!("{}", dataset.len());
}

fn generate_dataset(nb_per_scramble:i16, max_scrambles:i16) -> Vec<i128> {
    iproduct!(0..nb_per_scramble, 0..max_scrambles)
        .par_bridge()
        .map(|(n, scrambles)| { generate_scrambled_state(scrambles) })
        .collect::<Vec<i128>>()
}

static SOLVED_STATE:i128 = 0x2494926db924b6ddb6;

fn generate_scrambled_state(scrambles:i16) -> i128 {
    let mut state = SOLVED_STATE;
    let mut op = 100;
    let mut op_idx = 100;
    for step in 0..scrambles {
        loop {
            if op != op_idx {
                op = op_idx;
                break;
            }
            op_idx = rand::thread_rng().gen_range(0, FIXED_CUBIE_OPERATIONS.len());
        }
        state = FIXED_CUBIE_OPERATIONS[op_idx](state);
    }
    state
}

fn generate_state_lookup_table() {
    let mut stack: VecDeque<(i128, i8)> = VecDeque::new();
	let mut lookup = HashMap::new();

	stack.push_back((SOLVED_STATE, 0));
	lookup.insert(SOLVED_STATE, 0);
	println!("{}", lookup.len());

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
			stack.push_back((new_state, distance+1));
			lookup.insert(new_state, distance+1);
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
    fs::write("./results-cubies-fixed.txt", data).expect("Unable to write file");
}

// 101101011101101011010011100100110110001100010010001001010110100110011001
// 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
static ALL_OPERATIONS: &'static [fn(i128) -> i128] = &[
    // Something is broken here - it goes beyond 29 million states, why?
    lu,ld,ru,rd,fl,fr,ul,ur,dl,dr,bl,br,
];

static FIXED_CUBIE_OPERATIONS: &'static [fn(i128) -> i128] = &[
    // This finishes at 3674026 states as it's supposed to
    lu,ld,            ul,ur,      bl,br,
];

// static FIXED_CUBIE_OPERATIONS: &'static [fn(i128) -> i128] = &[
//    // This finishes at 3674026 states as it's supposed to
//           ru,rd,fl,fr,      dl,dr,
// ];

fn lu(x: i128) -> i128 {
    return ((x & 0x38000000) << 42)|((x & 0x1c71c71c71c7000fff) << 0)|((x & 0xe00000000) << 30)|((x & 0xe38000e38000000000) >> 12)|((x & 0x38000000e00000) >> 6)|((x & 0xe00000000000000) >> 18)|((x & 0x1c0000) << 3)|((x & 0x7000) << 6)|((x & 0x38000) >> 3);
}

fn ld(x: i128) -> i128 {
    return ((x & 0xe38000e38000000) << 12)|((x & 0x1c71c71c71c7000fff) << 0)|((x & 0x38000000000) << 18)|((x & 0xe00000038000) << 6)|((x & 0x38000000000000000) >> 30)|((x & 0xe00000000000000000) >> 42)|((x & 0xe00000) >> 3)|((x & 0x7000) << 3)|((x & 0x1c0000) >> 6);
}

fn ru(x: i128) -> i128 {
    return ((x & 0xe38e38e38e38fff000) << 0)|((x & 0x7000000) << 42)|((x & 0x1c0000000) << 30)|((x & 0x1c70001c7000000000) >> 12)|((x & 0x70000000001c0) >> 6)|((x & 0x1c0000000000000) >> 18)|((x & 0x38) << 6)|((x & 0xe00) >> 3)|((x & 0x7) << 3);
}

fn rd(x: i128) -> i128 {
    return ((x & 0xe38e38e38e38fff000) << 0)|((x & 0x1c70001c7000000) << 12)|((x & 0x7000000000) << 18)|((x & 0x1c0000000007) << 6)|((x & 0x7000000000000000) >> 30)|((x & 0x1c0000000000000000) >> 42)|((x & 0x1c0) << 3)|((x & 0xe00) >> 6)|((x & 0x38) >> 3);
}

fn fl(x: i128) -> i128 {
    return ((x & 0x1c0000000000000000) << 3)|((x & 0x7000000000000000) << 6)|((x & 0xe00000000000000000) >> 6)|((x & 0x38000000000000000) >> 3)|((x & 0xfc0ffffc0e381c7) << 0)|((x & 0xe00) << 42)|((x & 0x38) << 45)|((x & 0x1c0000) << 9)|((x & 0x7000) << 12)|((x & 0x7000000000000) >> 30)|((x & 0x38000000000000) >> 39)|((x & 0x7000000) >> 15)|((x & 0x38000000) >> 24);
}

fn fr(x: i128) -> i128 {
    return ((x & 0x38000000000000000) << 6)|((x & 0xe00000000000000000) >> 3)|((x & 0x7000000000000000) << 3)|((x & 0x1c0000000000000000) >> 6)|((x & 0xfc0ffffc0e381c7) << 0)|((x & 0x7000) << 39)|((x & 0x1c0000) << 30)|((x & 0x38) << 24)|((x & 0xe00) << 15)|((x & 0x38000000) >> 9)|((x & 0x7000000) >> 12)|((x & 0x38000000000000) >> 42)|((x & 0x7000000000000) >> 45);
}

fn bl(x: i128) -> i128 {
    return ((x & 0xfff03f00003f1c7e38) << 0)|((x & 0x1c0) << 51)|((x & 0x7) << 54)|((x & 0x1c0000000000) << 3)|((x & 0x7000000000) << 6)|((x & 0xe00000000000) >> 6)|((x & 0x38000000000) >> 3)|((x & 0xe00000) << 12)|((x & 0x38000) << 15)|((x & 0x1c0000e00000000) >> 33)|((x & 0xe00000000000000) >> 42)|((x & 0x1c0000000) >> 24);
}

fn br(x: i128) -> i128 {
    return ((x & 0xfff03f00003f1c7e38) << 0)|((x & 0x38000) << 42)|((x & 0xe00007) << 33)|((x & 0x38000000000) << 6)|((x & 0xe00000000000) >> 3)|((x & 0x7000000000) << 3)|((x & 0x1c0000000000) >> 6)|((x & 0x1c0) << 24)|((x & 0xe00000000) >> 12)|((x & 0x1c0000000) >> 15)|((x & 0xe00000000000000) >> 51)|((x & 0x1c0000000000000) >> 54);
}

fn ul(x: i128) -> i128 {
    return ((x & 0xfc0) << 60)|((x & 0x3f00003ffff03f03f) << 0)|((x & 0x38000000000000) << 6)|((x & 0xe00000000000000) >> 3)|((x & 0x7000000000000) << 3)|((x & 0x1c0000000000000) >> 6)|((x & 0x1c0000) << 27)|((x & 0xe00000) << 21)|((x & 0xfc0000000000000000) >> 48)|((x & 0x1c0000000000) >> 33)|((x & 0xe00000000000) >> 39);
}

fn ur(x: i128) -> i128 {
    return ((x & 0xfc0000) << 48)|((x & 0x3f00003ffff03f03f) << 0)|((x & 0x1c0000000000000) << 3)|((x & 0x7000000000000) << 6)|((x & 0xe00000000000000) >> 6)|((x & 0x38000000000000) >> 3)|((x & 0x1c0) << 39)|((x & 0xe00) << 33)|((x & 0x1c0000000000) >> 21)|((x & 0xe00000000000) >> 27)|((x & 0xfc0000000000000000) >> 60);
}

fn dl(x: i128) -> i128 {
    return ((x & 0xfc0ffffc0000fc0fc0) << 0)|((x & 0x3f) << 60)|((x & 0x7000) << 27)|((x & 0x38000) << 21)|((x & 0x38000000) << 6)|((x & 0xe00000000) >> 3)|((x & 0x7000000) << 3)|((x & 0x1c0000000) >> 6)|((x & 0x3f000000000000000) >> 48)|((x & 0x7000000000) >> 33)|((x & 0x38000000000) >> 39);
}

fn dr(x: i128) -> i128 {
    return ((x & 0xfc0ffffc0000fc0fc0) << 0)|((x & 0x3f000) << 48)|((x & 0x7) << 39)|((x & 0x38) << 33)|((x & 0x1c0000000) << 3)|((x & 0x7000000) << 6)|((x & 0xe00000000) >> 6)|((x & 0x38000000) >> 3)|((x & 0x7000000000) >> 21)|((x & 0x38000000000) >> 27)|((x & 0x3f000000000000000) >> 60);
}