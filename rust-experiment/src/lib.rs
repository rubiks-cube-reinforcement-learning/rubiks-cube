use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
extern crate fxhash;
use fxhash::FxHashMap;
#[path = "solve2.rs"] mod solve2;

#[pyfunction]
fn canary() -> PyResult<u64> {
    Ok(0u64)
}

static mut LOOKUP : Option<FxHashMap<i128, i8>> = None;

#[pyfunction]
fn load_lookup_table(path: &str) -> PyResult<u64> {
    let lookup_table_loaded : bool;
    unsafe {
        lookup_table_loaded = match &LOOKUP {
            Some(map) => map.len() > 0,
            None => false
        };
    }
    if lookup_table_loaded {
        return Ok(0);
    }
    unsafe {
        LOOKUP = Some(solve2::load_lookup_table(path));
    }
    return Ok(0);
}

#[pyfunction]
fn len_lookup_table() -> PyResult<u64> {
    let result: u64;
    unsafe {
        result = match &LOOKUP {
            Some(map) => map.len() as u64,
            None => 0
        };
    }
    return Ok(result);
}

#[pyfunction]
fn solve_batch<'a>(_py: Python<'a>, batch_states: Vec<i128>) -> PyResult<&'a PyList> {
    let lookup;
    unsafe {
        lookup = match &LOOKUP {
            Some(map) => map,
            None => panic!("Lookup not loaded")
        };
    }

    let results = batch_states
        .par_iter()
        .map(|state| { solve2::solve_cube(*state, &lookup) })
        .collect::<Vec<Vec<&str>>>();

    let list : &PyList = PyList::new(_py, results);

    return Ok(list);
}

#[pymodule]
fn rubiks_cube_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(canary))?;
    m.add_wrapped(wrap_pyfunction!(load_lookup_table))?;
    m.add_wrapped(wrap_pyfunction!(len_lookup_table))?;
    m.add_wrapped(wrap_pyfunction!(solve_batch))?;

    Ok(())
}

