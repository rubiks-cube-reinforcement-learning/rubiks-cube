import pickle
from pathlib import Path

from binary_code_generation import PythonCodeGenerator, RustCodeGenerator
from cube2.solver import find_solution, load_lookup_table as load_lookup_table_2, LOOKUP, load_precomputed_moves as load_precomputed_cube2_moves, \
    precompute_solutions as precompute_cube2_solutions, find_solution_precomputed as solve_2_cube_precomputed, \
    precompute_all_moves as precompute_all_cube2_moves, generate_binary_dataset as generate_binary_dataset_2
from cube2.cube import Cube2
from cube3.cube import Cube3
from cube3.solver import generate_binary_dataset as generate_binary_dataset_3
from cube3.generated_stickers_bitwise_ops import orient_cube as orient_cube_3, \
    cube3_to_cube2, FIXED_CUBIE_OPS_DICT as ops3
from cube2.generated_stickers_bitwise_ops import orient_cube as orient_cube_2, \
    FIXED_CUBIE_OPS_DICT as ops2, SOLVED_CUBE_STATE as solved2
from loggers import getLogger
logger = getLogger(__name__)


def refresh_bitwise_ops_code():
    base = Path(__file__).parent
    with Path(base / "cube2/generated_stickers_bitwise_ops.py").open('w+') as fp:
        fp.write(PythonCodeGenerator(Cube2).build())

    with Path(base / "cube3/generated_stickers_bitwise_ops.py").open('w+') as fp:
        fp.write(PythonCodeGenerator(Cube3).build())

    with Path(base / "rust-experiment/src/cube2.rs").open('w+') as fp:
        fp.write(RustCodeGenerator(Cube2).build())

    with Path(base / "rust-experiment/src/cube3.rs").open('w+') as fp:
        fp.write(RustCodeGenerator(Cube3).build())


def solve_3_cube_corners(cube3_scrambled, cube2_solve_fn=find_solution):
    cube3_oriented = orient_cube_3(cube3_scrambled)
    cube2 = cube3_to_cube2(cube3_oriented)
    corners_solution = cube2_solve_fn(cube2)
    for move in corners_solution:
        cube3_oriented = ops3[move](cube3_oriented)
    return cube3_oriented

DATASET_EXAMPLES_PER_SCRAMBLE = 1000
DATASET_SCRAMBLES = 100
N_REPEAT = 10

def bench_python():
    logger.info("============= bench_python =============")
    logger.info("Loading lookup table")
    load_lookup_table_2("./rust-experiment/results-cubies-fixed.txt")

    logger.info("Generating dataset")
    dataset_3 = generate_binary_dataset_3(DATASET_EXAMPLES_PER_SCRAMBLE, DATASET_SCRAMBLES)
    dataset_size = len(dataset_3)
    logger.info("Solving dataset of size %d, repeating %d times for a total of %d states", dataset_size, N_REPEAT, dataset_size * N_REPEAT)
    for i in range(N_REPEAT):
        if i % 10 == 0:
            logger.info("...iteration %d", i)
        for cube3_scrambled in dataset_3:
            cube3_solved_corners = solve_3_cube_corners(cube3_scrambled)
    logger.info("Solved!")


def bench_precomputed():
    # logger.info("Generating precomputed moves")
    # precompute_all_cube2_moves()

    logger.info("============= bench_precomputed =============")
    logger.info("Loading precomputed moves")
    moves = load_precomputed_cube2_moves()

    logger.info("Precomputing all solutions")
    solutions = precompute_cube2_solutions(moves)
    solve_precomputed = lambda oriented_state: solve_2_cube_precomputed(oriented_state, solutions)

    logger.info("Generating dataset")
    dataset_3 = generate_binary_dataset_3(DATASET_EXAMPLES_PER_SCRAMBLE, DATASET_SCRAMBLES)
    dataset_size = len(dataset_3)
    logger.info("Solving dataset of size %d, repeating %d times for a total of %d states", dataset_size, N_REPEAT, dataset_size * N_REPEAT)
    for i in range(N_REPEAT):
        if i % 10 == 0:
            logger.info("...iteration %d", i)
        for cube3_scrambled in dataset_3:
            cube3_solved_corners = solve_3_cube_corners(cube3_scrambled, solve_precomputed)
    logger.info("Solved!")


def bench_rust():
    """
    To use this, first go to rust-experiment and run build.sh to build the rust module
    """
    logger.info("============= bench_rust =============")
    logger.info("Loading rust solver")
    from rubiks_cube_rust import load_lookup_table as load_lookup_table_rust, solve_batch as solve_batch_rust
    logger.info("Loading lookup table")
    load_lookup_table_rust("./rust-experiment/results-cubies-fixed.txt")
    logger.info("Generating dataset")
    dataset_2 = generate_binary_dataset_2(DATASET_EXAMPLES_PER_SCRAMBLE, DATASET_SCRAMBLES)
    dataset_size = len(dataset_2)
    logger.info("Solving dataset of size %d, repeating %d times for a total of %d states", dataset_size, N_REPEAT, dataset_size * N_REPEAT)
    for i in range(N_REPEAT):
        if i % 10 == 0:
            logger.info("...iteration %d", i)
        solve_batch_rust(dataset_2)

    logger.info("Solved!")
