import pickle
from pathlib import Path

from binary_code_generation import PythonCodeGenerator, RustCodeGenerator
from cube2.solver import find_solution, load_lookup_table, LOOKUP
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


def precompute_all_cube2_moves(out_path=Path(__file__).parent / "all-moves.pickle"):
    load_lookup_table()

    all_moves = {}
    for i, (state, distance) in enumerate(list(LOOKUP.items())):
        neighbors = [op(state) for op in ops2]
        weights = [LOOKUP[new_state] for new_state in neighbors]
        moves = list(zip(ops2.keys(), weights, neighbors))
        optimal = min(moves, key=lambda move: move[1])
        all_moves[state] = {
            "distance": distance,
            "moves": moves,
            "optimal": optimal
        }
        if i % 100000 == 0:
            print(i)

    with out_path.open('wb+') as fp:
        pickle.dump(all_moves, fp)
    return all_moves


def load_precomputed_cube2_moves(path=Path(__file__).parent / "all-moves.pickle"):
    with path.open('rb+') as fp:
        return pickle.load(fp)


def precompute_cube2_solutions(precomputed_moves):
    solutions = {}
    for loaded_state, data in list(precomputed_moves.items())[:]:
        solution = []
        next = data
        while next['optimal'][1] < next['distance']:
            solution.append(next['optimal'][0])
            next = precomputed_moves[next['optimal'][2]]
        solutions[loaded_state] = dict(solution=solution, **data)
    return solutions


def solve_2_cube_precomputed(oriented_state, precomputed_solutions):
    return precomputed_solutions[oriented_state]['solution']


def solve_3_cube_corners(cube3_scrambled, cube2_solve_fn=find_solution):
    cube3_oriented = orient_cube_3(cube3_scrambled)
    cube2 = cube3_to_cube2(cube3_oriented)
    corners_solution = cube2_solve_fn(cube2)
    for move in corners_solution:
        cube3_oriented = ops3[move](cube3_oriented)
    return cube3_oriented


def fun():
    logger.info("Loading precomputed moves")
    moves = load_precomputed_cube2_moves()

    logger.info("Precomputing all solutions")
    solutions = precompute_cube2_solutions(moves)
    solve_precomputed = lambda oriented_state: solve_2_cube_precomputed(oriented_state, solutions)

    logger.info("Generating dataset")
    dataset_3 = generate_binary_dataset_3(1000, 100)
    logger.info("Solving dataset of size %d", len(dataset_3))
    for cube3_scrambled in dataset_3:
        cube3_solved_corners = solve_3_cube_corners(cube3_scrambled, solve_precomputed)
    logger.info("Solved!")


if __name__ == '__main__':
    # refresh_bitwise_ops_code()
    # precompute_all_cube2_moves()
    fun()
