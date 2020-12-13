import pickle
from pathlib import Path

from cube2.generated_stickers_bitwise_ops import FIXED_CUBIE_OPS, OPS, SOLVED_CUBE_STATE, orient_cube
from utils import generate_dataset, scramble

LOOKUP = {}

default_table_path = (Path(__file__).parent.parent / "rust-experiment/results-cubies-fixed.txt").__str__()
def load_lookup_table(path: str = default_table_path):
    if len(LOOKUP) > 0:
        return
    with open(path) as fp:
        for line in fp:
            moves, binary_rep = line.strip().split(' ')
            state = int(binary_rep, 2)
            moves_nb = int(moves)
            LOOKUP[state] = moves_nb


def generate_binary_dataset(nb_per_scramble, max_scrambles, fixed_cubie=False):
    scramble_fn = lambda n: scramble(SOLVED_CUBE_STATE, n, FIXED_CUBIE_OPS if fixed_cubie else OPS)
    return generate_dataset(nb_per_scramble, max_scrambles, scramble_fn)


def find_solution(state):
    load_lookup_table()
    path = []
    if state not in LOOKUP:
        state = orient_cube(state)

    if state not in LOOKUP:
        raise Exception("State not in lookup!")

    distance = LOOKUP[state]
    for i in range(distance):
        for op in FIXED_CUBIE_OPS:
            new_state = op(state)
            new_distance = LOOKUP[new_state]
            if new_distance < distance:
                path.append(op.__name__)
                state, distance = new_state, new_distance
                break
        else:
            raise Exception("Did not find any move leading to a shorter distance")
    return path


def precompute_all_moves(out_path=Path(__file__).parent / "all-moves.pickle"):
    load_lookup_table()

    all_moves = {}
    for i, (state, distance) in enumerate(list(LOOKUP.items())):
        neighbors = [op(state) for op in FIXED_CUBIE_OPS]
        weights = [LOOKUP[new_state] for new_state in neighbors]
        moves = list(zip(FIXED_CUBIE_OPS.keys(), weights, neighbors))
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


def load_precomputed_moves(path=Path(__file__).parent / "all-moves.pickle"):
    with path.open('rb+') as fp:
        return pickle.load(fp)


def precompute_solutions(precomputed_moves):
    solutions = {}
    for loaded_state, data in list(precomputed_moves.items())[:]:
        solution = []
        next = data
        while next['optimal'][1] < next['distance']:
            solution.append(next['optimal'][0])
            next = precomputed_moves[next['optimal'][2]]
        solutions[loaded_state] = dict(solution=solution, **data)
    return solutions


def find_solution_precomputed(oriented_state, precomputed_solutions):
    return precomputed_solutions[oriented_state]['solution']


if __name__ == '__main__':
    print("Loading lookup table")
    load_lookup_table()

    print("Generating dataset")
    dataset = generate_dataset(100, 100)

    print("Solving cubes")
    for i in range(100):
        for example in dataset:
            find_solution(example)
    print("Finished")
