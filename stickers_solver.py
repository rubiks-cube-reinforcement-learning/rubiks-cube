import random
from stickers_bitwise_ops import FIXED_CUBIE_OPS, OPS, SOLVED_CUBE_STATE, orient_cube

LOOKUP = {}


def load_lookup_table():
    if len(LOOKUP) > 0:
        return
    with open('rust-experiment/results-cubies-fixed.txt') as fp:
        for line in fp:
            moves, binary_rep = line.strip().split(' ')
            state = int(binary_rep, 2)
            moves_nb = int(moves)
            LOOKUP[state] = moves_nb


def get_scrambled_state_with_cubie_4_fixed(scrambles):
    return get_scrambled_state(scrambles, moves=FIXED_CUBIE_OPS)


def get_scrambled_state(scrambles, moves=OPS):
    state = SOLVED_CUBE_STATE
    op = last_op = None
    for i in range(scrambles):
        while op is last_op:
            op = random.choice(moves)
        state = op(state)
        last_op = op
    return state


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


def generate_dataset(nb_per_scramble, max_scrambles, scramble_fn=get_scrambled_state):
    dataset = []
    for i in range(nb_per_scramble):
        for scrambles in range(max_scrambles):
            dataset.append(scramble_fn(scrambles))
    return dataset


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
