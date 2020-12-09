import random
from stickers_bitwise_ops import FIXED_CUBIE_OPS, SOLVED_CUBE_STATE

LOOKUP = {}

def load_lookup_table():
    if len(LOOKUP) > 0:
        return
    with open('./rust-experiment/results-cubies-fixed-bak.txt') as fp:
        for line in fp:
            moves, binary_rep = line.strip().split(' ')
            state = int(binary_rep, 2)
            moves_nb = int(moves)
            LOOKUP[state] = moves_nb


def get_scrambled_state(scrambles):
    state = SOLVED_CUBE_STATE
    op = last_op = None
    for i in range(scrambles):
        while op is last_op:
            op = random.choice(FIXED_CUBIE_OPS)
        state = op(state)
        last_op = op
    return state


def find_solution(state):
    load_lookup_table()
    path = []
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


def generate_dataset(nb_per_scramble, max_scrambles):
    dataset = []
    for i in range(nb_per_scramble):
        for scrambles in range(max_scrambles):
            dataset.append(get_scrambled_state(scrambles))
    return dataset


if __name__ == '__main__':
    dataset = generate_dataset(10000, 100)
    # load_lookup_table()
    # for example in dataset:
    #     find_solution(example)
