# coding=utf-8

import json
from itertools import permutations
import numpy as np

def compute_possible_2x2x2_states():
    """
    See this Quora answer:
    https://www.quora.com/How-can-we-calculate-the-number-of-permutations-of-a-2x2x2-Rubik’s-cube-Can-you-clearly-explain-how

    :return:
    """
    cubies = [1, 2, 3, 4, 5, 6, 7, 8]
    possible_states = set()
    for permutation in permutations(cubies):
        for i in range(3**6):
            rotations = [int(c) for c in np.base_repr(i, base=3).zfill(7)]
            raw_state = list(zip(permutation, rotations))
            vector_state = []
    return list(possible_states)

def solve_cube(state):
    pass

with open("./2x2x2-states.json", "w+") as fp:
    json.dump(compute_possible_2x2x2_states(), fp)

# print("I %d" % i)
if __name__ == '__main__':
    pass
