import json
from itertools import chain
from code_generator.common import compute_stickers_permutation_per_operation
from cube3.cube import Cube3


def indices_to_one_hot(indices):
    return list(chain(*[range(i*6, i*6+6) for i in indices]))


def generate_3cube_numba_code():
    moves = "\n".join(['        ' + line for line in generate_moves_code()])
    moves_indices = ", ".join([f"{spec[0]}: '{name}'" for name, spec in cube_3_permutations.items()])
    moves_names = ", ".join([f"'{name}': {spec[0]}" for name, spec in cube_3_permutations.items()])
    return f'''
from numba import cuda, float32
import numpy as np
import math 

@cuda.jit
def apply_moves(cubes, buffer, moves):
    x = cuda.grid(1)

    if x >= cubes.shape[0]:
        return
    for move in moves[x]:
        if move == 0:
            break
        for i in range(324):
            buffer[x][i] = cubes[x][i]
{moves}


MOVES_INDICES = {{{moves_indices}}}
MOVES_NAMES = {{{moves_names}}}
'''


cube_3_permutations = {
    name: (i+1, indices_to_one_hot(indices))
    for i, (name, indices) in enumerate(compute_stickers_permutation_per_operation(Cube3).items())
}

def generate_moves_code():
    lines = []
    for name, spec in cube_3_permutations.items():
        lines.append(f"elif move == {spec[0]}:")
        indices = spec[1]
        for _from, _to in enumerate(indices):
            lines.append(f"    cubes[x][{_to}] = buffer[x][{_from}]")
    lines[0] = lines[0][2:]
    return lines

