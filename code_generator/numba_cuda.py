from itertools import chain
from code_generator.common import compute_stickers_permutation_per_operation
from cube3.cube import Cube3


def indices_to_one_hot(indices):
    return list(chain(*[range(i*6, i*6+6) for i in indices]))


def generate_3cube_numba_code():
    moves = "\n".join(['        ' + line for line in generate_moves_code()])
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

'''


cube_3_permutations = {
    name: indices_to_one_hot(indices)
    for name, indices in compute_stickers_permutation_per_operation(Cube3).items()
}

def generate_moves_code():
    lines = []
    for i, (name, indices) in enumerate(cube_3_permutations.items()):
        lines.append(f"elif move == {i + 1}:")
        for _from, _to in enumerate(indices):
            lines.append(f"    cubes[x][{_to}] = buffer[x][{_from}]")
    lines[0] = lines[0][2:]
    return lines
