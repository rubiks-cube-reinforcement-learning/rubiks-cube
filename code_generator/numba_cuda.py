import json
from itertools import chain
from code_generator.common import compute_stickers_permutation_per_operation
from cube3.cube import Cube3


def indices_to_one_hot(indices):
    return list(chain(*[range(i*6, i*6+6) for i in indices]))


def generate_3cube_numba_code():
    moves = "\n".join(['        %s,' % indices for (name, indices) in cube_3_permutations.values()])
    moves_indices = ", ".join([f"{spec[0]}: '{name}'" for name, spec in cube_3_permutations.items()])
    moves_names = ", ".join([f"'{name}': {spec[0]}" for name, spec in cube_3_permutations.items()])
    fixed_cubie_moves = {k:v for k,v in cube_3_permutations.items() if k in list(Cube3().FIXED_CUBIE_OPERATIONS.keys())}
    fixed_cubie_moves_indices = ", ".join([f"{spec[0]}: '{name}'" for name, spec in fixed_cubie_moves.items()])
    fixed_cubie_moves_names = ", ".join([f"'{name}': {spec[0]}" for name, spec in fixed_cubie_moves.items()])
    return f'''
from numba import cuda
import torch 

@cuda.jit
def apply_moves_fast(cubes, buffer, moves, recipes):
    x = cuda.grid(1)

    if x >= cubes.shape[0] or x >= moves.shape[0]:
        return
    for move in moves[x]:
        if move == 0:
            break
        for i in range(324):
            buffer[x][i] = cubes[x][i]
        for i in range(324):
            cubes[x][i] = buffer[x][recipes[move - 1][i]]

MOVES_CPU_TENSOR = torch.tensor([
{moves}
], dtype=torch.int16)

MOVES_NAMES = {{{moves_indices}}}
MOVES_INDICES = {{{moves_names}}}
FIXED_CUBIE_MOVES_NAMES = {{{fixed_cubie_moves_indices}}}
FIXED_CUBIE_MOVES_INDICES = {{{fixed_cubie_moves_names}}}
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
        for _to, _from in enumerate(indices):
            lines.append(f"    cubes[x][{_to}] = buffer[x][{_from}]")
    lines[0] = lines[0][2:]
    return lines

