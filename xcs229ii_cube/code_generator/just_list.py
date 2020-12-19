import json
from itertools import chain
from xcs229ii_cube.code_generator.common import compute_stickers_permutation_per_operation
from xcs229ii_cube.cube3.cube import Cube3


def generate_3cube_lists_code():
    moves = "\n".join(['        %s,' % indices for (name, indices) in cube_3_permutations.values()])
    moves_indices = ", ".join([f"{spec[0]}: '{name}'" for name, spec in cube_3_permutations.items()])
    moves_names = ", ".join([f"'{name}': {spec[0]}" for name, spec in cube_3_permutations.items()])
    fixed_cubie_moves = {k:v for k,v in cube_3_permutations.items() if k in list(Cube3().FIXED_CUBIE_OPERATIONS.keys())}
    fixed_cubie_moves_indices = ", ".join([f"{spec[0]}: '{name}'" for name, spec in fixed_cubie_moves.items()])
    fixed_cubie_moves_names = ", ".join([f"'{name}': {spec[0]}" for name, spec in fixed_cubie_moves.items()])

    moves_names_list = list(cube_3_permutations.keys())
    reverse_moves_names_list = [
        moves_names_list[i + 1] if i % 2 == 0 else moves_names_list[i - 1]
        for i in range(len(moves_names_list))
    ]
    return f'''
from numba import cuda
import torch 

def apply_move_lists(cubes, move):
    results = []
    for cube in cubes:
        results.append([
            cube[MOVES_DEFINITIONS[i]] for i in range(len(cube)) 
        ])
    return results

def apply_move_np(cubes, move):
    return cubes[:, move]

MOVES_DEFINITIONS = [
{moves}
]

MOVES_INDICES_TO_NAMES = {{{moves_indices}}}
MOVES_NAMES_TO_INDICES = {{{moves_names}}}
FIXED_CUBIE_MOVES_INDICES_TO_NAMES = {{{fixed_cubie_moves_indices}}}
FIXED_CUBIE_MOVES_NAMES_TO_INDICES = {{{fixed_cubie_moves_names}}}

MOVES_NAMES = {json.dumps(moves_names_list)}
REVERSE_MOVES_NAMES = {json.dumps(reverse_moves_names_list)}

if __name__ == "__main__":
    import numpy as np
    solved_states = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2,
         3, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 6]
    ])
    print(apply_move_np(solved_states, MOVES_DEFINITIONS[0]))
'''


cube_3_permutations = {
    name: (i, indices)
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

