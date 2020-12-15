"""
Every cube tensor is assumed to consist of n one-hot segments of size 6 representing the stickers
"""
from itertools import chain
import torch
import math
import numpy as np
from torch import Tensor
from code_generator.common import build_cube3_to_cube2_shifts
from numba import cuda

from cube3.cube import Cube3
from cube3.generated_numba import apply_moves

def indices_to_one_hot(indices):
    return list(chain(*[range(i*6, i*6+6) for i in indices]))

MOVES_WIDTH = 100
THREADS_PER_BLOCK = 1024
REPRESENTATION_WIDTH = len(indices_to_one_hot(Cube3().as_stickers_vector))

def build_numba_methods():
    threadsperblock = (THREADS_PER_BLOCK,)
    blockspergrid_x = math.ceil(MOVES_WIDTH / threadsperblock[0])
    blockspergrid = (blockspergrid_x,)

    compiled_apply_moves = apply_moves[blockspergrid, threadsperblock]
    # Initialize:
    d_buffer_small = cuda.to_device(np.random.randint(2, size=(10, REPRESENTATION_WIDTH), dtype=np.uint8))
    d_cubes_small = cuda.to_device(np.random.randint(2, size=(10, REPRESENTATION_WIDTH), dtype=np.uint8))
    d_moves_small = cuda.to_device(np.ones((2, MOVES_WIDTH), dtype=np.uint8))
    compiled_apply_moves(d_cubes_small, d_buffer_small, d_moves_small)

    return {
        "apply_moves": compiled_apply_moves
    }

methods = build_numba_methods()
NUMBA_APPLY_MOVES = methods['apply_moves']

def generate_oriented_3_cube_batch(n: int) -> Tensor:
    # @TODO
    # cubes = torch.Tensor()
    # moves = torch.tensor()
    apply_moves_to_3_cubes_in_place(cubes, moves)
    return cubes

cube_3_one_hot_indices = indices_to_one_hot(build_cube3_to_cube2_shifts().keys())
def convert_3_cubes_to_2_cubes(cubes: Tensor) -> Tensor:
    return cubes[:, cube_3_one_hot_indices]

def apply_moves_to_3_cubes_in_place(batch_of_cubes: Tensor, moves_per_cube: Tensor):
    d_cubes = cuda.from_cuda_array_interface(batch_of_cubes.__cuda_array_interface__)
    d_moves = cuda.from_cuda_array_interface(moves_per_cube.__cuda_array_interface__)
    d_buffer = cuda.to_device(np.empty(batch_of_cubes.shape, dtype=np.uint8))
    NUMBA_APPLY_MOVES(d_cubes, d_buffer, d_moves)

def orient_3_cubes_in_space(cubes: Tensor) -> Tensor:
    pass

if __name__ == "__main__":
    _3_cubes = torch.arange(0, 324).repeat(10, 1)
    _2_cubes = convert_3_cubes_to_2_cubes(_3_cubes)
