"""
Every cube tensor is assumed to consist of n one-hot segments of size 6 representing the stickers
"""
import random
from itertools import chain
import torch
import math
import numpy as np
from torch import Tensor
from code_generator.common import build_cube3_to_cube2_shifts
from numba import cuda

from cube3.cube import Cube3
from cube3.generated_numba import apply_moves_fast, FIXED_CUBIE_MOVES_INDICES, MOVES_CPU_TENSOR

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def one_hot(color):
    zeros = [0] * 6
    zeros[color - 1] = 1
    return zeros


def colors_to_one_hot(colors):
    return list(chain(*[one_hot(i) for i in colors]))


MOVES_WIDTH = 20
THREADS_PER_BLOCK = 1024
STICKERS_NB = len(Cube3().as_stickers_vector)
REPRESENTATION_WIDTH = len(colors_to_one_hot(Cube3().as_stickers_vector))


class Glue2To3Cube:

    def __init__(self, device):
        self.device = device
        self.numba_apply_moves = self._compile_numba_apply_moves()
        self.d_recipes = cuda.from_cuda_array_interface(MOVES_CPU_TENSOR.to(device).__cuda_array_interface__)

    def _compile_numba_apply_moves(self):
        """
        This will take a few minutes to complete the first time it's run and that's okay.
        :return:
        """
        threadsperblock = (THREADS_PER_BLOCK,)
        blockspergrid_x = math.ceil(MOVES_WIDTH / threadsperblock[0])
        blockspergrid = (blockspergrid_x,)

        return apply_moves_fast[blockspergrid, threadsperblock]

    def generate_oriented_3_cube_batch(self, batch_size: int, scrambles: int) -> (Tensor, Tensor):
        available_moves = list(FIXED_CUBIE_MOVES_INDICES.values())
        moves_vector = np.array([
            random.choices(available_moves, k=scrambles) + [0] * (MOVES_WIDTH - scrambles) for i in range(batch_size)
        ])
        moves = torch.tensor(moves_vector, dtype=torch.uint8).to(self.device)

        solved_cube = torch.tensor(colors_to_one_hot(Cube3().as_stickers_vector), dtype=torch.uint8).to(self.device)
        solved_cubes = solved_cube.repeat(batch_size, 1)
        self.apply_moves_to_3_cubes_in_place(solved_cubes, moves)
        return solved_cubes, moves

    def convert_3_cubes_to_2_cubes(self, cubes: Tensor) -> Tensor:
        cube_3_one_hot_indices = colors_to_one_hot(build_cube3_to_cube2_shifts().keys())
        return cubes[:, cube_3_one_hot_indices]

    def apply_moves_to_3_cubes_in_place(self, batch_of_cubes: Tensor, moves_per_cube: Tensor):
        d_cubes = cuda.from_cuda_array_interface(batch_of_cubes.__cuda_array_interface__)
        d_moves = cuda.from_cuda_array_interface(moves_per_cube.__cuda_array_interface__)
        d_buffer = cuda.to_device(np.empty(batch_of_cubes.shape, dtype=np.uint8))
        self.numba_apply_moves(d_cubes, d_buffer, d_moves, self.d_recipes)


# Below this line be dragons

def _stickers_one_hot_matrix_to_colors_vector(cubes_host):
    STICKERS_NB = len(Cube3().as_stickers_vector)
    itemindex = np.argwhere(cubes_host == 1)
    itemindex[:, 1] = itemindex[:, 1] % 6 + 1
    cube_vectors = itemindex[:, 1].reshape(cubes_host.shape[0], STICKERS_NB)
    return cube_vectors


if __name__ == "__main__":
    device = torch.device("cuda")
    glue = Glue2To3Cube(device)
    x = glue.generate_oriented_3_cube_batch(1000000, 19)
