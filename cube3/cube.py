# coding=utf-8
from typing import Dict, List

from cube2.cube import Cube2
from utils import Cubie, AXIS_X, AXIS_Y, AXIS_Z, Operation, CubiesCube, partition, flatten, compute_permutation_dict
from loggers import getLogger
import numpy as np

logger = getLogger(__name__)


class Cube3(CubiesCube):
    NB_CUBIES = 26

    FIXED_CUBIE_INDEX = 9
    CUBIES_WITH_EVEN_ORIENTATION = [3, 7, 18, 26]
    CUBIES_WITH_ODD_ORIENTATION = [1, 9, 20, 24]

    FACE_CUBIES_INDICES = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [18, 19, 20, 10, 11, 12, 1, 2, 3],
        3: [18, 19, 20, 21, 22, 23, 24, 25, 26],
        4: [24, 25, 26, 15, 16, 17, 7, 8, 9],
        5: [18, 10, 1, 21, 13, 4, 24, 15, 7],
        6: [3, 12, 20, 6, 14, 23, 9, 17, 26],
    }

    def __init__(self, cubies=None):
        super(Cube3, self).__init__(cubies or [
            Cubie(1, 2, 5, 1),
            Cubie(1, 2, None, 2),
            Cubie(1, 2, 6, 3),
            Cubie(1, None, 5, 4),
            Cubie(1, None, None, 5),
            Cubie(1, None, 6, 6),
            Cubie(1, 4, 5, 7),
            Cubie(1, 4, None, 8),
            Cubie(1, 4, 6, 9),

            Cubie(None, 2, 5, 10),
            Cubie(None, 2, None, 11),
            Cubie(None, 2, 6, 12),

            Cubie(None, None, 5, 13),
            Cubie(None, None, 6, 14),

            Cubie(None, 4, 5, 15),
            Cubie(None, 4, None, 16),
            Cubie(None, 4, 6, 17),

            Cubie(3, 2, 5, 18),
            Cubie(3, 2, None, 19),
            Cubie(3, 2, 6, 20),
            Cubie(3, None, 5, 21),
            Cubie(3, None, None, 22),
            Cubie(3, None, 6, 23),
            Cubie(3, 4, 5, 24),
            Cubie(3, 4, None, 25),
            Cubie(3, 4, 6, 26),
        ])

    @property
    def as_cube2(self):
        return Cube2([cubie for cubie in self.cubies if cubie.is_corner_cubie])


DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"


def face_rotation_spec(face: int, direction: str = DIRECTION_LEFT):
    return cubies_rotation_spec(Cube3.FACE_CUBIES_INDICES[face], direction)


def cubies_rotation_spec(cubies: List[int], direction: str = DIRECTION_LEFT) -> Dict[int, int]:
    matrix = np.array(partition(cubies, 3))
    if direction == DIRECTION_RIGHT:
        rotated_matrix = np.flip(matrix.T, 0)
    else:
        rotated_matrix = np.flip(matrix.T, 1)
    new_cubies = flatten(rotated_matrix.tolist())
    result = compute_permutation_dict(cubies, new_cubies)
    result.pop(None, None)  # Gracefully support None placeholders
    return result


LU = Operation(face_rotation_spec(5, DIRECTION_LEFT), AXIS_Z)
LD = LU.reverse()

RU = Operation(face_rotation_spec(6, DIRECTION_RIGHT), AXIS_Z)
RD = RU.reverse()

MU = Operation(cubies_rotation_spec([2, 11, 19, 5, None, 22, 8, 16, 25], DIRECTION_RIGHT), AXIS_Z)
MD = MU.reverse()

FL = Operation(face_rotation_spec(1, DIRECTION_LEFT), AXIS_X)
FR = FL.reverse()

ML_X = Operation(cubies_rotation_spec([10, 11, 12, 13, None, 14, 15, 16, 17], DIRECTION_LEFT), AXIS_X)
MR_X = ML_X.reverse()

BL = Operation(face_rotation_spec(3, DIRECTION_LEFT), AXIS_X)
BR = BL.reverse()

UL = Operation(face_rotation_spec(2, DIRECTION_RIGHT), AXIS_Y)
UR = UL.reverse()

ML_Y = Operation(cubies_rotation_spec([21, 22, 23, 13, None, 14, 4, 5, 6], DIRECTION_RIGHT), AXIS_Y)
MR_Y = ML_Y.reverse()

DL = Operation(face_rotation_spec(4, DIRECTION_RIGHT), AXIS_Y)
DR = DL.reverse()

OPERATIONS = {'LU': LU, 'LD': LD,
              'RU': RU, 'RD': RD,
              'FL': FL, 'FR': FR,
              'BL': BL, 'BR': BR,
              'UL': UL, 'UR': UR,
              'DL': DL, 'DR': DR,
              'MU': MU, 'MD': MD,
              'ML_X': ML_X, 'MR_X': MR_X,
              'ML_Y': ML_Y, 'MR_Y': MR_Y,
              }

CUBE3_ROTATION_MOVES = [
    lambda state: LU(MU(RU(state))),
    lambda state: LD(MD(RD(state))),
    lambda state: UL(ML_Y(DL(state))),
    lambda state: UR(MR_Y(DR(state))),
    lambda state: FL(ML_X(BL(state))),
    lambda state: FR(MR_X(BR(state))),
]

CUBE3_OPPOSITE_ROTATION_MOVES = partition(CUBE3_ROTATION_MOVES, 2)

Cube3.OPERATIONS = OPERATIONS
Cube3.ROTATION_MOVES = CUBE3_ROTATION_MOVES
Cube3.OPPOSITE_ROTATION_MOVES = CUBE3_OPPOSITE_ROTATION_MOVES
