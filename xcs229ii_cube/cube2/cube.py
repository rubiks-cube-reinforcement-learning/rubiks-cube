# coding=utf-8
from xcs229ii_cube.utils import Cubie, AXIS_X, AXIS_Y, AXIS_Z, Operation, CubiesCube, \
    partition
from xcs229ii_cube.loggers import getLogger

logger = getLogger(__name__)

LU = Operation({1: 5, 5: 7, 7: 3, 3: 1}, AXIS_Z)
LD = LU.reverse()
RU = Operation({2: 6, 6: 8, 8: 4, 4: 2}, AXIS_Z)
RD = RU.reverse()

FL = Operation({1: 3, 3: 4, 4: 2, 2: 1}, AXIS_X)
FR = FL.reverse()
BL = Operation({5: 7, 7: 8, 8: 6, 6: 5}, AXIS_X)
BR = BL.reverse()

UL = Operation({1: 5, 5: 6, 6: 2, 2: 1}, AXIS_Y)
UR = UL.reverse()
DL = Operation({3: 7, 7: 8, 8: 4, 4: 3}, AXIS_Y)
DR = DL.reverse()

OPERATIONS = {'LU': LU, 'LD': LD,
              'RU': RU, 'RD': RD,
              'FL': FL, 'FR': FR,
              'BL': BL, 'BR': BR,
              'UL': UL, 'UR': UR,
              'DL': DL, 'DR': DR}

CUBE2_ROTATION_MOVES = [
    lambda state: LU(RU(state)),
    lambda state: LD(RD(state)),
    lambda state: FL(BL(state)),
    lambda state: FR(BR(state)),
    lambda state: UL(DL(state)),
    lambda state: UR(DR(state)),
]

CUBE2_OPPOSITE_ROTATION_MOVES = partition(CUBE2_ROTATION_MOVES, 2)

class Cube2(CubiesCube):
    NB_CUBIES = 8

    FIXED_CUBIE_INDEX = 4
    CUBIES_WITH_EVEN_ORIENTATION = [2, 3, 5, 8]
    CUBIES_WITH_ODD_ORIENTATION = [1, 4, 6, 7]

    FACE_CUBIES_INDICES = {
        1: [1, 2, 3, 4],
        2: [5, 6, 1, 2],
        3: [5, 6, 7, 8],
        4: [7, 8, 3, 4],
        5: [5, 1, 7, 3],
        6: [2, 6, 4, 8],
    }

    def __init__(self, cubies=None):
        super(Cube2, self).__init__(cubies or [
            Cubie(1, 2, 5, 1),
            Cubie(1, 2, 6, 2),
            Cubie(1, 4, 5, 3),
            Cubie(1, 4, 6, 4),
            Cubie(3, 2, 5, 5),
            Cubie(3, 2, 6, 6),
            Cubie(3, 4, 5, 7),
            Cubie(3, 4, 6, 8),
        ])


Cube2.OPERATIONS = OPERATIONS
Cube2.ROTATION_MOVES = CUBE2_ROTATION_MOVES
Cube2.OPPOSITE_ROTATION_MOVES = CUBE2_OPPOSITE_ROTATION_MOVES
