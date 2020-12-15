from __future__ import annotations

import itertools
import random
from functools import reduce
from typing import Type, TypeVar, Generic, List, Dict, Tuple


def flatten(_list):
    return list(itertools.chain(*_list))


def partition(_list, bucket_size):
    return [_list[i:i + bucket_size] for i in range(0, len(_list), bucket_size)]


def normalize_binary_string(binary_string, expected_length):
    return (('0' * (expected_length - len(binary_string))) + binary_string)[-expected_length:]


def lists_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def sequences(_list):
    prev = _list[0]
    seq = []
    last_i = len(_list) - 2
    for i, item in enumerate(_list[1:]):
        seq.append(prev)
        if i == last_i:
            seq.append(item)
            yield seq
        elif item != prev + 1:
            yield seq
            seq = []
        prev = item


AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

class CubiesCube:
    FACE_TO_AXIS = {
        1: AXIS_X,
        3: AXIS_X,
        2: AXIS_Y,
        4: AXIS_Y,
        5: AXIS_Z,
        6: AXIS_Z,
    }
    AXIS_TO_FACES = {
        AXIS_X: [1, 3],
        AXIS_Y: [2, 4],
        AXIS_Z: [5, 6],
    }

    FIXED_CUBIE_INDEX = None
    FIXED_CUBIE_COLOR_PATTERN = [1, 4, 6]
    NB_CUBIES = None
    FACE_CUBIES_INDICES = None
    OPERATIONS = {}
    ROTATION_MOVES = {}
    OPPOSITE_ROTATION_MOVES = {}

    @property
    def FIXED_CUBIE_OPERATIONS(self):
        return {
            name: op for name, op in self.OPERATIONS.items() if self.FIXED_CUBIE_INDEX not in op.mapping
        }

    def __init__(self, cubies):
        self.cubies = [Cubie(None, None, None, None)] + cubies  # 1-based indexing

    def __str__(self):
        faces = [self.face(i) for i in range(1, 7)]
        return faces.__str__()

    def clone(self):
        return type(self)([c.clone() for c in self.cubies[1:]])

    def face(self, nb):
        axis = self.FACE_TO_AXIS[nb]
        colors = [self.cubies[idx].get_face(axis) for idx in self.FACE_CUBIES_INDICES[nb]]
        return colors

    @property
    def corner_cubies(self) -> List[Tuple[int, Cubie]]:
        return [(i, cubie) for i, cubie in enumerate(self.cubies) if cubie.is_corner_cubie]

    @property
    def as_stickers_vector(self):
        return StickerVectorSerializer(type(self)).serialize(self)

    @property
    def as_cubies_indices(self):
        return [cubie.idx for cubie in self.cubies[1:]]

    @property
    def as_stickers_int(self):
        from code_generator.bitwise_repr import IntSerializer
        from code_generator.bitwise_repr import StickerBinarySerializer
        return IntSerializer(StickerBinarySerializer(type(self))).serialize(self)

    @property
    def as_stickers_binary_string(self):
        from code_generator.bitwise_repr import StickerBinarySerializer
        return StickerBinarySerializer(type(self)).serialize(self)


class Cubie:
    def __init__(self, face_x, face_y, face_z, idx=None):
        self.idx = idx
        self.face_x = face_x
        self.face_y = face_y
        self.face_z = face_z

    def __repr__(self):
        return "Cubie (%s, %s, %s, %s)" % (self.face_x, self.face_y, self.face_z, self.idx)

    def __str__(self):
        return (self.face_x, self.face_y, self.face_z, self.idx,)

    @property
    def is_corner_cubie(self):
        return self.face_x is not None and self.face_y is not None and self.face_z is not None

    @property
    def faces(self):
        return [self.face_x, self.face_y, self.face_z]

    def get_face(self, axis):
        if axis == AXIS_X:
            return self.face_x
        elif axis == AXIS_Y:
            return self.face_y
        elif axis == AXIS_Z:
            return self.face_z

    def set_face(self, axis, color):
        new_cubie = self.clone()
        if axis == AXIS_X:
            new_cubie.face_x = color
        elif axis == AXIS_Y:
            new_cubie.face_y = color
        elif axis == AXIS_Z:
            new_cubie.face_z = color
        return new_cubie

    def rotate_along(self, axis):
        new_cubie = self.clone()
        if axis == AXIS_X:
            new_cubie.face_y, new_cubie.face_z = new_cubie.face_z, new_cubie.face_y
        elif axis == AXIS_Y:
            new_cubie.face_x, new_cubie.face_z = new_cubie.face_z, new_cubie.face_x
        elif axis == AXIS_Z:
            new_cubie.face_x, new_cubie.face_y = new_cubie.face_y, new_cubie.face_x
        return new_cubie

    def clone(self):
        return Cubie(self.face_x, self.face_y, self.face_z, self.idx)


T = TypeVar("T", bound=CubiesCube)


class Operation(Generic[T]):
    def __init__(self, mapping: Dict[int, int], axis: int):
        self.mapping = mapping
        self.axis = axis
        assert axis in [AXIS_X, AXIS_Y, AXIS_Z]

    def reverse(self):
        return Operation({v: k for k, v in self.mapping.items()}, self.axis)

    def __call__(self, cube: T) -> T:
        new_cube = cube.clone()
        for k, v in self.mapping.items():
            new_cube.cubies[v] = cube.cubies[k].rotate_along(self.axis)
        return new_cube


class CubeSerializer(Generic[T]):

    def __init__(self, cube_class: Type[T]):
        self.cube_class = cube_class

    def serialize(self, cube: T):
        raise NotImplementedError()

    def unserialize(self, value) -> T:
        raise NotImplementedError()


class StickerVectorSerializer(CubeSerializer):
    def serialize(self, cube: T):
        vector = []
        for i in range(1, 7):
            vector += cube.face(i)
        return vector

    def unserialize(self, vector) -> T:
        cube = self.cube_class()
        n = len(cube.FACE_CUBIES_INDICES[1])
        faces = [vector[i:i + n] for i in range(0, len(vector), n)]
        for i, colors in enumerate(faces):
            face_nb = i + 1
            axis = self.cube_class.FACE_TO_AXIS[face_nb]
            cubies = self.cube_class.FACE_CUBIES_INDICES[face_nb]
            for i, color in enumerate(colors):
                cube.cubies[cubies[i]] = cube.cubies[cubies[i]].set_face(axis, color)
        return cube


def identity(x):
    return x


def compute_stickers_permutation(target_cube: CubiesCube, source_cube: CubiesCube) -> List[int]:
    source_vector = list(filter(identity, source_cube.as_stickers_vector))
    target_vector = list(filter(identity, target_cube.as_stickers_vector))
    return [source_vector.index(i) for i in target_vector]


def compute_permutation_dict(before: List[int], after: List[int]) -> Dict[int, int]:
    return dict(list(zip(before, after)))


def cube_with_unique_sticker_codes(cube_class: Type[T]) -> T:
    cube = cube_class()
    def code_gen():
        i = 0
        while True:
            i += 1
            yield i
    code = code_gen()

    for cubie in cube.cubies[1:]:
        if cubie.face_x is not None:
            cubie.face_x = next(code)
        if cubie.face_y is not None:
            cubie.face_y = next(code)
        if cubie.face_z is not None:
            cubie.face_z = next(code)
    return cube


def apply_stickers_permutation(cube: T, permutation: List[int]) -> T:
    before = cube.as_stickers_vector
    after = [before[i] for i in permutation]
    return StickerVectorSerializer(type(cube)).unserialize(after)


def generate_dataset(nb_per_scramble, max_scrambles, scramble_fn):
    dataset = []
    for i in range(nb_per_scramble):
        for scrambles in range(max_scrambles):
            dataset.append(scramble_fn(scrambles))
    return dataset


def scramble(solved_state, scrambles, moves=None):
    state = solved_state
    op = last_op = None
    for i in range(scrambles):
        while op is last_op:
            op = random.choice(moves)
        state = op(state)
        last_op = op
    return state
