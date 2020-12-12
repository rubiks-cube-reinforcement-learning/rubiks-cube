import math
import random
from collections import defaultdict
from functools import reduce
from typing import Dict, TypeVar, Type, Any, Generic

from cube2.cube import Cube2
from cube3.cube import Cube3
from orientation import compute_all_orienting_permutations_by_cubie_and_stickers
from utils import Cubie, cube_with_unique_sticker_codes, CubiesCube, AXIS_X, AXIS_Y, AXIS_Z, CubeSerializer, \
    StickerVectorSerializer, normalize_binary_string, AXES, compute_stickers_permutation

T = TypeVar("T", bound=CubiesCube)


class BinaryRepresentation(Generic[T]):

    def __init__(self, cube_class: Type[T]):
        self.cube_class = cube_class


class IntSpec:
    def __init__(self, bits_per_color: int, data_bits: int) -> None:
        self.bits_per_color = bits_per_color
        self.data_bits = data_bits
        self.int_size = 2 ** math.ceil(math.log(self.data_bits, 2))
        self.offset = self.int_size - self.data_bits

    @staticmethod
    def for_cube(cube_class: Type[T]):
        nb_colors = len(cube_class().as_stickers_vector)
        bits_per_color = 3
        data_bits = nb_colors * bits_per_color
        return IntSpec(bits_per_color, data_bits)


def build_cube3_to_cube2_bitwise_ops():
    unique_cube3 = cube_with_unique_sticker_codes(Cube3)
    unique_cube2 = unique_cube3.as_cube2
    size_diff = len(unique_cube3.as_stickers_vector) - len(unique_cube2.as_stickers_vector)

    old_indices = compute_stickers_permutation(unique_cube2, unique_cube3)
    new_indices = [i + size_diff for i in range(len(old_indices))]
    shifts_spec = dict(list(zip(old_indices, new_indices)))
    return sticker_wise_permutation_to_bitwise_ops(shifts_spec, IntSpec.for_cube(Cube3))


def build_binary_orientation_spec(cube_class: Type[T]) -> Dict[int, Any]:
    int_spec = IntSpec.for_cube(cube_class)
    bitwise_spec = {}
    all_orienting_permutations = compute_all_orienting_permutations_by_cubie_and_stickers(cube_class)
    for cubie_idx, cubie_spec in all_orienting_permutations.items():
        bitwise_spec[cubie_idx] = []
        for entry in cubie_spec:
            bitwise_spec[cubie_idx].append({
                "cubie": cubie_idx,
                "color_pattern": entry["color_pattern"],
                "color_detection_bitwise_lhs": color_detection_bitwise_ops(cube_class, cubie_idx),
                "color_detection_bitwise_rhs": (
                        (entry["color_pattern"][0] << int_spec.bits_per_color * 2) |
                        (entry["color_pattern"][1] << int_spec.bits_per_color * 1) |
                        entry["color_pattern"][2]
                ),
                "orient_cube_bitwise_op": permutation_to_bitwise_ops(entry["permutation"], int_spec)
            })
    return bitwise_spec


def color_detection_bitwise_ops(cube_class: Type[T], cubie_idx) -> Dict[int, int]:
    empty_cubie = Cubie(0, 0, 0)

    unique_cube = cube_with_unique_sticker_codes(cube_class)
    unique_cubies = unique_cube.cubies
    unique_stickers = unique_cube.as_stickers_vector
    lookup_order = [AXIS_X, AXIS_Y, AXIS_Z]
    lookup_stickers = []
    for axis in lookup_order:
        for face_nb in cube_class.AXIS_TO_FACES[axis]:
            for face_cubie_idx in cube_class.FACE_CUBIES_INDICES[face_nb]:
                sticker_value = unique_cubies[face_cubie_idx].get_face(axis)
                lookup_stickers.append(unique_stickers.index(sticker_value))

    mask_cube = cube_class(
        [empty_cubie] * (cubie_idx - 1) +
        [Cubie(7, 7, 7)] +  # 7 is binary 111
        [empty_cubie] * (cube_class.NB_CUBIES - cubie_idx)
    )

    mask_vector = mask_cube.as_stickers_vector
    xyz_stickers_positions = [sticker_nb for sticker_nb in lookup_stickers if mask_vector[sticker_nb] == 7]
    sticker_permutations = {
        _from: _to for _from, _to in zip(xyz_stickers_positions, range(len(unique_stickers) - 3, len(unique_stickers)))
    }

    state_shifts = sticker_wise_permutation_to_bitwise_ops(sticker_permutations, IntSpec.for_cube(cube_class))
    return state_shifts


def permutation_to_bitwise_ops(permutation_vector, int_spec: IntSpec):
    shifts_specification = dict(zip(permutation_vector, range(len(permutation_vector))))
    return sticker_wise_permutation_to_bitwise_ops(
        shifts_specification,
        int_spec
    )


def sticker_wise_permutation_to_bitwise_ops(shifts_specification: Dict[int, int],
                                            int_spec: IntSpec):
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in shifts_specification.items():
        bitwise_mask = ['0'] * int_spec.int_size
        color_offset = int_spec.offset + old_idx * int_spec.bits_per_color
        for color_bit_offset in range(int_spec.bits_per_color):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (old_idx - new_idx) * int_spec.bits_per_color
        hex_mask = int(bitwise_mask_str, 2)
        reverse_shifts[bitwise_offset] |= hex_mask
    return dict(reverse_shifts)


def apply_bitwise_shifts(shifts, number):
    return reduce(
        lambda acc, item: acc | apply_bitwise_shift(item[0], item[1], number),
        shifts.items(),
        0
    )


def apply_bitwise_shift(offset, mask, x):
    masked = x & mask
    abs_offset = abs(offset)
    return masked >> abs_offset if offset < 0 else masked << abs_offset


class StickerBinarySerializer(CubeSerializer[T]):

    def __init__(self, cube_class: Type[T]):
        self.int_spec = IntSpec.for_cube(cube_class)
        super().__init__(cube_class)

    def serialize(self, cube: T):
        vector = StickerVectorSerializer(self.cube_class).serialize(cube)
        return "".join(["0"] * self.int_spec.offset + ["{0:03b}".format(n) for n in vector])

    def unserialize(self, binary_string: str) -> T:
        binary_string = normalize_binary_string(binary_string, self.int_spec.data_bits)
        n = 3
        vector = [binary_string[i:i + n] for i in range(0, len(binary_string), n)]
        vector = [int(part, 2) for part in vector]
        return StickerVectorSerializer(self.cube_class).unserialize(vector)


class IntSerializer(CubeSerializer[T]):

    def __init__(self, binary_serializer: CubeSerializer) -> None:
        self.binary_serializer = binary_serializer
        super().__init__(self.binary_serializer.cube_class)

    def serialize(self, cube: T):
        binary = self.binary_serializer.serialize(cube)
        return int(binary, 2)

    def unserialize(self, number: int) -> T:
        return self.binary_serializer.unserialize("{0:03b}".format(number))

