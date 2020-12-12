from collections import defaultdict
from functools import reduce
from typing import Dict, TypeVar, Type, Any

from cube2.cube import StickerBinarySerializer
from orientation import compute_all_orienting_permutations_by_cubie_and_stickers
from utils import Cubie, cube_with_unique_sticker_codes, CubiesCube, AXIS_X, AXIS_Y, AXIS_Z


T = TypeVar("T", bound=CubiesCube)


def build_binary_orientation_spec(cube_class: Type[T]) -> Dict[int, Any]:
    binary_spec = {}
    all_orienting_permutations = compute_all_orienting_permutations_by_cubie_and_stickers(cube_class)
    for cubie_idx, cubie_spec in all_orienting_permutations.items():
        binary_spec[cubie_idx] = []
        for entry in cubie_spec:
            binary_spec[cubie_idx].append({
                "cubie": cubie_idx,
                "color_pattern": entry["color_pattern"],
                "color_detection_bitwise_lhs": color_detection_bitwise_ops(cube_class, cubie_idx),
                "color_detection_bitwise_rhs": (
                        (entry["color_pattern"][0] << StickerBinarySerializer.COLOR_BITS * 2) |
                        (entry["color_pattern"][1] << StickerBinarySerializer.COLOR_BITS * 1) |
                        entry["color_pattern"][2]
                ),
                "orient_cube_bitwise_op": permutation_to_bitwise_ops(entry["permutation"]),
            })
    return binary_spec


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

    state_shifts = sticker_wise_permutation_to_bitwise_ops(sticker_permutations)
    return state_shifts


def permutation_to_bitwise_ops(permutation_vector, *args):
    shifts_specification = dict(zip(permutation_vector, range(len(permutation_vector))))
    return sticker_wise_permutation_to_bitwise_ops(
        shifts_specification,
        *args
    )

# The code below is specific to 2-cube for now

def sticker_wise_permutation_to_bitwise_ops(shifts_specification: Dict[int, int],
                                            offset=StickerBinarySerializer.OFFSET,
                                            int_size=StickerBinarySerializer.INT_LENGTH,
                                            bucket_size=StickerBinarySerializer.COLOR_BITS):
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in shifts_specification.items():
        bitwise_mask = ['0'] * int_size
        color_offset = offset + old_idx * bucket_size
        for color_bit_offset in range(bucket_size):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (old_idx - new_idx) * bucket_size
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
