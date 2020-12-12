from collections import defaultdict
from functools import reduce
from typing import Dict, List

from cube2.cube import StickerBinarySerializer, Cube2, CUBE2_ROTATION_MOVES
from orientation import compute_all_cube_rotations
from utils import compute_stickers_permutation, Cubie, flatten


def compute_bitwise_shifts(permutation_vector, *args):
    shifts_specification = dict(zip(permutation_vector, range(len(permutation_vector))))
    return compute_bitwise_shifts_for_specific_stickers(
        shifts_specification,
        *args
    )


def compute_bitwise_shifts_for_specific_stickers(shifts_specification: Dict[int, int],
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


CUBE_WITH_UNIQUE_STICKER_CODES = Cube2([
    Cubie(100, 101, 102, 1),
    Cubie(103, 104, 105, 2),
    Cubie(106, 107, 108, 3),
    Cubie(109, 110, 111, 4),
    Cubie(112, 113, 114, 5),
    Cubie(115, 116, 117, 6),
    Cubie(118, 119, 120, 7),
    Cubie(121, 122, 123, 8),
])


def compute_any_cube_to_oriented_cube_bit_checks_and_shifts():
    stickers_patterns = generate_fixed_cubie_stickers_bit_patterns()
    empty_cubie = Cubie(0, 0, 0)

    odd = Cube2.CUBIES_WITH_ODD_ORIENTATION

    x_face_stickers = [1, 2, 3, 4, 9, 10, 11, 12]
    y_face_stickers = [5, 6, 7, 8, 13, 14, 15, 16]
    z_face_stickers = [17, 18, 19, 20, 21, 22, 23, 24]

    checks_and_shifts = {}
    for i in range(Cube2.NB_CUBIES):
        candidate_nb = i + 1
        mask_cube = Cube2(
            [empty_cubie] * i +
            [Cubie(7, 7, 7)] +  # 7 is binary 111
            [empty_cubie] * (Cube2.NB_CUBIES - i - 1)
        )

        stickers_order = x_face_stickers + y_face_stickers + z_face_stickers
        mask_vector = mask_cube.as_stickers_vector
        xyz_stickers_positions = [sticker_nb - 1 for sticker_nb in stickers_order if mask_vector[sticker_nb - 1] == 7]
        state_shifts = compute_bitwise_shifts_for_specific_stickers({
            _from: _to for _from, _to in zip(xyz_stickers_positions, range(21, 24))
        })

        possible_colors_patterns = stickers_patterns['odd'] if candidate_nb in odd else stickers_patterns['even']
        checks_and_shifts[candidate_nb] = {"bitwise_color_detection_shifts": state_shifts,
                                           "possible_colors_patterns": possible_colors_patterns,}
    return checks_and_shifts


def generate_fixed_cubie_stickers_bit_patterns():
    colors_sequences = {
        "even": [(6, 4, 1), (4, 1, 6), (1, 6, 4)],
        "odd": [(1, 4, 6), (4, 6, 1), (6, 1, 4)],
    }

    bit_patterns = {}
    for variant, stickers_pattern in colors_sequences.items():
        bit_patterns[variant] = []
        for x, y, z in stickers_pattern:
            bit_patterns[variant].append(
                (x << StickerBinarySerializer.COLOR_BITS * 2) | \
                (y << StickerBinarySerializer.COLOR_BITS * 1) | \
                z
            )
    return bit_patterns


def find_permutations_to_orient_the_cube():
    checks_and_shifts = compute_any_cube_to_oriented_cube_bit_checks_and_shifts()
    possible_rotations = compute_all_cube_rotations(Cube2())
    possible_rotations_unique = compute_all_cube_rotations(CUBE_WITH_UNIQUE_STICKER_CODES)
    results_by_cubie = defaultdict(lambda: [])
    for i, state in enumerate(possible_rotations):
        numeric_state = state.as_stickers_int
        for cubie, details in checks_and_shifts.items():
            actual_color_pattern = apply_bitwise_shifts(details['bitwise_color_detection_shifts'], numeric_state)
            for possible_color_pattern in details["possible_colors_patterns"]:
                if actual_color_pattern == possible_color_pattern:
                    orienting_permutation = compute_stickers_permutation(
                        CUBE_WITH_UNIQUE_STICKER_CODES,
                        possible_rotations_unique[i]
                    )
                    results_by_cubie[cubie].append({
                        "cubie": cubie,
                        "color_pattern": actual_color_pattern,
                        "bitwise_color_detection_shifts": details['bitwise_color_detection_shifts'],
                        "bitwise_orienting_permutation": compute_bitwise_shifts(orienting_permutation)
                    })
    return dict(results_by_cubie)