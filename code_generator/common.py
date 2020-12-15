from typing import TypeVar, Type, List, Dict, Any

from cube3.cube import Cube3
from orientation import compute_all_orienting_permutations_by_cubie_and_stickers
from utils import CubiesCube, cube_with_unique_sticker_codes, compute_stickers_permutation, Cubie, AXIS_X, AXIS_Y, AXIS_Z

T = TypeVar("T", bound=CubiesCube)


def build_cube3_to_cube2_shifts():
    unique_cube3 = cube_with_unique_sticker_codes(Cube3)
    unique_cube2 = unique_cube3.as_cube2

    old_indices = compute_stickers_permutation(unique_cube2, unique_cube3)
    new_indices = [i for i in range(len(old_indices))]
    return dict(list(zip(old_indices, new_indices)))


def compute_sticker_indices_for_cubie(cube_class: Type[T], cubie_idx) -> List[int]:
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
        [Cubie(7, 7, 7)] +
        [empty_cubie] * (cube_class.NB_CUBIES - cubie_idx)
    )

    mask_vector = mask_cube.as_stickers_vector
    return [sticker_nb for sticker_nb in lookup_stickers if mask_vector[sticker_nb] == 7]


def compute_stickers_permutation_per_operation(cube_class: Type[T]):
    unique_cube = cube_with_unique_sticker_codes(cube_class)

    return {
        name: compute_stickers_permutation(op(unique_cube), unique_cube)
        for name, op in cube_class.OPERATIONS.items()
    }


def build_orientation_spec(cube_class: Type[T]) -> Dict[int, Any]:
    permutations_spec = {}
    all_orienting_permutations = compute_all_orienting_permutations_by_cubie_and_stickers(cube_class)
    for cubie_idx, cubie_spec in all_orienting_permutations.items():
        permutations_spec[cubie_idx] = []
        for entry in cubie_spec:
            permutations_spec[cubie_idx].append({
                "cubie": cubie_idx,
                "color_pattern": entry["color_pattern"],
                "stickers_indices": compute_sticker_indices_for_cubie(cube_class, cubie_idx),
                "permutation": entry["permutation"]
            })
    return permutations_spec
