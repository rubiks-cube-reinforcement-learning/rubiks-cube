from collections import defaultdict
from typing import List, TypeVar, Callable, Type, Dict, Tuple, Any

from utils import CubiesCube, flatten, compute_stickers_permutation, Cubie, cube_with_unique_sticker_codes, \
    apply_stickers_permutation, identity

T = TypeVar("T", bound=CubiesCube)
RotationMove = Callable[[T], T]


def compute_all_cube_rotations(initial_state: T) -> List[T]:
    def apply_all_rotation_moves(state):
        return [move(state) for move in type(initial_state).ROTATION_MOVES]

    lookup = [initial_state]
    for i in range(3):
        lookup += flatten([apply_all_rotation_moves(state) for state in lookup])

    all_rotations = {}
    for candidate in lookup:
        permutation = compute_stickers_permutation(candidate, initial_state)
        all_rotations["_".join(map(str, permutation))] = candidate
    assert len(all_rotations) == 24, "Must produce all 24 rotations (result: %d)" % len(all_rotations)
    return list(all_rotations.values())


def compute_all_orienting_permutations(cube_class: Type[T]) -> List[List[int]]:
    solved_cube = cube_with_unique_sticker_codes(cube_class)
    all_rotations = compute_all_cube_rotations(solved_cube)

    return [compute_stickers_permutation(solved_cube, rotated_cube) for rotated_cube in all_rotations]


def compute_all_orienting_permutations_by_cubie_and_stickers(cube_class: Type[T]) -> Dict[int, List[Dict[str, Any]]]:
    permutations = defaultdict(lambda: [])
    all_rotations = compute_all_cube_rotations(cube_class())
    all_orienting_permutations = compute_all_orienting_permutations(cube_class)
    for rotated_cube in all_rotations:
        for permutation in all_orienting_permutations:
            rotated = apply_stickers_permutation(rotated_cube, permutation)
            maybe_fixed_cubie = rotated.cubies[cube_class.FIXED_CUBIE_INDEX]
            if maybe_fixed_cubie.faces == cube_class.FIXED_CUBIE_COLOR_PATTERN:
                before_cubie = None
                before_cubie_idx = None
                for i, rotated_cubie in enumerate(rotated_cube.cubies):
                    fixed_faces = sorted(filter(identity, maybe_fixed_cubie.faces))
                    before_faces = sorted(filter(identity, rotated_cubie.faces))
                    if fixed_faces == before_faces:
                        before_cubie = rotated_cubie
                        before_cubie_idx = i
                        break
                cubie_idx = before_cubie_idx
                permutations[cubie_idx].append({
                    "cubie_idx": cubie_idx,
                    "color_pattern": before_cubie.faces,
                    "permutation": permutation,
                })
    return dict(permutations)


def find_stickers_permutation_to_orient_the_cube(cube: T) -> List[int]:
    all_orienting_permutations = compute_all_orienting_permutations(type(cube))
    for permutation in all_orienting_permutations:
        rotated = apply_stickers_permutation(cube, permutation)
        maybe_fixed_cubie = rotated.cubies[cube.FIXED_CUBIE_INDEX]
        if maybe_fixed_cubie.faces == cube.FIXED_CUBIE_COLOR_PATTERN:
            return permutation


def compute_possible_stickers_patterns(solved_pattern, is_even=True):
    seq = solved_pattern * 2
    if is_even:
        seq = seq[::-1]
    return [
        seq[0:3],
        seq[1:4],
        seq[2:5],
    ]


def compute_cubie_sticker_patterns_to_find_fixed_cubie_in_any_scramled_states(cube_class: Type[T]) -> Dict[
    int, List[List[int]]]:
    solved_cube = cube_class()
    fixed_cubie = solved_cube.cubies[cube_class.FIXED_CUBIE_INDEX]
    stickers_patterns = {
        "even": compute_possible_stickers_patterns(fixed_cubie.faces, True),
        "odd": compute_possible_stickers_patterns(fixed_cubie.faces, False),
    }

    odd_cubies = cube_class.CUBIES_WITH_ODD_ORIENTATION

    cubie_patterns = {}
    for cubie_nb, _ in solved_cube.corner_cubies:
        possible_sticker_patterns = stickers_patterns['odd'] if cubie_nb in odd_cubies else stickers_patterns['even']
        cubie_patterns[cubie_nb] = possible_sticker_patterns
    return cubie_patterns


def find_fixed_cubie_in_scrambled_state(scrambled_cube: T) -> Tuple[int, Cubie, List[int]]:
    sticker_patterns = compute_cubie_sticker_patterns_to_find_fixed_cubie_in_any_scramled_states(type(scrambled_cube))
    for i, cubie in scrambled_cube.corner_cubies:
        faces = cubie.faces
        if faces in sticker_patterns[i]:
            return i, cubie, faces
    return None, None, None


def orient_cube(cube: T) -> T:
    return apply_stickers_permutation(
        cube,
        find_stickers_permutation_to_orient_the_cube(cube)
    )
