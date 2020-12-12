import unittest

from cube2.cube import Cube2, CUBE2_OPPOSITE_ROTATION_MOVES, CUBE2_ROTATION_MOVES, OPERATIONS as CUBE2_OPERATIONS
from cube3.cube import Cube3, CUBE3_ROTATION_MOVES, CUBE3_OPPOSITE_ROTATION_MOVES, OPERATIONS as CUBE3_OPERATIONS
from orientation import compute_all_cube_rotations, \
    compute_cubie_sticker_patterns_to_find_fixed_cubie_in_any_scramled_states, find_fixed_cubie_in_scrambled_state, \
    find_stickers_permutations_to_orient_the_cube
from utils import cube_with_unique_sticker_codes


class TestCube3(unittest.TestCase):

    def test_space_rotations_basic(self):
        for rotation_move in CUBE3_ROTATION_MOVES:
            state = Cube3()
            for i in range(4):
                state = rotation_move(state)
            self.assertEqual(
                state.as_stickers_vector,
                Cube3().as_stickers_vector,
                "Cube rotated 4 times is different from unrotated cube"
            )

    def test_space_rotations_3_left_1_right(self):
        for r1, r2 in CUBE3_OPPOSITE_ROTATION_MOVES:
            state1 = Cube3()
            for i in range(3):
                state1 = r1(state1)
            state2 = r2(Cube3())
            self.assertEqual(
                state1.as_stickers_vector,
                state2.as_stickers_vector,
                "Cube rotated 3 times is one direction and 1 time in the opposite direction should be the same"
            )

    def test_as_stickers_vector(self):
        self.assertEqual(
            Cube3().as_stickers_vector,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "Solved cube3.as_stickers_vector should yield a solved cube2"
        )

    def test_to_cube2(self):
        self.assertEqual(
            Cube3().as_cube2.as_stickers_vector,
            Cube2().as_stickers_vector,
            "Solved cube3.as_stickers_vector should yield a solved cube2"
        )

    def test_build_cube_with_unique_sticker_codes(self):
        cube = cube_with_unique_sticker_codes(Cube3)
        self.assertEqual(sorted(cube.as_stickers_vector), list(range(1, 55)))


    def test_compute_all_cube_rotations_works(self):
        all_rotations = compute_all_cube_rotations(Cube3())
        self.assertEqual(
            24,
            len(all_rotations)
        )

    def test_sticker_patterns(self):
        sticker_patterns3 = compute_cubie_sticker_patterns_to_find_fixed_cubie_in_any_scramled_states(Cube3)
        self.assertEqual(
            sticker_patterns3,
            {1: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 3: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 7: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 9: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 18: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 20: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 24: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 26: [[6, 4, 1], [4, 1, 6], [1, 6, 4]]}
        )

    def test_find_fixed_cubie_in_scrambled_state(self):
        position, cubie, stickers = find_fixed_cubie_in_scrambled_state(Cube3())
        self.assertEqual(position, 9)
        self.assertEqual(stickers, [1, 4, 6])

        position, cubie, stickers = find_fixed_cubie_in_scrambled_state(CUBE3_OPERATIONS['RU'](Cube3()))
        self.assertEqual(position, 3)
        self.assertEqual(stickers, [4, 1, 6])

    def test_find_stickers_permutations_to_orient_the_cube(self):
        solved = Cube3()
        permutation = find_stickers_permutations_to_orient_the_cube(solved)
        self.assertEqual(permutation, list(range(54)))

        scrambled_still_fixed = Cube3.OPERATIONS['LU'](Cube3())
        permutation = find_stickers_permutations_to_orient_the_cube(scrambled_still_fixed)
        self.assertEqual(permutation, list(range(54)))

        scrambled_must_reorient = Cube3.OPERATIONS['RU'](Cube3())
        permutation = find_stickers_permutations_to_orient_the_cube(scrambled_must_reorient)
        self.assertEqual(permutation, [9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 21, 22, 23, 18,
                                       19, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 6, 7, 8, 3, 4, 5, 0,
                                       1, 2, 42, 39, 36, 43, 40, 37, 44, 41, 38, 47, 50, 53, 46, 49, 52,
                                       45, 48, 51])




class TestCube2(unittest.TestCase):

    def test_space_rotations_basic(self):
        for rotation_move in CUBE2_ROTATION_MOVES:
            state = Cube2()
            for i in range(4):
                state = rotation_move(state)
            self.assertEqual(
                state.as_stickers_vector,
                Cube2().as_stickers_vector,
                "Cube rotated 4 times is different from unrotated cube"
            )

    def test_space_rotations_3_left_1_right(self):
        for r1, r2 in CUBE2_OPPOSITE_ROTATION_MOVES:
            state1 = Cube2()
            for i in range(3):
                state1 = r1(state1)
            state2 = r2(Cube2())
            self.assertEqual(
                state1.as_stickers_vector,
                state2.as_stickers_vector,
                "Cube rotated 3 times is one direction and 1 time in the opposite direction should be the same"
            )

    def test_as_stickers_vector(self):
        self.assertEqual(
            Cube2().as_stickers_vector,
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
            "Solved cube3.as_stickers_vector should yield a solved cube2"
        )

    def test_build_cube_with_unique_sticker_codes(self):
        cube = cube_with_unique_sticker_codes(Cube2)
        self.assertEqual(sorted(cube.as_stickers_vector), list(range(1, 25)))


    def test_compute_all_cube_rotations_works(self):
        all_rotations = compute_all_cube_rotations(Cube2())
        self.assertEqual(
            24,
            len(all_rotations)
        )

    def test_sticker_patterns(self):
        sticker_patterns2 = compute_cubie_sticker_patterns_to_find_fixed_cubie_in_any_scramled_states(Cube2)
        self.assertEqual(
            sticker_patterns2,
            {1: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 2: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 3: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 4: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 5: [[6, 4, 1], [4, 1, 6], [1, 6, 4]], 6: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 7: [[1, 4, 6], [4, 6, 1], [6, 1, 4]], 8: [[6, 4, 1], [4, 1, 6], [1, 6, 4]]}
        )

    def test_find_fixed_cubie_in_scrambled_state(self):
        position, cubie, stickers = find_fixed_cubie_in_scrambled_state(Cube2())
        self.assertEqual(position, 4)
        self.assertEqual(stickers, [1, 4, 6])

        position, cubie, stickers = find_fixed_cubie_in_scrambled_state(CUBE2_OPERATIONS['RU'](Cube2()))
        self.assertEqual(position, 2)
        self.assertEqual(stickers, [4, 1, 6])

    def test_find_stickers_permutations_to_orient_the_cube(self):
        solved = Cube2()
        permutation = find_stickers_permutations_to_orient_the_cube(solved)
        self.assertEqual(permutation, list(range(24)))

        scrambled_still_fixed = Cube2.OPERATIONS['LU'](Cube2())
        permutation = find_stickers_permutations_to_orient_the_cube(scrambled_still_fixed)
        self.assertEqual(permutation, list(range(24)))

        scrambled_must_reorient = Cube2.OPERATIONS['RU'](Cube2())
        permutation = find_stickers_permutations_to_orient_the_cube(scrambled_must_reorient)
        self.assertEqual(permutation, [4, 5, 6, 7, 10, 11, 8, 9, 12, 13, 14, 15, 2, 3, 0, 1, 18, 16, 19, 17, 21, 23, 20, 22])


class TestCube2IntegrationWithCube3(unittest.TestCase):

    def test_applying_the_same_operations_on_both_cubes_simple(self):
        cube2_opnames = list(CUBE2_OPERATIONS.keys())
        for name in cube2_opnames:
            op2 = CUBE2_OPERATIONS[name]
            op3 = CUBE3_OPERATIONS[name]
            self.assertEqual(
                op3(Cube3()).as_cube2.as_stickers_vector,
                op2(Cube2()).as_stickers_vector,
                "Applying {0} to Cube2 and Cube3 should yield the same corner cubies.".format(name)
            )


if __name__ == '__main__':
    unittest.main()
