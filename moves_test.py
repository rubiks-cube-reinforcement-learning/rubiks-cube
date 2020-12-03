import unittest
from functools import reduce

from moves import lu, ld, ru, rd, ul, ur, dl, dr, fl, fr, bl, br

class MovesTestCase(unittest.TestCase):
    test_state = [
        1.1, 1.2, 1.3, 1.4,
        2.1, 2.2, 2.3, 2.4,
        3.1, 3.2, 3.3, 3.4,
        4.1, 4.2, 4.3, 4.4,
        5.1, 5.2, 5.3, 5.4,
        6.1, 6.2, 6.3, 6.4,
    ]

    moves = [
        lu, ld, ru, rd, ul, ur, dl, dr, fl, fr
    ]

    def test_moves_return_vector_with_24_elements(self):
        for move in self.moves:
            self.assertEqual(24, len(move(self.test_state)), "move: %s" % move.__name__)

    def test_moves_return_vector_with_24_unique_elements(self):
        for move in self.moves:
            self.assertEqual(24, len(set(move(self.test_state))), "move: %s" % move.__name__)

    def test_applying_any_move_4_times_results_in_the_starting_state(self):
        for move in self.moves:
            new_state = move(move(move(move(self.test_state))))
            self.assertListEqual(self.test_state, new_state, "move: %s" % move.__name__)

    def test_opposite_moves_cancel_each_other(self):
        opposite_pairs = [
            (lu, ld),
            (ru, rd),
            (dl, dr),
            (ul, ur),
            (fl, fr),
            (bl, br),
        ]

        for a, b in opposite_pairs:
            new_state = a(b(self.test_state))
            self.assertListEqual(self.test_state, new_state, "moves: %s %s" % (a.__name__, b.__name__))

        for a, b in opposite_pairs:
            new_state = b(a(self.test_state))
            self.assertListEqual(self.test_state, new_state, "moves: %s %s" % (a.__name__, b.__name__))

    # 1 - green
    # 2 - blue
    # 3 - red
    # 4 - yellow
    # 5 - white
    # 6 - orange
    def test_basic_scramble_and_solve_1(self):
        steps = [ur, ru, ru]
        scrambled_state = [5, 1, 4, 1, 2, 5, 2, 4, 5, 2, 4, 2, 1, 5, 1, 4, 3, 6, 3, 6, 3, 3, 6, 6]
        solved_state = [1, 1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 5, 5, 5, 5, 3, 3, 3, 3, 6, 6, 6, 6]
        self.assertListEqual(solved_state, solve(scrambled_state, steps))

    def test_basic_scramble_and_solve_2(self):
        steps = [ru, ul, fr, fr, ul, fr, ur, ur, fl, ru, ur, ur, fl]
        scrambled_state = [3, 5, 1, 6, 3, 4, 5, 6, 2, 4, 1, 2, 6, 4, 5, 3, 3, 4, 5, 1, 6, 2, 2, 1]
        solved_state = [6, 6, 6, 6, 5, 5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1]
        self.assertListEqual(solved_state, solve(scrambled_state, steps))

def solve(state, steps):
    return reduce(lambda state, step: step(state), steps, state)

if __name__ == '__main__':
    unittest.main()
