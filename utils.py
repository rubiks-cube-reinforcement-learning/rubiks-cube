from itertools import chain

from rubik_solver.Cubie import Cube
from rubik_solver.Move import Move
from rubik_solver.NaiveCube import NaiveCube
from rubik_solver.utils import solve
import random

class KociembaCube:
    SOLVED_STATE = "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
    FACES = [
        "y", "b", "r", "g", "o", "w"
    ]
    MOVES = (
        "F", "R", "U", "L", "B", "D",
        "F'", "R'", "U'", "L'", "B'", "D'",
        # "M", "E", "S",
        # "M'", "E'", "S'",
    )

    def __init__(self, state=SOLVED_STATE) -> None:
        self.reset(state=state)

    def __str__(self):
        return self.cube.to_naive_cube().get_cube().lower()

    def as_vector(self):
        return [self.FACES.index(char) for char in self.__str__()]

    def reset(self, state=SOLVED_STATE):
        naive_cube = NaiveCube()
        naive_cube.set_cube(state)
        self.cube = Cube()
        self.cube.from_naive_cube(naive_cube)

    def is_solved(self):
        self.cube.to_naive_cube().is_solved()

    def move(self, action):
        self.cube.move(Move(action))

    def get_random_move(self):
        return random.choice(self.MOVES)

    def get_move_by_idx(self, nb):
        return self.MOVES[nb]

    def get_solution(self):
        solver_solution = list(map(str, solve(self.cube, 'Kociemba')))
        solution = flatten([([x[0]] * 2 if x[-1] == "2" else [x]) for x in solver_solution])
        return solution

    def scramble(self, steps=2):
        sequence = []
        for _ in range(steps):
            m = self.get_random_move()
            self.move(m)
            sequence.append(m)
        return sequence


def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

if __name__ == "__main__":
    c = KociembaCube()
    print(c.scramble(2))
    print(c.__str__())
    print(c.get_solution())
    print(c.as_vector())