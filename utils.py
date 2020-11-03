from collections import namedtuple
from rubik.cube import Cube

SOLVED_STATE = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
ACTIONS = dict([
    ('L', 'Li'), ('R', 'Ri'), ('U', 'Ui'), ('D', 'Di'), ('F', 'Fi'), ('B', 'Bi'),
    ('M', 'Mi'), ('E', 'Ei'), ('S', 'Si'), ('X', 'Xi'), ('Y', 'Yi'), ('Z', 'Zi'),
])
ACTIONS.update({v: k for (k, v) in ACTIONS.items()})

NUM_ACTIONS = len(ACTIONS)
NUM_STATES = 43252003274489856000

ENCODING = {
    "O": 0,
    "Y": 1,
    "W": 2,
    "G": 3,
    "B": 4,
    "R": 5,
}
def state_to_vector(state):
    return [ENCODING[char] for char in state]

ScrambledState = namedtuple('ScrambledState', ['moves', 'state'])

def scramble(state: str):
    results = []
    for move, inverse_move in ACTIONS.items():
        c = Cube(state)
        getattr(c, move)()
        results.append(ScrambledState([inverse_move], c.flat_str()))
    return results


def scramble_steps(initial_state: str = SOLVED_STATE, steps=2):
    examples = scramble(initial_state)
    if steps == 1:
        return examples

    retval = []
    for (moves, state) in examples:
        for new_example in scramble_steps(state, steps - 1):
            retval.append(ScrambledState(moves + new_example.moves, new_example.state))
    return retval
