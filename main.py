from utils import scramble_steps, SOLVED_STATE, ScrambledState

POSSIBLE_STATES = [ScrambledState([], SOLVED_STATE)] + \
                  scramble_steps(SOLVED_STATE, steps=1) + \
                  scramble_steps(SOLVED_STATE, steps=2) + \
                  scramble_steps(SOLVED_STATE, steps=3)
