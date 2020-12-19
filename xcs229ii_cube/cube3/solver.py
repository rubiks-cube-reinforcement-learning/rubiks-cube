try:
    from xcs229ii_cube.cube3.generated_stickers_bitwise_ops import FIXED_CUBIE_OPS, OPS, SOLVED_CUBE_STATE
except:
    pass
from xcs229ii_cube.utils import generate_dataset, scramble

def generate_binary_dataset(nb_per_scramble, max_scrambles, fixed_cubie=False):
    scramble_fn = lambda n: scramble(SOLVED_CUBE_STATE, n, FIXED_CUBIE_OPS if fixed_cubie else OPS)
    return generate_dataset(nb_per_scramble, max_scrambles, scramble_fn)

