from pathlib import Path

from code_generation import PythonCodeGenerator, RustCodeGenerator
from cube2.solver import find_solution
from cube2.cube import Cube2
from cube3.cube import Cube3
from cube3.solver import generate_binary_dataset as generate_binary_dataset_3
from cube3.generated_stickers_bitwise_ops import orient_cube as orient_cube_3, \
    cube3_to_cube2, FIXED_CUBIE_OPS_DICT as ops3

def refresh_bitwise_code():
    base = Path(__file__).parent
    with Path(base / "cube2/generated_stickers_bitwise_ops.py").open('w+') as fp:
        fp.write(PythonCodeGenerator(Cube2).build())

    with Path(base / "cube3/generated_stickers_bitwise_ops.py").open('w+') as fp:
        fp.write(PythonCodeGenerator(Cube3).build())

    with Path(base / "rust-experiment/src/cube2.rs").open('w+') as fp:
        fp.write(RustCodeGenerator(Cube2).build())

    with Path(base / "rust-experiment/src/cube3.rs").open('w+') as fp:
        fp.write(RustCodeGenerator(Cube3).build())


def solve_3_cube_corners(cube3_scrambled):
    cube3_oriented = orient_cube_3(cube3_scrambled)
    cube2 = cube3_to_cube2(cube3_oriented)
    corners_solution = find_solution(cube2)
    for move in corners_solution:
        cube3_oriented = ops3[move](cube3_oriented)
    return cube3_oriented


def fun():
    dataset_3 = generate_binary_dataset_3(10, 100)
    for cube3_scrambled in dataset_3:
        cube3_solved_corners = solve_3_cube_corners(cube3_scrambled)


if __name__ == '__main__':
    # refresh_bitwise_code()
    fun()
