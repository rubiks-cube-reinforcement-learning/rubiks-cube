from typing import TypeVar, Type

from binary_representation import build_binary_orientation_spec, permutation_to_bitwise_ops
from cube2.cube import StickerBinarySerializer, IntSerializer, Cube2, OPERATIONS
from utils import compute_stickers_permutation, CubiesCube, cube_with_unique_sticker_codes

# For now this is specific to Cube2
BINARY_SERIALIZER = StickerBinarySerializer()
INT_SERIALIZER = IntSerializer(BINARY_SERIALIZER)
SOLVED_CUBE = Cube2()
SOLVED_CUBE_BINARY = BINARY_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_INT = INT_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_HEX = hex(SOLVED_CUBE_INT)

T = TypeVar("T", bound=CubiesCube)


def compute_stickers_bitwise_shifts_per_operation(cube_class: Type[T]):
    unique_cube = cube_with_unique_sticker_codes(cube_class)

    bitwise_shifts = {}
    for name, op in OPERATIONS.items():
        permutation = compute_stickers_permutation(op(unique_cube), unique_cube)
        bitwise_shifts[name] = permutation_to_bitwise_ops(permutation)
    return bitwise_shifts


def print_rust_code(cube_class: Type[T]):
    print(SOLVED_CUBE_BINARY)
    print(SOLVED_CUBE_HEX)
    print(f"""
fn main() {{
	let solved_state:i64 = {SOLVED_CUBE_HEX};
}}
""")
    print_rust_orienting_permutation(build_binary_orientation_spec(cube_class))
    print_rust_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation(cube_class))


def print_python_code(cube_class: Type[T]):
    print("""'''
This file was auto-generated, do not change manually
'''""")
    serializer_name = 'StickerBinarySerializer'
    print_python_orienting_permutation(build_binary_orientation_spec(cube_class))
    print_python_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation(cube_class))

    ops = list(OPERATIONS.keys())
    ops_str = (", ".join(ops)).lower()
    print(f"""
from cubies import *
from random import shuffle
def main():
    OPS = [{ops_str}]
    SOLVED_CUBE_INT = {SOLVED_CUBE_INT}
    print(SOLVED_CUBE_INT)
    current = SOLVED_CUBE_INT
    test = ru(SOLVED_CUBE_INT)
    print(test)
    print(orient_cube(test))
    for op in OPS:
        shuffle(OPS)
    #     for i in range(10):
    #         current = op(current)
    #     c = IntSerializer({serializer_name}()).unserialize(current)
    #     is_valid = c.assert_valid()

main()
""")


def print_rust_bitwise_shifts(bitwise_shifts):
    int_type = 'i128'
    for name, shifts in bitwise_shifts.items():
        print(f"""
fn {name.lower()}(x: {int_type}) -> {int_type} {{
    return {stringify_bitwise_shifts(shifts)};
}}""")


def print_rust_orienting_permutation(definitions):
    int_type = 'i128'
    print(f"""
fn orient_cube(x: {int_type}) -> {int_type} {{""")
    print(f'    let mut actual_color_pattern: {int_type};')
    for cubie, definitions in definitions.items():
        actual_color_pattern = stringify_bitwise_shifts(definitions[0]['color_detection_bitwise_lhs'])
        print(f'    actual_color_pattern = {actual_color_pattern};')
        for definition in definitions:
            oriented_cube = stringify_bitwise_shifts(definition['orient_cube_bitwise_op'])
            print(f'    if actual_color_pattern == {definition["color_detection_bitwise_rhs"]} {{')
            print(f'        return {oriented_cube};')
            print(f'    }}')
    print('    panic!("State was not possible to orient: {}", x);')
    print(f'}}')


def print_python_bitwise_shifts(bitwise_shifts):
    ops = list(bitwise_shifts.items())
    for name, shifts in ops:
        print(f"""
def {name.lower()}(x):
    return {stringify_bitwise_shifts(shifts)}
""")


def print_python_orienting_permutation(definitions):
    print("""
def orient_cube(x:int):""")
    for cubie, definitions in definitions.items():
        actual_color_pattern = stringify_bitwise_shifts(definitions[0]['color_detection_bitwise_lhs'])
        print(f'    # Cubie {cubie}')
        print(f'    actual_color_pattern = {actual_color_pattern}')
        for definition in definitions:
            oriented_cube = stringify_bitwise_shifts(definition['orient_cube_bitwise_op'])
            print(f'    if actual_color_pattern == {definition["color_detection_bitwise_rhs"]}:')
            print(f'        return {oriented_cube}')
    print('    raise Exception("State {0} was not possible to orient to fix cubie 4 in place".format(x))')


def stringify_bitwise_shifts(shifts):
    as_str = []
    for offset, mask in shifts.items():
        direction = '>>' if offset < 0 else '<<'
        as_str.append(f'((x & {hex(mask)}) {direction} {abs(offset)})')
    return "|".join(as_str)
