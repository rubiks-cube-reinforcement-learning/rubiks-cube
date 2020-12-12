from cube2.binary_representation import find_permutations_to_orient_the_cube, CUBE_WITH_UNIQUE_STICKER_CODES, \
    compute_bitwise_shifts
from cube2.cube import StickerBinarySerializer, IntSerializer, Cube2, OPERATIONS
from utils import compute_stickers_permutation

MODE_STICKERS = 'stickers'
MODE_CUBIES = 'cubies'
MODE = MODE_STICKERS
BINARY_SERIALIZER = StickerBinarySerializer()
INT_SERIALIZER = IntSerializer(BINARY_SERIALIZER)
SOLVED_CUBE = Cube2()
SOLVED_CUBE_BINARY = BINARY_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_INT = INT_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_HEX = hex(SOLVED_CUBE_INT)


def compute_stickers_bitwise_shifts_per_operation():
    bitwise_shifts = {}
    for name, op in OPERATIONS.items():
        permutation = compute_stickers_permutation(op(CUBE_WITH_UNIQUE_STICKER_CODES), CUBE_WITH_UNIQUE_STICKER_CODES)
        bitwise_shifts[name] = compute_bitwise_shifts(permutation)
    return bitwise_shifts


def print_rust_code():
    print(SOLVED_CUBE_BINARY)
    print(SOLVED_CUBE_HEX)
    print(f"""
fn main() {{
	let solved_state:i64 = {SOLVED_CUBE_HEX};
}}
""")
    if MODE is MODE_STICKERS:
        print_rust_orienting_permutation(find_permutations_to_orient_the_cube())
        print_rust_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation())
    elif MODE is MODE_CUBIES:
        pass
        # @TODO maybe


def print_python_code():
    print("""'''
This file was auto-generated, do not change manually
'''""")
    if MODE is MODE_STICKERS:
        serializer_name = 'StickerBinarySerializer'
        print_python_orienting_permutation(find_permutations_to_orient_the_cube())
        print_python_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation())
    elif MODE is MODE_CUBIES:
        serializer_name = 'CubieVectorBinarySerializer'
        # @TODO maybe

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
    int_type = 'i128' if MODE is MODE_STICKERS else 'i64'
    for name, shifts in bitwise_shifts.items():
        print(f"""
fn {name.lower()}(x: {int_type}) -> {int_type} {{
    return {stringify_bitwise_shifts(shifts)};
}}""")


def print_rust_orienting_permutation(definitions):
    int_type = 'i128' if MODE is MODE_STICKERS else 'i64'
    print(f"""
fn orient_cube(x: {int_type}) -> {int_type} {{""")
    print(f'    let mut actual_color_pattern: {int_type};')
    for cubie, definitions in definitions.items():
        actual_color_pattern = stringify_bitwise_shifts(definitions[0]['bitwise_color_detection_shifts'])
        print(f'    actual_color_pattern = {actual_color_pattern};')
        for definition in definitions:
            oriented_cube = stringify_bitwise_shifts(definition['bitwise_orienting_permutation'])
            print(f'    if actual_color_pattern == {definition["color_pattern"]} {{')
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
        actual_color_pattern = stringify_bitwise_shifts(definitions[0]['bitwise_color_detection_shifts'])
        print(f'    # Cubie {cubie}')
        print(f'    actual_color_pattern = {actual_color_pattern}')
        for definition in definitions:
            oriented_cube = stringify_bitwise_shifts(definition['bitwise_orienting_permutation'])
            print(f'    if actual_color_pattern == {definition["color_pattern"]}:')
            print(f'        return {oriented_cube}')
    print('    raise Exception("State {0} was not possible to orient to fix cubie 4 in place".format(x))')


def stringify_bitwise_shifts(shifts):
    as_str = []
    for offset, mask in shifts.items():
        direction = '>>' if offset < 0 else '<<'
        as_str.append(f'((x & {hex(mask)}) {direction} {abs(offset)})')
    return "|".join(as_str)
