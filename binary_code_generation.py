import math
from collections import defaultdict
from functools import reduce
from typing import Dict, TypeVar, Type, Any, Generic

from cube2.cube import Cube2
from cube3.cube import Cube3
from orientation import compute_all_orienting_permutations_by_cubie_and_stickers
from utils import Cubie, cube_with_unique_sticker_codes, CubiesCube, AXIS_X, AXIS_Y, AXIS_Z, CubeSerializer, \
    StickerVectorSerializer, normalize_binary_string, AXES, compute_stickers_permutation

T = TypeVar("T", bound=CubiesCube)


class BinaryRepresentation(Generic[T]):

    def __init__(self, cube_class: Type[T]):
        self.cube_class = cube_class


class IntSpec:
    def __init__(self, bits_per_color: int, data_bits: int) -> None:
        self.bits_per_color = bits_per_color
        self.data_bits = data_bits
        self.int_size = 2 ** math.ceil(math.log(self.data_bits, 2))
        self.offset = self.int_size - self.data_bits

    @staticmethod
    def for_cube(cube_class: Type[T]):
        nb_colors = len(cube_class().as_stickers_vector)
        bits_per_color = 3
        data_bits = nb_colors * bits_per_color
        return IntSpec(bits_per_color, data_bits)


def build_cube3_to_cube2_bitwise_ops():
    unique_cube3 = cube_with_unique_sticker_codes(Cube3)
    unique_cube2 = unique_cube3.as_cube2
    size_diff = len(unique_cube3.as_stickers_vector) - len(unique_cube2.as_stickers_vector)

    old_indices = compute_stickers_permutation(unique_cube2, unique_cube3)
    new_indices = [i + size_diff for i in range(len(old_indices))]
    shifts_spec = dict(list(zip(old_indices, new_indices)))
    return sticker_wise_permutation_to_bitwise_ops(shifts_spec, IntSpec.for_cube(Cube3))


def build_binary_orientation_spec(cube_class: Type[T]) -> Dict[int, Any]:
    int_spec = IntSpec.for_cube(cube_class)
    bitwise_spec = {}
    all_orienting_permutations = compute_all_orienting_permutations_by_cubie_and_stickers(cube_class)
    for cubie_idx, cubie_spec in all_orienting_permutations.items():
        bitwise_spec[cubie_idx] = []
        for entry in cubie_spec:
            bitwise_spec[cubie_idx].append({
                "cubie": cubie_idx,
                "color_pattern": entry["color_pattern"],
                "color_detection_bitwise_lhs": color_detection_bitwise_ops(cube_class, cubie_idx),
                "color_detection_bitwise_rhs": (
                        (entry["color_pattern"][0] << int_spec.bits_per_color * 2) |
                        (entry["color_pattern"][1] << int_spec.bits_per_color * 1) |
                        entry["color_pattern"][2]
                ),
                "orient_cube_bitwise_op": permutation_to_bitwise_ops(entry["permutation"], int_spec)
            })
    return bitwise_spec


def color_detection_bitwise_ops(cube_class: Type[T], cubie_idx) -> Dict[int, int]:
    empty_cubie = Cubie(0, 0, 0)

    unique_cube = cube_with_unique_sticker_codes(cube_class)
    unique_cubies = unique_cube.cubies
    unique_stickers = unique_cube.as_stickers_vector
    lookup_order = [AXIS_X, AXIS_Y, AXIS_Z]
    lookup_stickers = []
    for axis in lookup_order:
        for face_nb in cube_class.AXIS_TO_FACES[axis]:
            for face_cubie_idx in cube_class.FACE_CUBIES_INDICES[face_nb]:
                sticker_value = unique_cubies[face_cubie_idx].get_face(axis)
                lookup_stickers.append(unique_stickers.index(sticker_value))

    mask_cube = cube_class(
        [empty_cubie] * (cubie_idx - 1) +
        [Cubie(7, 7, 7)] +  # 7 is binary 111
        [empty_cubie] * (cube_class.NB_CUBIES - cubie_idx)
    )

    mask_vector = mask_cube.as_stickers_vector
    xyz_stickers_positions = [sticker_nb for sticker_nb in lookup_stickers if mask_vector[sticker_nb] == 7]
    sticker_permutations = {
        _from: _to for _from, _to in zip(xyz_stickers_positions, range(len(unique_stickers) - 3, len(unique_stickers)))
    }

    state_shifts = sticker_wise_permutation_to_bitwise_ops(sticker_permutations, IntSpec.for_cube(cube_class))
    return state_shifts


def permutation_to_bitwise_ops(permutation_vector, int_spec: IntSpec):
    shifts_specification = dict(zip(permutation_vector, range(len(permutation_vector))))
    return sticker_wise_permutation_to_bitwise_ops(
        shifts_specification,
        int_spec
    )


def sticker_wise_permutation_to_bitwise_ops(shifts_specification: Dict[int, int],
                                            int_spec: IntSpec):
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in shifts_specification.items():
        bitwise_mask = ['0'] * int_spec.int_size
        color_offset = int_spec.offset + old_idx * int_spec.bits_per_color
        for color_bit_offset in range(int_spec.bits_per_color):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (old_idx - new_idx) * int_spec.bits_per_color
        hex_mask = int(bitwise_mask_str, 2)
        reverse_shifts[bitwise_offset] |= hex_mask
    return dict(reverse_shifts)


def apply_bitwise_shifts(shifts, number):
    return reduce(
        lambda acc, item: acc | apply_bitwise_shift(item[0], item[1], number),
        shifts.items(),
        0
    )


def apply_bitwise_shift(offset, mask, x):
    masked = x & mask
    abs_offset = abs(offset)
    return masked >> abs_offset if offset < 0 else masked << abs_offset


class StickerBinarySerializer(CubeSerializer[T]):

    def __init__(self, cube_class: Type[T]):
        self.int_spec = IntSpec.for_cube(cube_class)
        super().__init__(cube_class)

    def serialize(self, cube: T):
        vector = StickerVectorSerializer(self.cube_class).serialize(cube)
        return "".join(["0"] * self.int_spec.offset + ["{0:03b}".format(n) for n in vector])

    def unserialize(self, binary_string: str) -> T:
        binary_string = normalize_binary_string(binary_string, self.int_spec.data_bits)
        n = 3
        vector = [binary_string[i:i + n] for i in range(0, len(binary_string), n)]
        vector = [int(part, 2) for part in vector]
        return StickerVectorSerializer(self.cube_class).unserialize(vector)


class IntSerializer(CubeSerializer[T]):

    def __init__(self, binary_serializer: CubeSerializer) -> None:
        self.binary_serializer = binary_serializer
        super().__init__(self.binary_serializer.cube_class)

    def serialize(self, cube: T):
        binary = self.binary_serializer.serialize(cube)
        return int(binary, 2)

    def unserialize(self, number: int) -> T:
        return self.binary_serializer.unserialize("{0:03b}".format(number))



class CodeGenerator(Generic[T]):

    def __init__(self, cube_class: Type[T]) -> None:
        self.cube_class = cube_class
        self.binary_serializer = StickerBinarySerializer(cube_class)
        self.int_serializer = IntSerializer(self.binary_serializer)
        self.solved_cube_binary = self.binary_serializer.serialize(self.solved_cube)
        self.solved_cube_int = self.int_serializer.serialize(self.solved_cube)
        super().__init__()

    @property
    def int_spec(self):
        return IntSpec.for_cube(self.cube_class)

    @property
    def solved_cube(self):
        return self.cube_class()

    def build(self):
        bitwise_shifts = compute_stickers_bitwise_shifts_per_operation(self.cube_class)
        moves_fn_names = list(bitwise_shifts.keys())
        fixed_cubie_fn_names = [move for move in moves_fn_names if move not in ["RU", "RD", "DR", "DL", "FR", "FL"]]
        return "\n".join([
            self.build_autogenerated_comment(),
            self.build_cube3_to_cube2(build_cube3_to_cube2_bitwise_ops()),
            self.build_cube_orientation_code(build_binary_orientation_spec(self.cube_class)),
            self.build_operations_code(bitwise_shifts),
            self.build_main_function(moves_fn_names, fixed_cubie_fn_names),
        ])

    def build_autogenerated_comment(self):
        raise NotImplementedError()

    def build_cube_orientation_code(self, orientation_spec):
        raise NotImplementedError()

    def build_operations_code(self, bitwise_shifts_per_op):
        raise NotImplementedError()

    def build_cube3_to_cube2(self, shift_spec):
        raise NotImplementedError()

    def build_main_function(self, moves_fn_names, fixed_cubie_fn_names):
        raise NotImplementedError()


class PythonCodeGenerator(CodeGenerator):

    def build_autogenerated_comment(self):
        return """'''
This file was auto-generated, do not change manually
'''"""

    def build_cube3_to_cube2(self, shift_spec):
        shifts = stringify_bitwise_shifts(shift_spec)
        return f'''
def cube3_to_cube2(x):
    return {shifts}

'''

    def build_cube_orientation_code(self, definitions):
        code = []
        code.append("""
def orient_cube(x:int):""")
        for cubie, definitions in definitions.items():
            actual_color_pattern = stringify_bitwise_shifts(definitions[0]['color_detection_bitwise_lhs'])
            code.append(f'    # Cubie {cubie}')
            code.append(f'    actual_color_pattern = {actual_color_pattern}')
            for definition in definitions:
                oriented_cube = stringify_bitwise_shifts(definition['orient_cube_bitwise_op'])
                code.append(f'    if actual_color_pattern == {definition["color_detection_bitwise_rhs"]}:')
                code.append(f'        return {oriented_cube}')
        code.append('    raise Exception("State {0} was not possible to orient to fix cubie in place".format(x))')
        return "\n".join(code)


    def build_operations_code(self, bitwise_shifts):
        code = []
        ops = list(bitwise_shifts.items())
        for name, shifts in ops:
            code.append(f"""
def {name.lower()}(x):
    return {stringify_bitwise_shifts(shifts)}
    """)
        return "\n".join(code)

    def build_main_function(self, moves_fn_names, fixed_cubie_fn_names):
        return f"""
OPS = [{(", ".join(moves_fn_names)).lower()}]
OPS_DICT = {{fn.__name__: fn for fn in OPS}}
FIXED_CUBIE_OPS = [{(", ".join(fixed_cubie_fn_names)).lower()}]
FIXED_CUBIE_OPS_DICT = {{fn.__name__: fn for fn in FIXED_CUBIE_OPS}}
SOLVED_CUBE_STATE = {self.solved_cube_int}
def main():
    pass
    """


class RustCodeGenerator(CodeGenerator):

    @property
    def int_type(self):
        return 'i%d' % self.int_spec.int_size

    def build_cube3_to_cube2(self, shift_spec):
        cube3_spec = IntSpec.for_cube(Cube3)
        cube2_spec = IntSpec.for_cube(Cube2)
        shifts = stringify_bitwise_shifts(shift_spec)
        return f'''
fn cube3_to_cube2(x: i{cube3_spec.int_size}) -> i{cube2_spec.int_size} {{
    return ({shifts}) as i{cube2_spec.int_size};
}}
'''

    def build_autogenerated_comment(self):
        return "// This file was auto-generated, do not change manually"

    def build_cube_orientation_code(self, definitions):
        code = []
        code.append(f"""
pub fn orient_cube(x: {self.int_type}) -> {self.int_type} {{""")
        code.append(f'    let mut actual_color_pattern: {self.int_type};')
        for cubie, definitions in definitions.items():
            actual_color_pattern = stringify_bitwise_shifts(definitions[0]['color_detection_bitwise_lhs'])
            code.append(f'    actual_color_pattern = {actual_color_pattern};')
            for definition in definitions:
                oriented_cube = stringify_bitwise_shifts(definition['orient_cube_bitwise_op'])
                code.append(f'    if actual_color_pattern == {definition["color_detection_bitwise_rhs"]} {{')
                code.append(f'        return {oriented_cube};')
                code.append(f'    }}')
        code.append('    panic!("State was not possible to orient: {}", x);')
        code.append(f'}}')
        return "\n".join(code)

    def build_operations_code(self, bitwise_shifts):
        code = []
        for name, shifts in bitwise_shifts.items():
            code.append(f"""
pub fn {name.lower()}(x: {self.int_type}) -> {self.int_type} {{
    return {stringify_bitwise_shifts(shifts)};
}}""")
        return "\n".join(code)

    def build_main_function(self, moves_fn_names, fixed_cubie_fn_names):
        return f"""
// self.solved_cube_binary
pub static ALL_OPERATIONS: &'static [fn(i128) -> i128] = &[{(", ".join(moves_fn_names)).lower()}];
pub static FIXED_CUBIE_OPERATIONS: &'static [fn(i128) -> i128] = &[{(", ".join(fixed_cubie_fn_names)).lower()}];
pub static SOLVED_STATE: i128 = {hex(self.solved_cube_int)};
    """


def compute_stickers_bitwise_shifts_per_operation(cube_class: Type[T]):
    unique_cube = cube_with_unique_sticker_codes(cube_class)

    bitwise_shifts = {}
    for name, op in cube_class.OPERATIONS.items():
        permutation = compute_stickers_permutation(op(unique_cube), unique_cube)
        bitwise_shifts[name] = permutation_to_bitwise_ops(permutation, IntSpec.for_cube(cube_class))
    return bitwise_shifts


def stringify_bitwise_shifts(shifts):
    as_str = []
    for offset, mask in shifts.items():
        direction = '>>' if offset < 0 else '<<'
        as_str.append(f'((x & {hex(mask)}) {direction} {abs(offset)})')
    return "|".join(as_str)
