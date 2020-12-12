# coding=utf-8
from collections import defaultdict
from functools import reduce
from typing import List, Dict

from loggers import getLogger
from utils import flatten, partition, normalize_binary_string

logger = getLogger(__name__)

AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3


class Cubie:
    def __init__(self, face_x, face_y, face_z, idx=None):
        self.idx = idx
        self.face_x = face_x
        self.face_y = face_y
        self.face_z = face_z

    def __repr__(self):
        return "Cubie (%s, %s, %s, %s)" % (self.face_x, self.face_y, self.face_z, self.idx)

    def __str__(self):
        return (self.face_x, self.face_y, self.face_z, self.idx,)

    @property
    def faces(self):
        return [self.face_x, self.face_y, self.face_z]

    def get_face(self, axis):
        if axis == AXIS_X:
            return self.face_x
        elif axis == AXIS_Y:
            return self.face_y
        elif axis == AXIS_Z:
            return self.face_z

    def set_face(self, axis, color):
        new_cubie = self.clone()
        if axis == AXIS_X:
            new_cubie.face_x = color
        elif axis == AXIS_Y:
            new_cubie.face_y = color
        elif axis == AXIS_Z:
            new_cubie.face_z = color
        return new_cubie

    def clone(self):
        return Cubie(self.face_x, self.face_y, self.face_z, self.idx)


class Operation:
    def __init__(self, mapping, axis):
        self.mapping = mapping
        self.axis = axis

    def reverse(self):
        return Operation({v: k for k, v in self.mapping.items()}, self.axis)

    def __call__(self, cube):
        new_cube = cube.clone()
        for k, v in self.mapping.items():
            cubie = new_cube.cubies[v] = cube.cubies[k].clone()
            if self.axis == AXIS_X:
                cubie.face_y, cubie.face_z = cubie.face_z, cubie.face_y
            elif self.axis == AXIS_Y:
                cubie.face_x, cubie.face_z = cubie.face_z, cubie.face_x
            elif self.axis == AXIS_Z:
                cubie.face_x, cubie.face_y = cubie.face_y, cubie.face_x
        return new_cube


LU = Operation({1: 5, 5: 7, 7: 3, 3: 1}, AXIS_Z)
LD = LU.reverse()
RU = Operation({2: 6, 6: 8, 8: 4, 4: 2}, AXIS_Z)
RD = RU.reverse()

FL = Operation({1: 3, 3: 4, 4: 2, 2: 1}, AXIS_X)
FR = FL.reverse()
BL = Operation({5: 7, 7: 8, 8: 6, 6: 5}, AXIS_X)
BR = BL.reverse()

UL = Operation({1: 5, 5: 6, 6: 2, 2: 1}, AXIS_Y)
UR = UL.reverse()
DL = Operation({3: 7, 7: 8, 8: 4, 4: 3}, AXIS_Y)
DR = DL.reverse()

OPERATIONS = {'LU': LU, 'LD': LD,
              'RU': RU, 'RD': RD,
              'FL': FL, 'FR': FR,
              'BL': BL, 'BR': BR,
              'UL': UL, 'UR': UR,
              'DL': DL, 'DR': DR}

CUBE_ROTATION_MOVES = [
    lambda state: LU(RU(state)),
    lambda state: FL(BL(state)),
    lambda state: UL(DL(state)),
    lambda state: LD(RD(state)),
    lambda state: FR(BR(state)),
    lambda state: UR(DR(state)),
]


class CubiesCube:
    NB_CUBIES = 8

    CUBIES_WITH_EVEN_ORIENTATION = [2, 3, 5, 8]
    CUBIES_WITH_ODD_ORIENTATION = [1, 4, 6, 7]

    FACE_CUBIES_IDX = {
        1: [1, 2, 3, 4],
        2: [5, 6, 1, 2],
        3: [5, 6, 7, 8],
        4: [7, 8, 3, 4],
        5: [5, 1, 7, 3],
        6: [2, 6, 4, 8],
    }

    FACE_TO_AXIS = {
        1: AXIS_X,
        3: AXIS_X,
        2: AXIS_Y,
        4: AXIS_Y,
        5: AXIS_Z,
        6: AXIS_Z,
    }

    def __init__(self, cubies=None):
        if cubies is None:
            cubies = [
                Cubie(1, 2, 5, 1),
                Cubie(1, 2, 6, 2),
                Cubie(1, 4, 5, 3),
                Cubie(1, 4, 6, 4),
                Cubie(3, 2, 5, 5),
                Cubie(3, 2, 6, 6),
                Cubie(3, 4, 5, 7),
                Cubie(3, 4, 6, 8),
            ]
        self.cubies = [Cubie(0, 0, 0, 0)] + cubies  # 1-based indexing

    def __str__(self):
        faces = [self.face(i) for i in range(1, 7)]
        return faces.__str__()

    @property
    def as_stickers_vector(self):
        return StickerVectorSerializer().serialize(self)

    @property
    def as_stickers_int(self):
        return IntSerializer(StickerBinarySerializer()).serialize(self)

    @property
    def as_stickers_binary_string(self):
        return StickerBinarySerializer().serialize(self)

    def clone(self):
        return CubiesCube([c.clone() for c in self.cubies[1:]])

    def face(self, nb):
        axis = self.FACE_TO_AXIS[nb]
        colors = [self.cubies[idx].get_face(axis) for idx in self.FACE_CUBIES_IDX[nb]]
        return colors


class CubeSerializer:
    def serialize(self, cube: CubiesCube):
        raise NotImplementedError()

    def unserialize(self, value) -> CubiesCube:
        raise NotImplementedError()

    def clone_cube(self, cube):
        return self.unserialize(self.serialize(cube))


class CubieVectorSerializer(CubeSerializer):

    def __init__(self, reference_cube=None) -> None:
        if reference_cube is None:
            reference_cube = CubiesCube()
        self.reference_cube = reference_cube
        super().__init__()

    def serialize(self, shuffled_cube: CubiesCube):
        reference_cube = self.reference_cube
        cubies_indexes = []
        x_face_indexes = []
        for i in range(1, 9):
            shuffled_cubie = shuffled_cube.cubies[i]
            reference_cubie = [c for c in reference_cube.cubies if c.idx == shuffled_cubie.idx][0]
            cubies_indexes += [shuffled_cubie.idx]
            x_face_indexes += [reference_cubie.faces.index(shuffled_cubie.face_x)]
        return cubies_indexes + x_face_indexes

    def unserialize(self, vector) -> CubiesCube:
        reference_cube = self.reference_cube
        unserialized_cube = CubiesCube()
        cubies_indexes = vector[:8]
        x_face_indexes = vector[8:]
        for i, (cubie_idx, x_face_index) in enumerate(zip(cubies_indexes, x_face_indexes)):
            new_cubie_idx = i + 1
            x, y, z = self.decode_x_y_z(
                reference_cube.cubies[cubie_idx],
                x_face_index,
                new_cubie_idx
            )
            unserialized_cube.cubies[new_cubie_idx] = Cubie(x, y, z, cubie_idx)
        return unserialized_cube

    def decode_x_y_z(self, ref_cubie, x_face_idx, new_idx) -> List[int]:
        ref_idx = ref_cubie.idx

        seq = ref_cubie.faces
        even = CubiesCube.CUBIES_WITH_EVEN_ORIENTATION
        changed_orientation = (ref_idx in even) ^ (new_idx in even)
        if changed_orientation:
            seq = seq[::-1]

        new_x = ref_cubie.faces[x_face_idx]
        seq_x_idx = seq.index(new_x)
        x, y, z = seq[seq_x_idx], seq[seq_x_idx - 2], seq[seq_x_idx - 1]

        return [x, y, z]


class CubieVectorBinarySerializer(CubeSerializer):
    INT_LENGTH = 64
    CUBIES = 8
    IDX_BITS = 3
    ROTATION_BITS = 2
    IDXES_LENGTH = CUBIES * IDX_BITS
    ROTATIONS_LENGTH = CUBIES * ROTATION_BITS
    DATA_LENGTH = IDXES_LENGTH + ROTATIONS_LENGTH
    OFFSET = INT_LENGTH - DATA_LENGTH

    def serialize(self, cube: CubiesCube):
        vector = CubieVectorSerializer().serialize(cube)
        binary_string = ["0"] * self.OFFSET
        binary_string += ["{0:03b}".format(n - 1) for n in vector[:8]]
        binary_string += ["{0:02b}".format(n) for n in vector[8:]]
        return "".join(binary_string)

    def unserialize(self, binary_string: str) -> CubiesCube:
        binary_string = normalize_binary_string(binary_string, self.DATA_LENGTH)
        binary_idxes = partition(binary_string[:self.IDXES_LENGTH], self.IDX_BITS)
        binary_rotations = partition(binary_string[self.IDXES_LENGTH:], self.ROTATION_BITS)

        idxes = [int(binary_idx, 2) + 1 for binary_idx in binary_idxes]
        rotations = [int(binary_rotation, 2) for binary_rotation in binary_rotations]
        return CubieVectorSerializer().unserialize(idxes + rotations)


class StickerVectorSerializer(CubeSerializer):
    def serialize(self, cube: CubiesCube):
        vector = []
        for i in range(1, 7):
            vector += cube.face(i)
        return vector

    def unserialize(self, vector) -> CubiesCube:
        cube = CubiesCube()
        n = 4
        faces = [vector[i:i + n] for i in range(0, len(vector), n)]
        for i, colors in enumerate(faces):
            face_nb = i + 1
            axis = CubiesCube.FACE_TO_AXIS[face_nb]
            cubies = CubiesCube.FACE_CUBIES_IDX[face_nb]
            for i, color in enumerate(colors):
                cube.cubies[cubies[i]] = cube.cubies[cubies[i]].set_face(axis, color)
        return cube


class StickerBinarySerializer(CubeSerializer):
    INT_LENGTH = 128
    CUBIES_NB = 24
    COLOR_BITS = 3
    DATA_LENGTH = COLOR_BITS * CUBIES_NB
    OFFSET = INT_LENGTH - DATA_LENGTH

    def serialize(self, cube: CubiesCube):
        vector = StickerVectorSerializer().serialize(cube)
        return "".join(["0"] * self.OFFSET + ["{0:03b}".format(n) for n in vector])

    def unserialize(self, binary_string: str) -> CubiesCube:
        binary_string = normalize_binary_string(binary_string, self.DATA_LENGTH)
        n = 3
        vector = [binary_string[i:i + n] for i in range(0, len(binary_string), n)]
        vector = [int(part, 2) for part in vector]
        return StickerVectorSerializer().unserialize(vector)


class IntSerializer(CubeSerializer):

    def __init__(self, binary_serializer: CubeSerializer) -> None:
        self.binary_serializer = binary_serializer
        super().__init__()

    def serialize(self, cube: CubiesCube):
        binary = self.binary_serializer.serialize(cube)
        return int(binary, 2)

    def unserialize(self, number: int) -> CubiesCube:
        return self.binary_serializer.unserialize("{0:03b}".format(number))


def compute_stickers_permutation(target_cube: CubiesCube, source_cube: CubiesCube):
    serializer = StickerVectorSerializer()
    source_vector = serializer.serialize(source_cube)
    target_vector = serializer.serialize(target_cube)
    return [source_vector.index(i) for i in target_vector]


def compute_bitwise_shifts(permutation_vector, *args):
    shifts_specification = dict(zip(permutation_vector, range(len(permutation_vector))))
    return compute_bitwise_shifts_for_specific_stickers(
        shifts_specification,
        *args
    )


def compute_bitwise_shifts_for_specific_stickers(shifts_specification: Dict[int, int],
                                                 offset=StickerBinarySerializer.OFFSET,
                                                 int_size=StickerBinarySerializer.INT_LENGTH,
                                                 bucket_size=StickerBinarySerializer.COLOR_BITS):
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in shifts_specification.items():
        bitwise_mask = ['0'] * int_size
        color_offset = offset + old_idx * bucket_size
        for color_bit_offset in range(bucket_size):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (old_idx - new_idx) * bucket_size
        hex_mask = int(bitwise_mask_str, 2)
        reverse_shifts[bitwise_offset] |= hex_mask
    return dict(reverse_shifts)


def stringify_bitwise_shifts(shifts):
    as_str = []
    for offset, mask in shifts.items():
        direction = '>>' if offset < 0 else '<<'
        as_str.append(f'((x & {hex(mask)}) {direction} {abs(offset)})')
    return "|".join(as_str)


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


CUBE_WITH_UNIQUE_STICKER_CODES = CubiesCube([
    Cubie(100, 101, 102, 1),
    Cubie(103, 104, 105, 2),
    Cubie(106, 107, 108, 3),
    Cubie(109, 110, 111, 4),
    Cubie(112, 113, 114, 5),
    Cubie(115, 116, 117, 6),
    Cubie(118, 119, 120, 7),
    Cubie(121, 122, 123, 8),
])


def compute_stickers_bitwise_shifts_per_operation():
    bitwise_shifts = {}
    for name, op in OPERATIONS.items():
        permutation = compute_stickers_permutation(op(CUBE_WITH_UNIQUE_STICKER_CODES), CUBE_WITH_UNIQUE_STICKER_CODES)
        bitwise_shifts[name] = stringify_bitwise_shifts(compute_bitwise_shifts(permutation))
    return bitwise_shifts


def compute_any_cube_to_oriented_cube_bit_checks_and_shifts():
    stickers_patterns = generate_fixed_cubie_stickers_bit_patterns()
    empty_cubie = Cubie(0, 0, 0)

    odd = CubiesCube.CUBIES_WITH_ODD_ORIENTATION

    to_vector = lambda cube: StickerVectorSerializer().serialize(cube)

    x_face_stickers = [1, 2, 3, 4, 9, 10, 11, 12]
    y_face_stickers = [5, 6, 7, 8, 13, 14, 15, 16]
    z_face_stickers = [17, 18, 19, 20, 21, 22, 23, 24]

    checks_and_shifts = {}
    for i in range(CubiesCube.NB_CUBIES):
        candidate_nb = i + 1
        mask_cube = CubiesCube(
            [empty_cubie] * i +
            [Cubie(7, 7, 7)] +  # 7 is binary 111
            [empty_cubie] * (CubiesCube.NB_CUBIES - i - 1)
        )

        stickers_order = x_face_stickers + y_face_stickers + z_face_stickers
        mask_vector = to_vector(mask_cube)
        xyz_stickers_positions = [sticker_nb - 1 for sticker_nb in stickers_order if mask_vector[sticker_nb - 1] == 7]
        state_shifts = compute_bitwise_shifts_for_specific_stickers({
            _from: _to for _from, _to in zip(xyz_stickers_positions, range(21, 24))
        })

        possible_colors_patterns = stickers_patterns['odd'] if candidate_nb in odd else stickers_patterns['even']
        checks_and_shifts[candidate_nb] = {"bitwise_color_detection_shifts": state_shifts,
                                           "possible_colors_patterns": possible_colors_patterns,}
    return checks_and_shifts


def compute_all_cube_rotations(initial_state: CubiesCube = None) -> List[CubiesCube]:
    if initial_state is None:
        initial_state = CubiesCube()

    def apply_all_rotation_moves(state):
        return [move(state) for move in CUBE_ROTATION_MOVES]

    lookup = [initial_state]
    for i in range(3):
        lookup += flatten([apply_all_rotation_moves(state) for state in lookup])

    all_rotations = {}
    for candidate in lookup:
        permutation = compute_stickers_permutation(candidate, initial_state)
        all_rotations["_".join(map(str, permutation))] = candidate
    assert len(all_rotations) == 24, "Must produce all 24 rotations"
    return list(all_rotations.values())


def generate_fixed_cubie_stickers_bit_patterns():
    colors_sequences = {
        "even": [(6, 4, 1), (4, 1, 6), (1, 6, 4)],
        "odd": [(1, 4, 6), (4, 6, 1), (6, 1, 4)],
    }

    bit_patterns = {}
    for variant, stickers_pattern in colors_sequences.items():
        bit_patterns[variant] = []
        for x, y, z in stickers_pattern:
            bit_patterns[variant].append(
                (x << StickerBinarySerializer.COLOR_BITS * 2) | \
                (y << StickerBinarySerializer.COLOR_BITS * 1) | \
                z
            )
    return bit_patterns


def find_permutations_to_orient_the_cube():
    checks_and_shifts = compute_any_cube_to_oriented_cube_bit_checks_and_shifts()
    possible_rotations = compute_all_cube_rotations()
    possible_rotations_unique = compute_all_cube_rotations(CUBE_WITH_UNIQUE_STICKER_CODES)
    results_by_cubie = defaultdict(lambda: [])
    for i, state in enumerate(possible_rotations):
        numeric_state = state.as_stickers_int
        for cubie, details in checks_and_shifts.items():
            actual_color_pattern = apply_bitwise_shifts(details['bitwise_color_detection_shifts'], numeric_state)
            for possible_color_pattern in details["possible_colors_patterns"]:
                if actual_color_pattern == possible_color_pattern:
                    orienting_permutation = compute_stickers_permutation(
                        CUBE_WITH_UNIQUE_STICKER_CODES,
                        possible_rotations_unique[i]
                    )
                    results_by_cubie[cubie].append({
                        "cubie": cubie,
                        "color_pattern": actual_color_pattern,
                        "bitwise_color_detection_shifts": details['bitwise_color_detection_shifts'],
                        "bitwise_orienting_permutation": compute_bitwise_shifts(orienting_permutation)
                    })
    return dict(results_by_cubie)


MODE_STICKERS = 'stickers'
MODE_CUBIES = 'cubies'
MODE = MODE_STICKERS

if MODE is MODE_CUBIES:
    BINARY_SERIALIZER = CubieVectorBinarySerializer()
elif MODE is MODE_STICKERS:
    BINARY_SERIALIZER = StickerBinarySerializer()

INT_SERIALIZER = IntSerializer(BINARY_SERIALIZER)

SOLVED_CUBE = CubiesCube()
SOLVED_CUBE_BINARY = BINARY_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_INT = INT_SERIALIZER.serialize(SOLVED_CUBE)
SOLVED_CUBE_HEX = hex(SOLVED_CUBE_INT)

BITWISE_SHIFTS = compute_stickers_bitwise_shifts_per_operation()


def print_rust_code():
    print(SOLVED_CUBE_BINARY)
    print(SOLVED_CUBE_HEX)
    print(f"""
fn main() {{
	let solved_state:i64 = {SOLVED_CUBE_HEX};
}}
""")
    if MODE is MODE_STICKERS:
        print_rust_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation())
        print_rust_orienting_permutation(find_permutations_to_orient_the_cube())
    elif MODE is MODE_CUBIES:
        pass
        # @TODO maybe


def print_python_code():
    if MODE is MODE_STICKERS:
        serializer_name = 'StickerBinarySerializer'
        print_python_bitwise_shifts(compute_stickers_bitwise_shifts_per_operation())
        print_python_orienting_permutation(find_permutations_to_orient_the_cube())
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
    return {shifts};
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
    return {shifts}
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


if __name__ == '__main__':
    # print_python_code()
    print_rust_code()
