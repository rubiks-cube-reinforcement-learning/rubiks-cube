# coding=utf-8
from collections import defaultdict, Counter
from random import shuffle
from typing import List
from typing import Type

from loggers import getLogger

logger = getLogger(__name__)

AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3


class Cubie:
    def __init__(self, face_x, face_y, face_z, idx):
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

class CubiesCube:
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

    def clone(self):
        return CubiesCube([c.clone() for c in self.cubies[1:]])

    def face(self, nb):
        axis = self.FACE_TO_AXIS[nb]
        colors = [self.cubies[idx].get_face(axis) for idx in self.FACE_CUBIES_IDX[nb]]
        return colors

    def assert_valid(self):
        # @todo This does not account for accidentally switching two stickers on the same cubie
        vector = StickerVectorSerializer().serialize(self)
        assert len(vector) == 24, "Length of a vector representation should be 24"

        cnt = Counter(vector)
        print(vector)
        assert len(cnt) == 6, "There should be 6 different colors on the cube"

        cnt_values = list(set(cnt.values()))
        assert cnt_values[0] == 4, "There should be 4 occurences of each color"

        for cubie in self.cubies[1:]:
            colors = list(set([cubie.face_x, cubie.face_y, cubie.face_z]))
            assert len(colors) == 3, "Each cubie should have 3 different faces"


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
        changed_orientation = (ref_idx in [2, 3, 5, 8]) ^ (new_idx in [2, 3, 5, 8])
        if changed_orientation:
            seq = seq[::-1]

        new_x = ref_cubie.faces[x_face_idx]
        seq_x_idx = seq.index(new_x)
        x, y, z = seq[seq_x_idx], seq[seq_x_idx-2], seq[seq_x_idx-1]

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
        binary_string += ["{0:03b}".format(n-1) for n in vector[:8]]
        binary_string += ["{0:02b}".format(n) for n in vector[8:]]
        return "".join(binary_string)

    def unserialize(self, binary_string: str) -> CubiesCube:
        binary_string = normalize_binary_string(binary_string, self.DATA_LENGTH)
        binary_idxes = partition(binary_string[:self.IDXES_LENGTH], self.IDX_BITS)
        binary_rotations = partition(binary_string[self.IDXES_LENGTH:], self.ROTATION_BITS)

        idxes = [int(binary_idx, 2) + 1 for binary_idx in binary_idxes]
        rotations = [int(binary_rotation, 2) for binary_rotation in binary_rotations]
        return CubieVectorSerializer().unserialize(idxes + rotations)

def partition(_list, bucket_size):
    return [_list[i:i + bucket_size] for i in range(0, len(_list), bucket_size)]

def normalize_binary_string(binary_string, expected_length):
    return (('0' * (expected_length - len(binary_string))) + binary_string)[-expected_length:]

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


def compute_permutation(operation: Operation, serializer: CubeSerializer, reference_cube:CubiesCube=None):
    if reference_cube is None:
        reference_cube = BR(LU(CubiesCube()))
    transformed_cube = operation(reference_cube)

    source_vector = serializer.serialize(reference_cube)
    target_vector = serializer.serialize(transformed_cube)
    return [source_vector.index(i) for i in target_vector], source_vector, target_vector

def compute_bitwise_shifts(permutation_vector, offset, int_size, bucket_size):
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in enumerate(permutation_vector):
        bitwise_mask = ['0'] * int_size
        color_offset = offset + new_idx * bucket_size
        for color_bit_offset in range(bucket_size):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (new_idx - old_idx) * bucket_size
        hex_mask = int(bitwise_mask_str, 2)

        reverse_shifts[bitwise_offset] |= hex_mask

    shifts = []
    for offset, mask in reverse_shifts.items():
        direction = '>>' if offset < 0 else '<<'
        shifts.append(f'((x & {hex(mask)}) {direction} {abs(offset)})')
    return shifts

def compute_stickers_bitwise_shifts():
    unique_stickers_cube = CubiesCube([
        Cubie(100, 101, 102, 1),
        Cubie(103, 104, 105, 2),
        Cubie(106, 107, 108, 3),
        Cubie(109, 110, 111, 4),
        Cubie(112, 113, 114, 5),
        Cubie(115, 116, 117, 6),
        Cubie(118, 119, 120, 7),
        Cubie(121, 122, 123, 8),
    ])

    bitwise_shifts = {}
    for name, op in OPERATIONS.items():
        permutations, _, _ = compute_permutation(op, StickerVectorSerializer(), unique_stickers_cube)
        bitwise_shifts[name] = compute_bitwise_shifts(
            permutations,
            StickerBinarySerializer.OFFSET,
            StickerBinarySerializer.INT_LENGTH,
            StickerBinarySerializer.COLOR_BITS,
        )
    return bitwise_shifts

def compute_cubies_bitwise_shifts():
    bitwise_shifts = {}
    for name, op in list(OPERATIONS.items())[0:]:
        permutations, before, after = compute_permutation(op, CubieVectorSerializer())[:8]
        shifts = compute_bitwise_shifts(
            permutations,
            CubieVectorBinarySerializer.OFFSET,
            CubieVectorBinarySerializer.INT_LENGTH,
            CubieVectorBinarySerializer.IDX_BITS,
        )

        for i, (rot_before, rot_after) in enumerate(zip(before[8:], after[8:])):
            if rot_before != rot_after:
                diff = int(rot_after) - int(rot_before)

                # bitwise_mask = ['0'] * CubieVectorBinarySerializer.INT_LENGTH
                # shifts.append(f'((x & {hex(mask)}) )')

                shifts += [
                    diff
                ]

        print(before[8:])
        print(after[8:])
        print(shifts)
        exit(0)
        bitwise_shifts[name] = shifts
    return bitwise_shifts

def print_rust_bitwise_shifts(bitwise_shifts):
    int_type = 'i128' if MODE is MODE_STICKERS else 'i64'
    for name, shifts in bitwise_shifts.items():
        rust_shifts = "|".join(shifts)
        print(f"""
fn {name.lower()}(x: {int_type}) -> {int_type} {{
    return {rust_shifts};
}}""")

def print_python_bitwise_shifts(bitwise_shifts):
    ops = list(bitwise_shifts.items())
    for name, shifts in ops:
        rust_shifts = "|".join(shifts)
        print(f"""
def {name.lower()}(x):
    return {rust_shifts}
""")


def sequences(_list):
    prev = _list[0]
    seq = []
    last_i = len(_list) - 2
    for i, item in enumerate(_list[1:]):
        seq.append(prev)
        if i == last_i:
            seq.append(item)
            yield seq
        elif item != prev + 1:
            yield seq
            seq = []
        prev = item

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

BITWISE_SHIFTS = compute_stickers_bitwise_shifts()

def print_rust_code():
    print(SOLVED_CUBE_BINARY)
    print(SOLVED_CUBE_HEX)
    print(f"""
fn main() {{
	let solved_state:i64 = {SOLVED_CUBE_HEX};
}}
""")
    if MODE is MODE_STICKERS:
        print_rust_bitwise_shifts(compute_stickers_bitwise_shifts())
    elif MODE is MODE_CUBIES:
        print_rust_bitwise_shifts(compute_cubies_bitwise_shifts())


def print_python_code():
    if MODE is MODE_STICKERS:
        serializer_name = 'StickerBinarySerializer'
        print_python_bitwise_shifts(compute_stickers_bitwise_shifts())
    elif MODE is MODE_CUBIES:
        serializer_name = 'CubieVectorBinarySerializer'
        print_python_bitwise_shifts(compute_cubies_bitwise_shifts())

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
    print(ru(lu(ld(SOLVED_CUBE_INT))))
    # for op in OPS:
    #     shuffle(OPS)
    #     for i in range(10):
    #         current = op(current)
    #     c = IntSerializer({serializer_name}()).unserialize(current)
    #     is_valid = c.assert_valid()
    print(current)

main()
""")


if __name__ == '__main__':
    # print_python_code()
    print_rust_code()
