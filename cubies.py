# coding=utf-8
from collections import defaultdict

from loggers import getLogger

logger = getLogger(__name__)

AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3

class Cubie:
    def __init__(self, face_x, face_y, face_z):
        self.face_x = face_x
        self.face_y = face_y
        self.face_z = face_z

    def __repr__(self):
        return "Cubie (%s, %s, %s)" % (self.face_x, self.face_y, self.face_z)

    def __str__(self):
        return (self.face_x, self.face_y, self.face_z)

    def get_face(self, axis):
        if axis == AXIS_X:
            return self.face_x
        elif axis == AXIS_Y:
            return self.face_y
        elif axis == AXIS_Z:
            return self.face_z

    def clone(self):
        return Cubie(self.face_x, self.face_y, self.face_z)


class Operation:
    def __init__(self, mapping, axis):
        self.mapping = mapping
        self.axis = axis

    def reverse(self):
        return Operation({v: k for k, v in self.mapping.items()}, self.axis)

    def compute_permutation(self):
        reference_cube = CubiesCube([
            Cubie(100, 101, 102),
            Cubie(103, 104, 105),
            Cubie(106, 107, 108),
            Cubie(109, 110, 111),
            Cubie(112, 113, 114),
            Cubie(115, 116, 117),
            Cubie(118, 119, 120),
            Cubie(121, 122, 123),
        ])
        transformed_cube = reference_cube.apply_operation(self)

        source_vector = reference_cube.to_vector()
        target_vector = transformed_cube.to_vector()
        return [source_vector.index(i) for i in target_vector]


class CubiesCube:
    LU = Operation({1: 5, 5: 7, 7: 3, 3: 1}, AXIS_Z)
    LD = LU.reverse()
    RU = Operation({2: 6, 6: 8, 8: 4, 4: 2}, AXIS_Z)
    RD = RU.reverse()

    FL = Operation({1: 2, 2: 3, 3: 4, 4: 1}, AXIS_X)
    FR = FL.reverse()
    BL = Operation({5: 6, 6: 7, 7: 8, 8: 5}, AXIS_X)
    BR = BL.reverse()

    UL = Operation({1: 5, 5: 6, 6: 2, 2: 1}, AXIS_Y)
    UR = UL.reverse()
    DL = Operation({3: 7, 7: 8, 8: 4, 4: 3}, AXIS_Y)
    DR = DL.reverse()

    OPERATIONS = {'LU': LU, 'LD': LD, 'RU': RU, 'RD': RD, 'FL': FL, 'FR': FR, 'BL': BL, 'BR': BR, 'UL': UL, 'UR': UR,
                  'DL': DL, 'DR': DR}

    def __init__(self, cubies):
        self.cubies = [Cubie(0, 0, 0)] + cubies

    def __str__(self):
        faces = [self.face(i) for i in range(1, 7)]
        return faces.__str__()

    def clone(self):
        return CubiesCube([c.clone() for c in self.cubies[1:]])

    def to_vector(self):
        vector = []
        for i in range(1, 7):
            vector += self.face(i)
        return vector

    def face(self, nb):
        if nb in [1, 3]:
            axis = AXIS_X
        elif nb in [2, 4]:
            axis = AXIS_Y
        elif nb in [5, 6]:
            axis = AXIS_Z

        cubies = {
            1: [1, 2, 3, 4],
            2: [5, 6, 1, 2],
            3: [5, 6, 7, 8],
            4: [7, 8, 3, 4],
            5: [5, 1, 7, 3],
            6: [2, 6, 4, 8],
        }

        colors = [self.cubies[idx].get_face(axis) for idx in cubies[nb]]
        return colors

    def apply_operation(self, operation):
        new_cube = self.clone()
        for k, v in operation.mapping.items():
            cubie = new_cube.cubies[v] = self.cubies[k].clone()
            if operation.axis == AXIS_X:
                cubie.face_y, cubie.face_z = cubie.face_z, cubie.face_y
            elif operation.axis == AXIS_Y:
                cubie.face_x, cubie.face_z = cubie.face_z, cubie.face_x
            elif operation.axis == AXIS_Z:
                cubie.face_x, cubie.face_y = cubie.face_y, cubie.face_x
        return new_cube

PERMUTATIONS = {}
for name, op in CubiesCube.OPERATIONS.items():
    PERMUTATIONS[name] = op.compute_permutation()

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

BITS = 128
OCCUPIED_BITS = 72
COLOR_SIZE = 3
OFFSET = BITS - OCCUPIED_BITS

BITWISE_SHIFTS = defaultdict(lambda: [])
for name, permutation in PERMUTATIONS.items():
    reverse_shifts = defaultdict(lambda: 0)
    for old_idx, new_idx in enumerate(permutation):
        bitwise_mask = ['0'] * BITS
        color_offset = OFFSET + new_idx * COLOR_SIZE
        for color_bit_offset in range(COLOR_SIZE):
            bitwise_mask[color_offset + color_bit_offset] = '1'
        bitwise_mask_str = ''.join(bitwise_mask)
        bitwise_offset = (new_idx - old_idx) * COLOR_SIZE
        hex_mask = int(bitwise_mask_str, 2)

        direction = '>>' if bitwise_offset < 0 else '<<'
        reverse_shifts[bitwise_offset] |= hex_mask

    for offset, mask in reverse_shifts.items():
        direction = '>>' if offset < 0 else '<<'
        BITWISE_SHIFTS[name].append(f'((x & {hex(mask)}) {direction} {abs(offset)})')

SOLVED_CUBE = CubiesCube([
    Cubie(1, 2, 5),
    Cubie(1, 2, 6),
    Cubie(1, 4, 5),
    Cubie(1, 4, 6),
    Cubie(3, 2, 5),
    Cubie(3, 2, 6),
    Cubie(3, 4, 5),
    Cubie(3, 4, 6),
])

SOLVED_CUBE_BINARY = "".join(["0"] * OFFSET + ["{0:03b}".format(n) for n in SOLVED_CUBE.to_vector()])
SOLVED_CUBE_INT = int(SOLVED_CUBE_BINARY, 2)
SOLVED_CUBE_HEX = hex(SOLVED_CUBE_INT)

def print_rust_code():
    print(SOLVED_CUBE_BINARY)
    print(SOLVED_CUBE_HEX)
    print(f"""
fn main() {{
	let solved_state:i128 = {SOLVED_CUBE_HEX};
}}
""")
    for name, shifts in BITWISE_SHIFTS.items():
        rust_shifts = "|".join(shifts)
        print(f"""
fn {name.lower()}(x: i128) -> i128 {{
    return {rust_shifts};
}}""")

if __name__ == '__main__':
    print_rust_code()
