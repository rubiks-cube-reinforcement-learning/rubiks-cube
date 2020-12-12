from typing import List

from cube2.cube import Cube2
from utils import CubeSerializer, Cubie, normalize_binary_string, partition


class CubieVectorSerializer(CubeSerializer):

    def __init__(self, reference_cube=None) -> None:
        if reference_cube is None:
            reference_cube = Cube2()
        self.reference_cube = reference_cube
        super().__init__()

    def serialize(self, shuffled_cube: Cube2):
        reference_cube = self.reference_cube
        cubies_indexes = []
        x_face_indexes = []
        for i in range(1, 9):
            shuffled_cubie = shuffled_cube.cubies[i]
            reference_cubie = [c for c in reference_cube.cubies if c.idx == shuffled_cubie.idx][0]
            cubies_indexes += [shuffled_cubie.idx]
            x_face_indexes += [reference_cubie.faces.index(shuffled_cubie.face_x)]
        return cubies_indexes + x_face_indexes

    def unserialize(self, vector) -> Cube2:
        reference_cube = self.reference_cube
        unserialized_cube = Cube2()
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
        even = Cube2.CUBIES_WITH_EVEN_ORIENTATION
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

    def serialize(self, cube: Cube2):
        vector = CubieVectorSerializer().serialize(cube)
        binary_string = ["0"] * self.OFFSET
        binary_string += ["{0:03b}".format(n - 1) for n in vector[:8]]
        binary_string += ["{0:02b}".format(n) for n in vector[8:]]
        return "".join(binary_string)

    def unserialize(self, binary_string: str) -> Cube2:
        binary_string = normalize_binary_string(binary_string, self.DATA_LENGTH)
        binary_idxes = partition(binary_string[:self.IDXES_LENGTH], self.IDX_BITS)
        binary_rotations = partition(binary_string[self.IDXES_LENGTH:], self.ROTATION_BITS)

        idxes = [int(binary_idx, 2) + 1 for binary_idx in binary_idxes]
        rotations = [int(binary_rotation, 2) for binary_rotation in binary_rotations]
        return CubieVectorSerializer().unserialize(idxes + rotations)

