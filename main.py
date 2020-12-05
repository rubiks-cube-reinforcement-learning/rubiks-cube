# coding=utf-8

import numpy as np
from itertools import permutations, chain
from moves import lu, ld, ru, rd, ul, ur, dl, dr, fl, fr, bl, br, hash_state

cubies_colors = {
    1: [1, 2, 3],
    2: [1, 2, 4],
    3: [1, 3, 5],
    4: [1, 4, 5],
    5: [6, 2, 3],
    6: [6, 2, 4],
    7: [6, 3, 5],
    8: [6, 4, 5],
}
cubies = list(cubies_colors.keys())
e = list(chain(*list(cubies_colors.values())))


def rotate(stickers, rotation):
    return stickers[rotation:] + stickers[:rotation]


def possible_2x2x2_states():
    """
    See this Quora answer:
    https://www.quora.com/How-can-we-calculate-the-number-of-permutations-of-a-2x2x2-Rubik’s-cube-Can-you-clearly-explain-how

    :return:
    """
    for permutation in permutations(cubies):
        for i in range(3 ** 6):
            rotations = [int(c) for c in np.base_repr(i, base=3).zfill(8)]
            rotated = [rotate(cubies_colors[p], r) for p, r in zip(permutation, rotations)]
            yield [
                rotated[0][0], rotated[1][0],
                rotated[2][0], rotated[3][0],

                rotated[4][1], rotated[4][1],
                rotated[0][1], rotated[1][1],

                rotated[6][0], rotated[7][0],
                rotated[4][0], rotated[5][0],

                rotated[2][2], rotated[3][2],
                rotated[6][2], rotated[7][2],

                rotated[0][2], rotated[2][1],
                rotated[4][2], rotated[6][1],

                rotated[1][2], rotated[3][1],
                rotated[5][2], rotated[7][1],
            ]


def cache_2x2x2_states():
    with open("./2x2x2-states.txt", "w+") as fp:
        for state in possible_2x2x2_states():
            fp.write("".join([hash_state(s) for s in state]) + "\n")


def load_2x2x2_states():
    states = set()
    with open("./2x2x2-states.txt", "r+") as fp:
        for line in fp:
            states.add(line)
    return states

if __name__ == '__main__':

    solved_state = [1, 1, 1, 1, 2, 2, 2, 2, 6, 6, 6, 6, 5, 5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4]
    moves = [lu, ld, ru, rd, ul, ur, dl, dr, fl, fr]
    moves_table = {
        hash_state(solved_state): 0
    }

    def explore(state, cost, levels):
        if levels == 0:
            return

        for move in moves:
            next_state = hash_state(move(state))
            if next_state not in moves_table:
                moves_table[next_state] = cost + 1

        for move in moves:
            next_state = move(state)
            next_state_str = hash_state(next_state)
            explore(next_state, moves_table[next_state_str], levels - 1)

    explore(solved_state, 0, 10)

    import pickle
    pickle.dump(moves_table, open("moves_table.p", "wb"))

    def solve(state):
        distance = moves_table[hash_state(state)]
        it = 0
        while True:
            for move in moves:
                next_state = move(state)
                state_distance = moves_table[hash_state(next_state)]
                if state_distance < distance:
                    state = next_state
                    distance = moves_table[hash_state(state)]
                    break
            it += 1
            if it > 30:
                print("30 moves :D")
                break
            if hash_state(state) == hash_state(solved_state):
                break
        return state

    scrambled_state = ul(fl(ru(ld(ur(solved_state)))))
    print(hash_state(solve(scrambled_state)))

    # states = load_2x2x2_states()
