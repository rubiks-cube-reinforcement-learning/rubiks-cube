# coding=utf-8

def print_state(state):
    print("""
      {18} {16}
      {19} {17}
{2} {0} {6} {4} {10} {8} {14} {12}
{3} {1} {7} {5} {11} {9} {15} {13}
      {22} {20}
      {23} {21}
    """.format(*state))

def ur(old_state):
    return get_indices(old_state, [12, 1, 14, 3, 0, 5, 2, 7, 4, 9, 6, 11, 8, 13, 10, 15, 17, 19, 16, 18, 20, 21, 22, 23])

def ul(old_state):
    return get_indices(old_state, [4, 1, 6, 3, 8, 5, 10, 7, 12, 9, 14, 11, 0, 13, 2, 15, 18, 16, 19, 17, 20, 21, 22, 23])

def dr(old_state):
    return get_indices(old_state, [0, 13, 2, 15, 4, 1, 6, 3, 8, 5, 10, 7, 12, 9, 14, 11, 16, 17, 18, 19, 22, 20, 23, 21])

def dl(old_state):
    return get_indices(old_state, [0, 5, 2, 7, 4, 9, 6, 11, 8, 13, 10, 15, 12, 1, 14, 3, 16, 17, 18, 19, 21, 23, 20, 22])

def ld(old_state):
    return get_indices(old_state, [1, 3, 0, 2, 4, 5, 22, 23, 8, 9, 10, 11, 19, 18, 14, 15, 16, 17, 6, 7, 20, 21, 13, 12])

def lu(old_state):
    return get_indices(old_state, [2, 0, 3, 1, 4, 5, 18, 19, 8, 9, 10, 11, 23, 22, 14, 15, 16, 17, 13, 12, 20, 21, 6, 7])

def ru(old_state):
    return get_indices(old_state, [0, 1, 2, 3, 20, 21, 6, 7, 10, 8, 11, 9, 12, 13, 17, 16, 4, 5, 18, 19, 15, 14, 22, 23])

def rd(old_state):
    return get_indices(old_state, [0, 1, 2, 3, 16, 17, 6, 7, 9, 11, 8, 10, 12, 13, 21, 20, 15, 14, 18, 19, 4, 5, 22, 23])

def br(old_state):
    return get_indices(old_state, [0, 1, 23, 21, 4, 5, 6, 7, 18, 16, 10, 11, 14, 12, 15, 13, 2, 17, 3, 19, 20, 8, 22, 9])

def bl(old_state):
    return get_indices(old_state, [0, 1, 16, 18, 4, 5, 6, 7, 21, 23, 10, 11, 13, 15, 12, 14, 9, 17, 8, 19, 20, 3, 22, 2])

def fr(old_state):
    return get_indices(old_state, [22, 20, 2, 3, 6, 4, 7, 5, 8, 9, 19, 17, 12, 13, 14, 15, 16, 0, 18, 1, 10, 21, 11, 23])

def fl(old_state):
    return get_indices(old_state, [17, 19, 2, 3, 5, 7, 4, 6, 8, 9, 20, 22, 12, 13, 14, 15, 16, 11, 18, 10, 1, 21, 0, 23])

def get_indices(state, indices):
    return [state[index] for index in indices]

def hash_state(state):
    return ",".join(["%d" % s for s in state])

if __name__ == '__main__':
    pass
