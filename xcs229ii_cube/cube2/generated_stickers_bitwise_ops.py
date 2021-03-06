'''
This file was auto-generated, do not change manually
'''

def cube3_to_cube2(x):
    return ((x & 0x38000000000000000000000000000000000000000) >> 90)|((x & 0xe00000000000000000000000000000000000000) >> 87)|((x & 0xe00000000000000000000000000000000000) >> 78)|((x & 0x3f000000000000000000000000000000000) >> 75)|((x & 0x1c0000000000000000000000000000000) >> 72)|((x & 0x1c0000000000000000000000000000) >> 63)|((x & 0x7e00000000000000000000000000) >> 60)|((x & 0x38000000000000000000000000) >> 57)|((x & 0x38000000000000000000000) >> 48)|((x & 0xfc0000000000000000000) >> 45)|((x & 0x7000000000000000000) >> 42)|((x & 0x7000000000000000) >> 33)|((x & 0x1f8000000000000) >> 30)|((x & 0xe00000000000) >> 27)|((x & 0xe00000000) >> 18)|((x & 0x3f000000) >> 15)|((x & 0x1c0000) >> 12)|((x & 0x1c0) >> 3)|((x & 0x7) << 0)



def orient_cube(x:int):
    # Cubie 4
    actual_color_pattern = ((x & 0x7000000000000000) >> 54)|((x & 0x7000000) >> 21)|((x & 0x38) >> 3)
    if actual_color_pattern == 102:
        return ((x & 0xffffffffffffffffff) << 0)
    if actual_color_pattern == 305:
        return ((x & 0x1c0) << 63)|((x & 0x7) << 66)|((x & 0xe00) << 54)|((x & 0x38) << 57)|((x & 0xe07000000000) << 12)|((x & 0x38000000000) << 15)|((x & 0x1c0000000000) << 9)|((x & 0xe07000) << 24)|((x & 0x38000) << 27)|((x & 0x1c0000) << 21)|((x & 0xe07fff000000000000) >> 36)|((x & 0x38000000000000000) >> 33)|((x & 0x1c0000000000000000) >> 39)|((x & 0x1c7000000) >> 21)|((x & 0xe38000000) >> 27)
    if actual_color_pattern == 396:
        return ((x & 0xe07fff000) << 36)|((x & 0x38000000) << 39)|((x & 0x1c0000000) << 33)|((x & 0xe07000000000000) >> 12)|((x & 0x38000000000000) >> 9)|((x & 0x1c0000000000000) >> 15)|((x & 0x1c7) << 27)|((x & 0xe38) << 21)|((x & 0xe07000000000) >> 24)|((x & 0x38000000000) >> 21)|((x & 0x1c0000000000) >> 27)|((x & 0x38000000000000000) >> 54)|((x & 0xe00000000000000000) >> 63)|((x & 0x7000000000000000) >> 57)|((x & 0x1c0000000000000000) >> 66)
    # Cubie 2
    actual_color_pattern = ((x & 0x1c0000000000000000) >> 60)|((x & 0x7000000000000) >> 45)|((x & 0xe00) >> 9)
    if actual_color_pattern == 270:
        return ((x & 0xfff000fff000000) << 12)|((x & 0x3f000000000) << 18)|((x & 0xfc0000038007) << 6)|((x & 0x3f000000000000000) >> 30)|((x & 0xfc0000000000000000) >> 42)|((x & 0xe00038) >> 3)|((x & 0x71c0) << 3)|((x & 0x1c0e00) >> 6)
    if actual_color_pattern == 116:
        return ((x & 0x38000038000000000) << 6)|((x & 0xe00000e00000000000) >> 3)|((x & 0x7000007000000000) << 3)|((x & 0x1c00001c0000000000) >> 6)|((x & 0x38000) << 42)|((x & 0xe00007) << 33)|((x & 0x7000) << 39)|((x & 0x1c0000) << 30)|((x & 0x1f8) << 24)|((x & 0xe00) << 15)|((x & 0xe07000000) >> 12)|((x & 0x38000000) >> 9)|((x & 0x1c0000000) >> 15)|((x & 0x38000000000000) >> 42)|((x & 0xe00000000000000) >> 51)|((x & 0x7000000000000) >> 45)|((x & 0x1c0000000000000) >> 54)
    if actual_color_pattern == 417:
        return ((x & 0x7) << 69)|((x & 0x38) << 63)|((x & 0x1c0) << 57)|((x & 0xe00) << 51)|((x & 0xe07000000) << 24)|((x & 0x38000000) << 27)|((x & 0x1c0000000) << 21)|((x & 0x3f000) << 30)|((x & 0xfc0000) << 18)|((x & 0xe07000000000000) >> 24)|((x & 0x38000000000000) >> 21)|((x & 0x1c0000000000000) >> 27)|((x & 0x3f000000000) >> 18)|((x & 0xfc0000000000) >> 30)|((x & 0x7000000000000000) >> 51)|((x & 0x38000000000000000) >> 57)|((x & 0x1c0000000000000000) >> 63)|((x & 0xe00000000000000000) >> 69)
    # Cubie 8
    actual_color_pattern = ((x & 0x7000000000) >> 30)|((x & 0x1c0000000) >> 27)|((x & 0x7) << 0)
    if actual_color_pattern == 270:
        return ((x & 0x3f000000) << 42)|((x & 0xfc0000000) << 30)|((x & 0xfff000fff000000000) >> 12)|((x & 0x3f000000e001c0) >> 6)|((x & 0xfc0000000000000) >> 18)|((x & 0x1c0007) << 3)|((x & 0x7038) << 6)|((x & 0x38e00) >> 3)
    if actual_color_pattern == 417:
        return ((x & 0xfff) << 60)|((x & 0x38000038000000) << 6)|((x & 0xe00000e00000000) >> 3)|((x & 0x7000007000000) << 3)|((x & 0x1c00001c0000000) >> 6)|((x & 0x1c7000) << 27)|((x & 0xe38000) << 21)|((x & 0xfff000000000000000) >> 48)|((x & 0x1c7000000000) >> 33)|((x & 0xe38000000000) >> 39)
    if actual_color_pattern == 116:
        return ((x & 0xe07000000e07) << 24)|((x & 0x38000000038) << 27)|((x & 0x1c00000001c0) << 21)|((x & 0x1c0000) << 39)|((x & 0x7000) << 42)|((x & 0xe00000) << 30)|((x & 0x38000) << 33)|((x & 0xe07000000e07000000) >> 24)|((x & 0x38000000038000000) >> 21)|((x & 0x1c00000001c0000000) >> 27)|((x & 0x38000000000000) >> 30)|((x & 0xe00000000000000) >> 39)|((x & 0x7000000000000) >> 33)|((x & 0x1c0000000000000) >> 42)
    # Cubie 3
    actual_color_pattern = ((x & 0x38000000000000000) >> 57)|((x & 0x38000000) >> 24)|((x & 0x7000) >> 12)
    if actual_color_pattern == 116:
        return ((x & 0x1c00001c0000000000) << 3)|((x & 0x7000007000000000) << 6)|((x & 0xe00000e00000000000) >> 6)|((x & 0x38000038000000000) >> 3)|((x & 0x1c0) << 51)|((x & 0x7) << 54)|((x & 0xe00) << 42)|((x & 0x38) << 45)|((x & 0xe07000) << 12)|((x & 0x38000) << 15)|((x & 0x1c0000) << 9)|((x & 0x1c0000e00000000) >> 33)|((x & 0x7000000000000) >> 30)|((x & 0xe00000000000000) >> 42)|((x & 0x38000000000000) >> 39)|((x & 0x7000000) >> 15)|((x & 0x1f8000000) >> 24)
    if actual_color_pattern == 417:
        return ((x & 0xfff000) << 48)|((x & 0x1c00001c0000000) << 3)|((x & 0x7000007000000) << 6)|((x & 0xe00000e00000000) >> 6)|((x & 0x38000038000000) >> 3)|((x & 0x1c7) << 39)|((x & 0xe38) << 33)|((x & 0x1c7000000000) >> 21)|((x & 0xe38000000000) >> 27)|((x & 0xfff000000000000000) >> 60)
    if actual_color_pattern == 270:
        return ((x & 0x1c7000000) << 39)|((x & 0xe38000000) << 33)|((x & 0x1c70000001c0) << 15)|((x & 0xe38000000038) << 9)|((x & 0x1c7000000007000) >> 9)|((x & 0xe38000000e00000) >> 15)|((x & 0x1c7000000000000000) >> 33)|((x & 0xe38000000000000000) >> 39)|((x & 0x7) << 18)|((x & 0xe00) << 6)|((x & 0x38000) >> 6)|((x & 0x1c0000) >> 18)
    # Cubie 6
    actual_color_pattern = ((x & 0x1c0000000000) >> 36)|((x & 0x1c0000000000000) >> 51)|((x & 0x1c0) >> 6)
    if actual_color_pattern == 102:
        return ((x & 0x3f03f000000) << 30)|((x & 0xfc0fc0000000) << 18)|((x & 0x3f03f000000000000) >> 18)|((x & 0xfc0fc0000000000000) >> 30)|((x & 0x7007) << 9)|((x & 0x38038) << 3)|((x & 0x1c01c0) >> 3)|((x & 0xe00e00) >> 9)
    if actual_color_pattern == 396:
        return ((x & 0x38000038000fc0) << 18)|((x & 0xe00000e00000000) << 9)|((x & 0x7000007000000) << 15)|((x & 0x1c00001c0000000) << 6)|((x & 0x7000) << 45)|((x & 0x38000) << 39)|((x & 0x1c0000) << 33)|((x & 0xe00000) << 27)|((x & 0x3f) << 30)|((x & 0x38000000000000000) >> 42)|((x & 0xe00000000000000000) >> 51)|((x & 0x7000000000000000) >> 45)|((x & 0x1c0000000000000000) >> 54)|((x & 0xe07000000000) >> 36)|((x & 0x38000000000) >> 33)|((x & 0x1c0000000000) >> 39)
    if actual_color_pattern == 305:
        return ((x & 0x38) << 66)|((x & 0xe00) << 57)|((x & 0x7) << 63)|((x & 0x1c0) << 54)|((x & 0x3800003803f000000) >> 6)|((x & 0xe00000e00000000000) >> 15)|((x & 0x7000007000000000) >> 9)|((x & 0x1c00001c0fc0000000) >> 18)|((x & 0x7000) << 33)|((x & 0x1f8000) << 24)|((x & 0xe00000) << 15)|((x & 0x7000000000000) >> 39)|((x & 0x38000000000000) >> 45)|((x & 0x1c0000000000000) >> 51)|((x & 0xe00000000000000) >> 57)
    # Cubie 1
    actual_color_pattern = ((x & 0xe00000000000000000) >> 63)|((x & 0x38000000000000) >> 48)|((x & 0x1c0000) >> 18)
    if actual_color_pattern == 305:
        return ((x & 0x38000) << 54)|((x & 0xe00007) << 45)|((x & 0x7000) << 51)|((x & 0x1c0000) << 42)|((x & 0x7000000000) << 21)|((x & 0x1f8000000000) << 12)|((x & 0xe00000000000) << 3)|((x & 0x1f8) << 36)|((x & 0xe00) << 27)|((x & 0x7000000000000000) >> 27)|((x & 0x1f8000000000000000) >> 36)|((x & 0xe00000000000000000) >> 45)|((x & 0x1c7000000) >> 9)|((x & 0xe38000000) >> 15)|((x & 0xfff000000000000) >> 48)
    if actual_color_pattern == 396:
        return ((x & 0x1c00001c0000000) << 15)|((x & 0x700000703f000) << 18)|((x & 0xe00000e00fc0000) << 6)|((x & 0x38000038000000) << 9)|((x & 0x7) << 57)|((x & 0x38) << 51)|((x & 0x1c0) << 45)|((x & 0xe00) << 39)|((x & 0x7000000000) >> 15)|((x & 0x1f8000000000) >> 24)|((x & 0xe00000000000) >> 33)|((x & 0x1c0000000000000000) >> 57)|((x & 0x7000000000000000) >> 54)|((x & 0xe00000000000000000) >> 66)|((x & 0x38000000000000000) >> 63)
    if actual_color_pattern == 102:
        return ((x & 0x70000070000001c0) << 9)|((x & 0x38000038000000e00) << 3)|((x & 0x1c00001c0000007000) >> 3)|((x & 0xe00000e00000038000) >> 9)|((x & 0x1c7000000) << 27)|((x & 0xe38000007) << 21)|((x & 0x1c7000000e00000) >> 21)|((x & 0xe38000000000000) >> 27)|((x & 0x38) << 15)|((x & 0x1c0000) >> 15)
    # Cubie 7
    actual_color_pattern = ((x & 0x38000000000) >> 33)|((x & 0xe00000000) >> 30)|((x & 0x38000) >> 15)
    if actual_color_pattern == 305:
        return ((x & 0x1c0000) << 51)|((x & 0x7000) << 54)|((x & 0xe00000) << 42)|((x & 0x38000) << 45)|((x & 0x1c00001c0000000000) >> 9)|((x & 0x7000007000000000) >> 6)|((x & 0xe00000e0003f000000) >> 18)|((x & 0x38000038000000000) >> 15)|((x & 0xe07) << 36)|((x & 0x38) << 39)|((x & 0x1c0) << 33)|((x & 0x7000000000000) >> 27)|((x & 0x38000000000000) >> 33)|((x & 0x1c0000000000000) >> 39)|((x & 0xe00000000000000) >> 45)|((x & 0xfc0000000) >> 30)
    if actual_color_pattern == 396:
        return ((x & 0x7000000) << 45)|((x & 0x1f8000000) << 36)|((x & 0xe00000000) << 27)|((x & 0xfff) << 48)|((x & 0x7000000000000) >> 3)|((x & 0x1f8000000000000) >> 12)|((x & 0xe00000000000000) >> 21)|((x & 0x1c7000) << 15)|((x & 0xe38000) << 9)|((x & 0x1c0000e00000000000) >> 45)|((x & 0x7000000000000000) >> 42)|((x & 0xe00000000000000000) >> 54)|((x & 0x38000000000000000) >> 51)|((x & 0x7000000000) >> 27)|((x & 0x1f8000000000) >> 36)
    if actual_color_pattern == 102:
        return ((x & 0x1c7000000000) << 27)|((x & 0xe38000000000) << 21)|((x & 0x7000007000000) << 9)|((x & 0x38000038000000) << 3)|((x & 0x1c00001c0000000) >> 3)|((x & 0xe00000e00000000) >> 9)|((x & 0x1c7000000000000000) >> 21)|((x & 0xe38000000000000000) >> 27)|((x & 0xfff) << 12)|((x & 0xfff000) >> 12)
    # Cubie 5
    actual_color_pattern = ((x & 0xe00000000000) >> 39)|((x & 0xe00000000000000) >> 54)|((x & 0xe00000) >> 21)
    if actual_color_pattern == 116:
        return ((x & 0x7000000000) << 33)|((x & 0x1f8000000000) << 24)|((x & 0xe00000000000) << 15)|((x & 0x38) << 54)|((x & 0xe00) << 45)|((x & 0x7) << 51)|((x & 0x1c0) << 42)|((x & 0x7000000000000000) >> 15)|((x & 0x1f8000000000000000) >> 24)|((x & 0xe00000000000000000) >> 33)|((x & 0x7000) << 21)|((x & 0x1f8000) << 12)|((x & 0xe00000) << 3)|((x & 0x7000000) >> 3)|((x & 0x1f8000000) >> 12)|((x & 0xe00000000) >> 21)|((x & 0x1c0000000000000) >> 45)|((x & 0x7000000000000) >> 42)|((x & 0xe00000000000000) >> 54)|((x & 0x38000000000000) >> 51)
    if actual_color_pattern == 417:
        return ((x & 0x7000) << 57)|((x & 0x38000) << 51)|((x & 0x1c0000) << 45)|((x & 0xe00000) << 39)|((x & 0x7000000) << 33)|((x & 0x1f8000000) << 24)|((x & 0xe00000000) << 15)|((x & 0x3f) << 42)|((x & 0xfc0) << 30)|((x & 0x7000000000000) >> 15)|((x & 0x1f8000000000000) >> 24)|((x & 0xe00000000000000) >> 33)|((x & 0x7000000000000000) >> 39)|((x & 0x38000000000000000) >> 45)|((x & 0x1c0000000000000000) >> 51)|((x & 0xe00000000000000000) >> 57)|((x & 0x3f000000000) >> 30)|((x & 0xfc0000000000) >> 42)
    if actual_color_pattern == 270:
        return ((x & 0x7000007000000) << 21)|((x & 0x38000038000007) << 15)|((x & 0x1c00001c0000e00) << 9)|((x & 0xe00000e00000000) << 3)|((x & 0x7000007000000000) >> 3)|((x & 0x380000380001c0000) >> 9)|((x & 0x1c00001c0000038000) >> 15)|((x & 0xe00000e00000000000) >> 21)|((x & 0x38) << 18)|((x & 0x1c0) << 6)|((x & 0x7000) >> 6)|((x & 0xe00000) >> 18)
    raise Exception("State {0} was not possible to orient to fix cubie in place".format(x))

def lu(x):
    return ((x & 0x38000000) << 42)|((x & 0x1c71c71c71c7000fff) << 0)|((x & 0xe00000000) << 30)|((x & 0xe38000e38000000000) >> 12)|((x & 0x38000000e00000) >> 6)|((x & 0xe00000000000000) >> 18)|((x & 0x1c0000) << 3)|((x & 0x7000) << 6)|((x & 0x38000) >> 3)
    

def ld(x):
    return ((x & 0xe38000e38000000) << 12)|((x & 0x1c71c71c71c7000fff) << 0)|((x & 0x38000000000) << 18)|((x & 0xe00000038000) << 6)|((x & 0x38000000000000000) >> 30)|((x & 0xe00000000000000000) >> 42)|((x & 0xe00000) >> 3)|((x & 0x7000) << 3)|((x & 0x1c0000) >> 6)
    

def ru(x):
    return ((x & 0xe38e38e38e38fff000) << 0)|((x & 0x7000000) << 42)|((x & 0x1c0000000) << 30)|((x & 0x1c70001c7000000000) >> 12)|((x & 0x70000000001c0) >> 6)|((x & 0x1c0000000000000) >> 18)|((x & 0x38) << 6)|((x & 0xe00) >> 3)|((x & 0x7) << 3)
    

def rd(x):
    return ((x & 0xe38e38e38e38fff000) << 0)|((x & 0x1c70001c7000000) << 12)|((x & 0x7000000000) << 18)|((x & 0x1c0000000007) << 6)|((x & 0x7000000000000000) >> 30)|((x & 0x1c0000000000000000) >> 42)|((x & 0x1c0) << 3)|((x & 0xe00) >> 6)|((x & 0x38) >> 3)
    

def fl(x):
    return ((x & 0x1c0000000000000000) << 3)|((x & 0x7000000000000000) << 6)|((x & 0xe00000000000000000) >> 6)|((x & 0x38000000000000000) >> 3)|((x & 0xfc0ffffc0e381c7) << 0)|((x & 0xe00) << 42)|((x & 0x38) << 45)|((x & 0x1c0000) << 9)|((x & 0x7000) << 12)|((x & 0x7000000000000) >> 30)|((x & 0x38000000000000) >> 39)|((x & 0x7000000) >> 15)|((x & 0x38000000) >> 24)
    

def fr(x):
    return ((x & 0x38000000000000000) << 6)|((x & 0xe00000000000000000) >> 3)|((x & 0x7000000000000000) << 3)|((x & 0x1c0000000000000000) >> 6)|((x & 0xfc0ffffc0e381c7) << 0)|((x & 0x7000) << 39)|((x & 0x1c0000) << 30)|((x & 0x38) << 24)|((x & 0xe00) << 15)|((x & 0x38000000) >> 9)|((x & 0x7000000) >> 12)|((x & 0x38000000000000) >> 42)|((x & 0x7000000000000) >> 45)
    

def bl(x):
    return ((x & 0xfff03f00003f1c7e38) << 0)|((x & 0x1c0) << 51)|((x & 0x7) << 54)|((x & 0x1c0000000000) << 3)|((x & 0x7000000000) << 6)|((x & 0xe00000000000) >> 6)|((x & 0x38000000000) >> 3)|((x & 0xe00000) << 12)|((x & 0x38000) << 15)|((x & 0x1c0000e00000000) >> 33)|((x & 0xe00000000000000) >> 42)|((x & 0x1c0000000) >> 24)
    

def br(x):
    return ((x & 0xfff03f00003f1c7e38) << 0)|((x & 0x38000) << 42)|((x & 0xe00007) << 33)|((x & 0x38000000000) << 6)|((x & 0xe00000000000) >> 3)|((x & 0x7000000000) << 3)|((x & 0x1c0000000000) >> 6)|((x & 0x1c0) << 24)|((x & 0xe00000000) >> 12)|((x & 0x1c0000000) >> 15)|((x & 0xe00000000000000) >> 51)|((x & 0x1c0000000000000) >> 54)
    

def ul(x):
    return ((x & 0xfc0) << 60)|((x & 0x3f00003ffff03f03f) << 0)|((x & 0x38000000000000) << 6)|((x & 0xe00000000000000) >> 3)|((x & 0x7000000000000) << 3)|((x & 0x1c0000000000000) >> 6)|((x & 0x1c0000) << 27)|((x & 0xe00000) << 21)|((x & 0xfc0000000000000000) >> 48)|((x & 0x1c0000000000) >> 33)|((x & 0xe00000000000) >> 39)
    

def ur(x):
    return ((x & 0xfc0000) << 48)|((x & 0x3f00003ffff03f03f) << 0)|((x & 0x1c0000000000000) << 3)|((x & 0x7000000000000) << 6)|((x & 0xe00000000000000) >> 6)|((x & 0x38000000000000) >> 3)|((x & 0x1c0) << 39)|((x & 0xe00) << 33)|((x & 0x1c0000000000) >> 21)|((x & 0xe00000000000) >> 27)|((x & 0xfc0000000000000000) >> 60)
    

def dl(x):
    return ((x & 0xfc0ffffc0000fc0fc0) << 0)|((x & 0x3f) << 60)|((x & 0x7000) << 27)|((x & 0x38000) << 21)|((x & 0x38000000) << 6)|((x & 0xe00000000) >> 3)|((x & 0x7000000) << 3)|((x & 0x1c0000000) >> 6)|((x & 0x3f000000000000000) >> 48)|((x & 0x7000000000) >> 33)|((x & 0x38000000000) >> 39)
    

def dr(x):
    return ((x & 0xfc0ffffc0000fc0fc0) << 0)|((x & 0x3f000) << 48)|((x & 0x7) << 39)|((x & 0x38) << 33)|((x & 0x1c0000000) << 3)|((x & 0x7000000) << 6)|((x & 0xe00000000) >> 6)|((x & 0x38000000) >> 3)|((x & 0x7000000000) >> 21)|((x & 0x38000000000) >> 27)|((x & 0x3f000000000000000) >> 60)
    

OPS = [lu, ld, ru, rd, fl, fr, bl, br, ul, ur, dl, dr]
OPS_DICT = {fn.__name__: fn for fn in OPS}
FIXED_CUBIE_OPS = [lu, ld, bl, br, ul, ur]
FIXED_CUBIE_OPS_DICT = {fn.__name__: fn for fn in FIXED_CUBIE_OPS}
SOLVED_CUBE_STATE = 674788526559709289910
def main():
    pass
    