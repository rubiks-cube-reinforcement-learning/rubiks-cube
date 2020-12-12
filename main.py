from cube2.code_generation import print_rust_code
from cube3.cube import Cube3

if __name__ == '__main__':
    print(Cube3().as_stickers_vector)
    # print_python_code()
    # print_rust_code()
import doctest
doctest.testmod()
