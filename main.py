from code_generation import print_python_code
from cube2.cube import Cube2
from cube3.cube import Cube3

if __name__ == '__main__':
    print(Cube3().as_stickers_vector)
    print_python_code(Cube2)
    # print_rust_code()
import doctest
doctest.testmod()
