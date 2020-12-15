from code_generator.numba_cuda import generate_3cube_numba_code
from examples_and_commands import bench_rust, bench_python, bench_precomputed, refresh_bitwise_ops_code, \
    build_nn_2cube_dataset

if __name__ == '__main__':
    # refresh_bitwise_ops_code()
    # precompute_all_cube2_moves()
    # bench_rust()
    # bench_python()
    # bench_precomputed()
    # build_nn_2cube_dataset()
    print(generate_3cube_numba_code())
