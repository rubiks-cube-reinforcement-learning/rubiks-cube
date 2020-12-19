from xcs229ii_cube.code_generator.numba_cuda import generate_3cube_numba_code
from xcs229ii_cube.examples_and_commands import bench_rust, bench_python, bench_precomputed, refresh_generated_code, \
    build_nn_2cube_dataset

if __name__ == '__main__':
    refresh_generated_code()
    # precompute_all_cube2_moves()
    # bench_rust()
    # bench_python()
    # bench_precomputed()
    # build_nn_2cube_dataset()
