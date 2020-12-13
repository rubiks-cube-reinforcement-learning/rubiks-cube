#!/bin/bash

set -e

cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
cp target/release/librubiks_cube_rust.dylib ./rubiks_cube_rust.so
echo "import rubiks_cube_rust as r; print(r.canary()) " | python
echo "import rubiks_cube_rust as r; r.load_lookup_table('./results-cubies-fixed.txt'); print(r.solve_batch([674788526559709289910, 1643468079509558581426])) " | python
cp ./rubiks_cube_rust.so ../

echo "Build successful!"