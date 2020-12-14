## Rubik's cube solver

This repo, for Stanford's xCs229ii course, is many things:

1. A high-level rubik's cube representation based on cubies
1. A code generator for computationally-efficient program representation based on bits
1. An optimal solver for 2x2x2 cube
1. A reinforcement learning 2x2x2 rubik's cube solver
1. A reinforcement learning 3x3x3 rubik's cube solver 

## Getting started

1. Unpack a database of all 2-cube states: `gunzip rust-experiment/results-cubies-fixed.txt.gz`
1. ... more instructions will come ...

See examples_and_commands.py for some benchmarks and usage examples.

## Building `rubiks_cube_rust` python module

On mac:

```bash
cd rust-experiment
brew install rustup
rustup-init
rustup override set nightly  
bash build.sh
```
