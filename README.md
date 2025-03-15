# pytorch_unit_test

Unit tests for pytorch distributed of MPI.

- src/torch: Collective tests using torch API.
- src/mpi: Collective tests using MPI API (not using torch).

## build

```bash
export TORCH_LIB_PATH=/path/to/your/torch_lib
export TORCH_INCLUDE_PATH=/path/to/your/torch_include # e.g. pytorch/torch/include
mkdir bin
make
```

## run

### C++ test

```bash
LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
mpirun -np 2 ./bin/test_reducescatter_cpp
```

### Python test

```bash
mpirun -n 2 python ./src/python/test_reducescatter.py
```
