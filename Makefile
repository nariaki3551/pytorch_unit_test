all: torch_test mpi_test

torch_test:
	make -C src/torch

mpi_test:
	make -C src/mpi

clean:
	make -C src/torch clean
	make -C src/mpi clean
