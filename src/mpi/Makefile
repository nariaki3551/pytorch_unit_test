CXX = mpicxx
CXXFLAGS = -g -O0 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
BIN_DIR = ../../bin

SRC = test_reducescatter test_allgather test_allreduce test_overlap \
      test_allgather_cuda test_reducescatter_cuda test_overlap_cuda cuda_support_check

all: $(SRC)

%: %.cpp
	$(CXX) -o $(BIN_DIR)/$@_mpi $< $(CXXFLAGS)

clean:
	rm -f $(SRC)
