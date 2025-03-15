ifndef TORCH_LIB_PATH
$(error TORCH_LIB_PATH is not set. Please set it before running make.)
endif
ifndef TORCH_INCLUDE_PATH
$(error TORCH_INCLUDE_PATH is not set. Please set it before running make.)
endif

INCLUDE_DIRS = -I$(TORCH_INCLUDE_PATH) -I$(TORCH_INCLUDE_PATH)/torch/csrc/api/include -I./src
LIB_DIRS = -L$(TORCH_LIB_PATH)
LIBS = -ltorch -ltorch_cpu -lc10
BIN_DIR = ../../bin
SRCS = test_allgather.cpp test_reducescatter.cpp
BINS = $(SRCS:.cpp=)

all: $(BINS)

%: %.cpp
	mpicxx -std=c++17 -DUSE_C10D_MPI -g -O0 $< -o $(BIN_DIR)/$@_torch $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

clean:
	rm -f $(addprefix $(BIN_DIR)/, $(BINS))
