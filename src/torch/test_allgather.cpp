#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <iostream>
#include <vector>
#include "utils.h"

void testAllgather(c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg) {
    int count = 64;

    auto size_ = pg->getSize();
    auto rank = pg->getRank();

    auto inputTensors = std::vector<at::Tensor>(1);
    inputTensors[0] = at::ones({count}, at::kInt) * rank;

    auto outputTensors = std::vector<std::vector<at::Tensor>>(1);
    outputTensors[0].resize(size_);
    for (const auto k : c10::irange(size_)) {
        outputTensors[0][k] = at::zeros({count}, at::kInt);
    }

    std::cout << "[Rank " << rank << "] inputs: " << tensor_to_string(inputTensors[0]) << std::endl;
    auto work = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupMPI::AsyncWorkDev>(
        pg->allgather(outputTensors, inputTensors)
    );
    work->hardWait();
    std::cout << "[Rank " << rank << "] outputs: " << tensor_to_string(outputTensors[0]) << std::endl;
}

int main(int argc, char* argv[]) {

    auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
    
    testAllgather(pg);

    return 0;
}
