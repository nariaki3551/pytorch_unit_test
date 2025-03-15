#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "utils.h"

void testReducescatter(c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg) {
    int count = 64;

    auto size_ = pg->getSize();
    auto rank = pg->getRank();

    auto inputTensors = std::vector<std::vector<at::Tensor>>(1);
    inputTensors[0] = std::vector<at::Tensor>(size_);
    for (const auto i : c10::irange(size_)) {
        inputTensors[0][i] = at::ones({count}, at::kInt) * i;
    }
    auto outputTensors = std::vector<at::Tensor>(1);
    outputTensors[0] = at::zeros({count}, at::kInt);

    std::cout << "[Rank " << rank << "] inputs: " << tensor_to_string(inputTensors[0]) << std::endl;
    auto work = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupMPI::AsyncWorkDev>(
        pg->reduce_scatter(outputTensors, inputTensors)
    );
    work->hardWait();
    std::cout << "[Rank " << rank << "] outputs: " << tensor_to_string(outputTensors[0]) << std::endl;
}

int main(int argc, char* argv[]) {
    // sleep(10);
    // std::this_thread::sleep_for(std::chrono::seconds(10));

    auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
    
    testReducescatter(pg);

    return 0;
}
