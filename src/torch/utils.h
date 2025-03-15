#include <torch/torch.h>
#include <vector>
#include <string>
#include <sstream>

std::string tensor_to_string(const at::Tensor& tensor) {
    std::stringstream ss;
    ss << "[";
    for(const auto i : c10::irange(tensor.numel())) {
        ss << tensor[i].item<float>() << " ";
    }
    ss << "]";
    return ss.str();
}

std::string tensor_to_string(const std::vector<at::Tensor>& tensor_vector) {
    std::stringstream ss;
    ss << "[";
    for(const auto i : c10::irange(tensor_vector.size())) {
        ss << tensor_to_string(tensor_vector[i]) << " ";
    }
    ss << "]";
    return ss.str();
}

