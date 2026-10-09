#include <torch/extension.h>

void reconstruct(at::Tensor unpacked, at::Tensor packed, float bits, bool mcg, bool mul1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("reconstruct", &reconstruct);
}
