#include <torch/extension.h>

#include "hashencoder.h"

// 将C++函数 hash_encode_forward 和 hash_encode_backward 绑定（bind）到Python的PyTorch扩展模块上，从而使这两个函数可以从Python环境中直接调用。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_encode_forward", &hash_encode_forward, "hash encode forward (CUDA)");
    m.def("hash_encode_backward", &hash_encode_backward, "hash encode backward (CUDA)");
}