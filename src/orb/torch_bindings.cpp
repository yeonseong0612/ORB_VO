#include <torch/extension.h>
#include "orb_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("detect_and_compute", &detect_and_compute, "CUDA ORB detect_and_compute");
}