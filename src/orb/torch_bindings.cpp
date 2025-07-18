#include <torch/extension.h>
#include <tuple>
#include "orb_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("orb_match", &orb_match, "ORB Detect and Match {CUDA}");
}