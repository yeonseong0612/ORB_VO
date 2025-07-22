#include "orb_wrapper.h"
#include "orb.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("orb_match", &orb_match, "ORB match function");
}