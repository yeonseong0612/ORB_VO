#include "orb_wrapper.h"
#include "orb.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_device", &init_device, "Initialize CUDA Device");
    m.def("init_detector", &init_detector, "Initialize ORB Detector", 
        py::arg("width"), py::arg("height"));
    m.def("orb_match", &orb_match, "Run ORB Matching",
        py::arg("img1"), py::arg("img2"), py::arg("max_features"), py::arg("device"));
}