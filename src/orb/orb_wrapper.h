#pragma once
#include "orb_structures.h"
#include <torch/extension.h>
#include <tuple>


std::tuple<at::Tensor, at::Tensor> detect_and_compute(at::Tensor image_tensor);
