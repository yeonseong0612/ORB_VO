#pragma once
#include "orb_structures.h"
#include <torch/extension.h>
#include <tuple>


std::tuple<at::Tensor, at::Tensor, at::Tensor> orb_match(at::Tensor image1, at::Tensor image2, int max_features);
