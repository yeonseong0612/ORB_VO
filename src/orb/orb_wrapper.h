#pragma once
#include "orb_structures.h"
#include <torch/extension.h>
#include <tuple>


void init_device(int dev);
void init_detector(int width, int height);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> orb_match(const at::Tensor img1, const at::Tensor img2, int max_features, int dev);
