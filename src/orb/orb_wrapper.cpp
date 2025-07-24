#include "orb_wrapper.h"
#include "orb.h"
#include "warmup.h"
#include "orb_structures.h"
#include "cuda_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>

void init_device(int dev){
    warmup();
    std::cout<< "Initializing data.." <<std::endl;
    initDevice(dev);

    

}

std::shared_ptr<orb::Orbor> detector = nullptr;

void init_detector(int width, int height) {
if (!detector) detector = std::make_shared<orb::Orbor>();

	int noctaves = 5;
	int edge_threshold = 31;
	int wta_k = 4;
	orb::ScoreType score_type = orb::ScoreType::HARRIS_SCORE;
	int patch_size = 31;
	int fast_threshold = 20;
	int max_pts = 10000;
	int retain_topn = -1;

	detector->init(noctaves, edge_threshold, wta_k, score_type, patch_size, fast_threshold, retain_topn, max_pts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> orb_match(const at::Tensor img1, const at::Tensor img2, int max_features, int dev) {
	int3 whp1, whp2;
    whp1.x = img1.size(1); whp1.y = img1.size(0); whp1.z = iAlignUp(whp1.x, 128);
    whp2.x = img2.size(1); whp2.y = img2.size(0); whp2.z = iAlignUp(whp2.x, 128);

	size_t size1 = whp1.y * whp1.z * sizeof(unsigned char);
	size_t size2 = whp2.y * whp2.z * sizeof(unsigned char);
	unsigned char* img_1 = NULL;
	unsigned char* img_2 = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img_1, &tmp_pitch, sizeof(unsigned char) * whp1.x, whp1.y));
	CHECK(cudaMallocPitch((void**)&img_2, &tmp_pitch, sizeof(unsigned char) * whp2.x, whp2.y));

	const size_t dpitch1 = sizeof(unsigned char) * whp1.z;
	const size_t spitch1 = sizeof(unsigned char) * whp1.x;
	const size_t dpitch2 = sizeof(unsigned char) * whp2.z;
	const size_t spitch2 = sizeof(unsigned char) * whp2.x;
	CHECK(cudaMemcpy2D(img_1, dpitch1, img1.data_ptr<unsigned char>(), spitch1, spitch1, whp1.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(img_2, dpitch2, img2.data_ptr<unsigned char>(), spitch2, spitch2, whp2.y, cudaMemcpyHostToDevice));

	orb::OrbData orb_data1, orb_data2;
	detector->initOrbData(orb_data1, max_features, true, true);
	detector->initOrbData(orb_data2, max_features, true, true);

	unsigned char* orb_desc1 = NULL;
	unsigned char* orb_desc2 = NULL;

	detector->detectAndCompute(img_1, orb_data1, whp1, (void**)&orb_desc1, true);
	detector->detectAndCompute(img_2, orb_data2, whp2, (void**)&orb_desc2, true);

	detector->match(orb_data1, orb_data2, orb_desc1, orb_desc2);

	// 1. keypoints1
	std::vector<float> kpts1_xy;
	for (int i = 0; i < orb_data1.num_pts; ++i) {
		kpts1_xy.push_back(orb_data1.h_data[i].x);
		kpts1_xy.push_back(orb_data1.h_data[i].y);
	}
	at::Tensor keypoints1 = torch::from_blob(kpts1_xy.data(), { orb_data1.num_pts, 2 }, at::kFloat).clone();

	// 2. keypoints2
	std::vector<float> kpts2_xy;
	for (int i = 0; i < orb_data2.num_pts; ++i) {
		kpts2_xy.push_back(orb_data2.h_data[i].x);
		kpts2_xy.push_back(orb_data2.h_data[i].y);
	}
	at::Tensor keypoints2 = torch::from_blob(kpts2_xy.data(), { orb_data2.num_pts, 2 }, at::kFloat).clone();

	// matched indices from img1 to img2
	std::vector<int> idx1, idx2;
	for (int i = 0; i < orb_data1.num_pts; ++i) {
		int match_idx = orb_data1.h_data[i].match;
		if (match_idx >= 0 && match_idx < orb_data2.num_pts) {
			idx1.push_back(i);
			idx2.push_back(match_idx);
		}
	}

	at::Tensor matches1 = torch::from_blob(idx1.data(), { (long)idx1.size() }, at::kInt).clone();
	at::Tensor matches2 = torch::from_blob(idx2.data(), { (long)idx2.size() }, at::kInt).clone();

	// Free ORB data from device
	detector->freeOrbData(orb_data1);
	detector->freeOrbData(orb_data2);
	if (orb_desc1)
		CHECK(cudaFree(orb_desc1));
	if (orb_desc2)
		CHECK(cudaFree(orb_desc2));
	CHECK(cudaFree(img_1));
	CHECK(cudaFree(img_2));

	return std::make_tuple(keypoints1, keypoints2, matches1, matches2);

}
