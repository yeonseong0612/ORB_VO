#include "orb_wrapper.h"
#include "orb.h"
#include "orb_structures.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> orb_match(at::Tensor image1, at::Tensor image2, int max_features) {
    TORCH_CHECK(image1.is_cuda() && image2.is_cuda(), "Image must be CUDA tensor");
    TORCH_CHECK(image1.dtype() == at::kByte && image2.dtype() == at::kByte, "Image must be uint8");
    TORCH_CHECK(image1.dim() == 2 && image1.dim() == 2, "Image must be 2D grayscale");

    //이미지 정보
    int height = image1.size(0);
    int width = image1.size(1);
    int3 whp0 = make_int3(width, height, 0);
    unsigned char* img1_ptr = image1.data_ptr<unsigned char>();
    unsigned char* img2_ptr = image2.data_ptr<unsigned char>();

    // ORB 초기화
    orb::Orbor detector;
    detector.init(
        5,
        31,
        2,
        orb::HARRIS_SCORE,
        31,
        20,
        -1,
        max_features);

    // ORB 데이터 구조
    orb::OrbData data1, data2;
    detector.initOrbData(data1, max_features, true, true);
    detector.initOrbData(data2, max_features, true, true);

    void* desc1_ptr = nullptr;
    void* desc2_ptr = nullptr;

    // 특징점 추출 및 디스크립터 계산
    detector.detectAndCompute(img1_ptr, data1, whp0, &desc1_ptr, true);
    detector.detectAndCompute(img2_ptr, data2, whp0, &desc2_ptr, true);

    int n1 = data1.num_pts;
    int n2 = data2.num_pts;
    // GPU -> CPU로 전체 구조체 복사
    CHECK(cudaMemcpy(data1.h_data, data1.d_data, sizeof(orb::OrbPoint) * n1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(data2.h_data, data2.d_data, sizeof(orb::OrbPoint) * n2, cudaMemcpyDeviceToHost));

    // 매칭 수행
    detector.match(data1, data2, (unsigned char*)desc1_ptr, (unsigned char*)desc2_ptr);

    // 유요한 매칭만 추출
    std::vector<float> kpts1_vec, kpts2_vec;
    std::vector<int64_t> match_idx_vec;

    for(int i = 0; i < n1; i++) {
        int match_idx = data1.h_data[i].match;
        if(match_idx >= 0 && match_idx < n2) {
            kpts1_vec.push_back(static_cast<float>(data1.h_data[i].x));
            kpts1_vec.push_back(static_cast<float>(data1.h_data[i].y));
            kpts2_vec.push_back(static_cast<float>(data2.h_data[match_idx].x));
            kpts2_vec.push_back(static_cast<float>(data2.h_data[match_idx].y));
            match_idx_vec.push_back(match_idx);
        }
    }

    int N = match_idx_vec.size();
    at::Tensor kpts1 = torch::from_blob(kpts1_vec.data(), {N, 2}, torch::dtype(torch::kFloat32)).clone().cuda();
    at::Tensor kpts2 = torch::from_blob(kpts2_vec.data(), {N, 2}, torch::dtype(torch::kFloat32)).clone().cuda();
    at::Tensor matches = torch::from_blob(match_idx_vec.data(), {N}, torch::dtype(torch::kInt64)).clone().cuda();

    //메모리 해제
    detector.freeOrbData(data1);
    detector.freeOrbData(data2);
    cudaFree(desc1_ptr);
    cudaFree(desc2_ptr);

    return {kpts1, kpts2, matches};
}