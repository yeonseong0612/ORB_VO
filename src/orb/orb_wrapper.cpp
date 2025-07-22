#include "orb_wrapper.h"
#include "orb.h"
#include "orb_structures.h"


namespace {
    const int kMaxFeatures = 1000;
    const int kDescSize = 32; // 256 bits

    orb::Orbor detector;

    bool initialized = false;

    void initialize_orb_detector() {
        if (!initialized) {
            detector.init(
                5,              // noctaves
                31,             // edge_threshold
                4,              // wta_k
                orb::HARRIS_SCORE, // score_type
                31,             // patch_size
                20,             // fast_threshold
                -1,             // retain_topn (-1 means all)
                kMaxFeatures    // max_pts
            );
            initialized = true;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> orb_match(at::Tensor image1, at::Tensor image2, int max_features) {
    TORCH_CHECK(image1.is_cuda() && image2.is_cuda(), "Image must be CUDA tensor");
    TORCH_CHECK(image1.dtype() == at::kByte && image2.dtype() == at::kByte, "Image must be uint8");
    TORCH_CHECK(image1.dim() == 2 && image1.dim() == 2, "Image must be 2D grayscale");
    TORCH_CHECK(image1.is_contiguous(), "Image1 must be contiguous");
    TORCH_CHECK(image2.is_contiguous(), "Image2 must be contiguous");
    
    // 1. ORB 초기화
    initialize_orb_detector();


    // 1. 이미지 정보
    const int height = image1.size(0);
    const int width = image1.size(1);
    const int pitch = image1.size(1);
    int3 whp0 = make_int3(1280, height, 1280);
    std::cout << "size: " <<whp0.x<< whp0.y << whp0.z <<std::endl;

    // 2. CUDA 이미지 포인터 얻기
    unsigned char* img1_ptr = image1.data_ptr<unsigned char>();
    unsigned char* img2_ptr = image2.data_ptr<unsigned char>();

    // 3. ORB 데이터 초기화 및 특징점 계산
    orb::OrbData data1, data2;
    detector.initOrbData(data1, max_features, true, true);
    detector.initOrbData(data2, max_features, true, true);

    unsigned char* desc1 = nullptr;
    unsigned char* desc2 = nullptr;

    // 4. 특징점 추출 및 디스크립터 계산
    detector.detectAndCompute(img1_ptr, data1, whp0, (void**)&desc1, true);
    detector.detectAndCompute(img2_ptr, data2, whp0, (void**)&desc2, true);

    std::cout << "[DEBUG] Num Keypoints in Image1: " << data1.num_pts << std::endl;
    std::cout << "[DEBUG] Num Keypoints in Image2: " << data2.num_pts << std::endl;

    int n1 = data1.num_pts;
    int n2 = data2.num_pts;

    // 5. GPU -> CPU로 전체 구조체 복사
    CHECK(cudaMemcpy(data1.h_data, data1.d_data, sizeof(orb::OrbPoint) * n1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(data2.h_data, data2.d_data, sizeof(orb::OrbPoint) * n2, cudaMemcpyDeviceToHost));
    std::cout << "[DEBUG] data1 Keypoints (first 20):\n";
    for (int i = 0; i < std::min(n1, 20); ++i) {
        std::cout << "  [" << i << "] (x, y) = (" << data1.h_data[i].x << ", " << data1.h_data[i].y << "), "
                << "octave: " << (int)data1.h_data[i].octave << ", angle: " << data1.h_data[i].angle << std::endl;
    }

    // 6. 디스크립터 텐서
    at::Tensor desc1_tensor = torch::empty({n1, kDescSize}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    at::Tensor desc2_tensor = torch::empty({n2, kDescSize}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    CHECK(cudaMemcpy(desc1_tensor.data_ptr<uint8_t>(), desc1, sizeof(uint8_t) * n1 * kDescSize, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(desc2_tensor.data_ptr<uint8_t>(), desc2, sizeof(uint8_t) * n2 * kDescSize, cudaMemcpyDeviceToDevice));

    // 7. 매칭 
    detector.match(data1, data2, desc1, desc2);


    // 8. 매칭 키포인트 추출
    std::vector<float> kpts1_vec, kpts2_vec;
    std::vector<int64_t> match_idx_vec;
    for (int i = 0; i < n1; ++i) {
        int match_idx = data1.h_data[i].match;
        if (match_idx >= 0 && match_idx < n2) {
            kpts1_vec.push_back(data1.h_data[i].x);
            kpts1_vec.push_back(data1.h_data[i].y);
            kpts2_vec.push_back(data2.h_data[match_idx].x);
            kpts2_vec.push_back(data2.h_data[match_idx].y);
            match_idx_vec.push_back(match_idx);
        }
    }
    std::cout << "[DEBUG] Valid Matches: " << match_idx_vec.size() << " / " << n1 << std::endl;


    int N = match_idx_vec.size();
    at::Tensor kpts1 = torch::from_blob(kpts1_vec.data(), {N, 2}, torch::dtype(torch::kFloat32)).clone().cuda();
    at::Tensor kpts2 = torch::from_blob(kpts2_vec.data(), {N, 2}, torch::dtype(torch::kFloat32)).clone().cuda();
    at::Tensor matches = torch::from_blob(match_idx_vec.data(), {N}, torch::dtype(torch::kInt64)).clone().cuda();

    // (선택) 디스크립터 첫 바이트 확인
    uint8_t host_desc[kDescSize];
    cudaMemcpy(host_desc, desc1, kDescSize, cudaMemcpyDeviceToHost);
    std::cout << "[DEBUG] First descriptor: ";
    for (int i = 0; i < kDescSize; ++i) std::cout << std::hex << (int)host_desc[i] << " ";
    std::cout << std::dec << std::endl;

    std::cout << "[DEBUG] Freeing desc1_ptr: " << (void*)desc1 << std::endl;


    // 9. 메모리 해제
    detector.freeOrbData(data1);
    detector.freeOrbData(data2);
    if (desc1) {
    std::cout << "[DEBUG] Freeing desc1..." << std::endl;
    CHECK(cudaFree(desc1));
    }
    if (desc2) {
        std::cout << "[DEBUG] Freeing desc2..." << std::endl;
        CHECK(cudaFree(desc2));
    }

    return {kpts1, kpts2, desc1_tensor, desc2_tensor, matches};

}