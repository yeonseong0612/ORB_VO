#include "orb_wrapper.h"
#include "orb.h"
#include "orb_structures.h"

std::tuple<at::Tensor, at::Tensor> detect_and_compute(at::Tensor image_tensor){
    TORCH_CHECK(image_tensor.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(image_tensor.dtype() == at::kByte, "Input must be uint8.");
    TORCH_CHECK(image_tensor.dim() == 2, "Input must be 2D grayscale image");

    auto sizes = image_tensor.sizes();
    int height = sizes[0];
    int width = sizes[1];
    unsigned char* image_ptr = image_tensor.data_ptr<unsigned char>();
    int3 whp0 = make_int3(width, height, 0);

    orb::Orbor detector;
    detector.init(
        5, 31, 2, orb::HARRIS_SCORE,
        31, 20, -1, 1000
    );

    orb::OrbData result;
    detector.initOrbData(result, 1000, true, true);
    
    std::cout << "[Wrapper] npoints: " << result.num_pts << std::endl;


    void* desc_ptr = nullptr;
    detector.detectAndCompute(image_ptr, result, whp0, &desc_ptr, true);
    std::cout << "[Wrapper] npoints: " << result.num_pts << std::endl;

    int num_pts = result.num_pts;
    at::Tensor kpt_tensor = torch::zeros({num_pts, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    at::Tensor desc_tensor = torch::zeros({num_pts, 32}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    // ✅ 구조체 전체 복사 (GPU → CPU)
    CHECK(cudaMemcpy(result.h_data, result.d_data,
                     sizeof(orb::OrbPoint) * num_pts,
                     cudaMemcpyDeviceToHost));

    // ✅ CPU에서 x, y만 추출해서 GPU 텐서로 다시 복사
    std::vector<float> xy_host(2 * num_pts);
    for (int i = 0; i < num_pts; ++i) {
        xy_host[2 * i]     = static_cast<float>(result.h_data[i].x);
        xy_host[2 * i + 1] = static_cast<float>(result.h_data[i].y);
    }

    cudaMemcpy(kpt_tensor.data_ptr<float>(), xy_host.data(),
               sizeof(float) * 2 * num_pts, cudaMemcpyHostToDevice);

    // ✅ 디스크립터 복사 (GPU → GPU)
    cudaMemcpy(desc_tensor.data_ptr<uint8_t>(), desc_ptr,
               sizeof(uint8_t) * 32 * num_pts, cudaMemcpyDeviceToDevice);   
    

    detector.freeOrbData(result);
    return {kpt_tensor, desc_tensor};
}