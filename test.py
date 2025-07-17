import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cuda_orb_ext

image_path = "000117.png"
img =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"Image not found at: {image_path}"

img_tensor = torch.from_numpy(img).to(dtype=torch.uint8, device='cuda')

# 3. ORB 실행
keypoints_tensor, _ = cuda_orb_ext.detect_and_compute(img_tensor)

# 4. Tensor → NumPy (CPU로 이동)
keypoints = keypoints_tensor.cpu().numpy().astype(np.int32)



print(img.shape, img.dtype, img.min(), img.max())
print(keypoints[:10])

# 5. 원본 이미지에 keypoints 시각화
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for pt in keypoints:
    x, y = pt
    cv2.circle(img_color, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

# 6. Matplotlib으로 출력
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("CUDA ORB Keypoints")
plt.axis("off")
plt.show()