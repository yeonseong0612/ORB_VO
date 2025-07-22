import numpy as np
import torch
from utils.dataloader import load_kitti_images
from utils.disparitiy import compute_disparities
import orb_cuda
import cv2
import matplotlib.pyplot as plt
def visualize_keypoints(image1, image2, kpt1, kpt2):
    # numpy로 변환
    img1_np = image1.cpu().numpy()
    img2_np = image2.cpu().numpy()

    # BGR로 변환
    img1_color = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_np, cv2.COLOR_GRAY2BGR)

    # 키포인트 그리기
    for pt in kpt1.cpu().numpy():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img1_color, (x, y), 2, (0, 255, 0), -1)

    for pt in kpt2.cpu().numpy():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img2_color, (x, y), 2, (0, 255, 0), -1)

    # 상하로 붙이기
    combined = np.vstack((img1_color, img2_color))

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.imshow(combined[..., ::-1])  # BGR to RGB
    plt.axis('off')
    plt.title("ORB Keypoints: Top = Left Image, Bottom = Right Image")
    plt.show()

def main():
    seq_path = "datasets/sequences/00"
    max_frames = 1000

    trajectory = []
    pose = np.eye(4)
    left_imgs, right_imgs = load_kitti_images(seq_path, max_frames)
    left_img = left_imgs[0]
    right_img = right_imgs[0]

    img1_tensor = torch.from_numpy(left_img).to(torch.uint8).contiguous().cuda()
    img2_tensor = torch.from_numpy(right_img).to(torch.uint8).contiguous().cuda()
    
    print("start")
    kpt1, kpt2, _, _, matches = orb_cuda.orb_match(img1_tensor, img2_tensor, 1000)
    disparities = compute_disparities(kpt1, kpt2)

    visualize_keypoints(img1_tensor, img2_tensor, kpt1, kpt2)

if __name__ == "__main__":
    main()