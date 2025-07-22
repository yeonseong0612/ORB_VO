import torch
import cv2
import orb_cuda  # 너가 만든 PyTorch Extension 모듈
import numpy as np

def load_image_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"이미지를 불러올 수 없음: {path}"
    return torch.from_numpy(img).to(torch.uint8).cuda(), img  # (Tensor, numpy)

def visualize_matches(img1_np, img2_np, kpt1, kpt2, matches, max_matches=3000):
    kpt1_np = kpt1.cpu().numpy()
    kpt2_np = kpt2.cpu().numpy()
    matches_np = matches.cpu().numpy()

    img1_color = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_np, cv2.COLOR_GRAY2BGR)

    h = max(img1_color.shape[0], img2_color.shape[0])
    w = img1_color.shape[1] + img2_color.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img1_color.shape[0], :img1_color.shape[1]] = img1_color
    canvas[:img2_color.shape[0], img1_color.shape[1]:] = img2_color

    offset = img1_color.shape[1]

    for i in range(min(len(matches_np), max_matches)):
        idx1 = i
        idx2 = matches_np[i]
        if idx2 < 0 or idx2 >= len(kpt2_np):
            continue  # 잘못된 매칭 무시
        pt1 = tuple(np.round(kpt1_np[idx1]).astype(int))
        pt2 = tuple(np.round(kpt2_np[idx2]).astype(int) + np.array([offset, 0]))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    cv2.imshow("ORB Matching", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 이미지 불러오기
    img1_tensor, img1_np = load_image_grayscale("datasets/sequences/00/image_0/000000.png")
    img2_tensor, img2_np = load_image_grayscale("datasets/sequences/00/image_1/000000.png")
    
    img1_tensor = img1_tensor.contiguous().cuda()
    img2_tensor = img2_tensor.contiguous().cuda()
    # 특징점 추출 및 매칭
    kpt1, kpt2, _, _, matches = orb_cuda.orb_match(img1_tensor, img2_tensor, 3000)

    print(f"[INFO] Keypoints 1: {kpt1.shape}, Keypoints 2: {kpt2.shape}, Matches: {matches.shape}")
    

    # 시각화
    visualize_matches(img1_np, img2_np, kpt1, kpt2, matches)

if __name__ == "__main__":
    main()