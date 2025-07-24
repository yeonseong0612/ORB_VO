import os
import cv2
import numpy as np

def load_kitti_image_pair(seq_path, index):
    """
    KITTI 시퀀스 경로와 프레임 인덱스를 받아,
    왼쪽 이미지 (image_0), 오른쪽 이미지 (image_1) 각각 하나씩 반환한다.

    Args:
        seq_path (str): KITTI 시퀀스 경로 (예: '.../sequences/00')
        index (int): 이미지 인덱스 (예: 0 → 000000.png)

    Returns:
        left_img (np.ndarray): (H, W), uint8
        right_img (np.ndarray): (H, W), uint8
    """

    filename = f"{index:06d}.png"

    left_path = os.path.join(seq_path, "image_0", filename)
    right_path = os.path.join(seq_path, "image_1", filename)

    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise FileNotFoundError(f"[ERROR] Image Load failed\n → {left_path}\n → {right_path}")

    return left_img, right_img


def count_kitti_images(seq_path):
    img0_path = os.path.join(seq_path, "image_0")
    img_files = [f for f in os.listdir(img0_path) if f.endswith(".png")]
    return len(img_files)

def read_calib_txt(calib_path):
    P2 = None
    P2 = None
    
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith("P2"):
                values = list(map(float, line.strip().split()[1:]))
                P2 = np.array(values).reshape(3, 4)
            elif line.startswith("P3:"):
                values = list(map(float, line.strip().split()[1:]))
                P3 = np.array(values).reshape(3, 4)
        if P2 is None or P3 is None:
            raise RuntimeError("P2 or P3 matrix not found in calib.txt")
        
        fx = P2[0, 0]
        fy = P2[1, 1]
        cx = P2[0, 2]
        cy = P2[1, 2]
        tx_left = P2[0, 3] / fx
        tx_right = P3[0, 3] / fx
        baseline = 0.54

        return fx, fy, cx, cy, baseline
    