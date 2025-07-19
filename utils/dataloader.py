import os
import cv2
import numpy as np

import os
import cv2

def load_kitti_images(seq_path, max_frames):
    img0_path = os.path.join(seq_path, 'image_0')
    img1_path = os.path.join(seq_path, 'image_1')
    
    img_files = sorted(f for f in os.listdir(img0_path) if f.endswith('.png'))[:max_frames]
    
    left_images = []
    right_images = []
    
    for frame in img_files:
        left_path = os.path.join(img0_path, frame)
        right_path = os.path.join(img1_path, frame)

        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left is None or right is None:
            print(f'[WARING] Failed to load image pair: {frame}')
            print(f' → Left path: {left_path}')
            print(f' → Right path: {right_path}')
            continue
        
        left_images.append(left)
        right_images.append(right)
    
    print(f'[INFO] Loaded {len(left_images)} image pairs')
    return left_images, right_images
