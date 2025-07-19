import numpy as np
import torch
from utils.dataloader import load_kitti_images
from utils.disparitiy import compute_disparities
import orb_cuda

def main():
    seq_path ="datasets/sequences/00"
    max_frames = 1

    trajectory = []
    pose = np.eye(4)
    left_imgs, right_imgs = load_kitti_images('datasets/sequences/00', max_frames)
    left_img = left_imgs[0]
    right_img = right_imgs[0]
    img1_tensor = torch.from_numpy(left_img).to(torch.uint8).contiguous().cuda()
    img2_tensor = torch.from_numpy(right_img).to(torch.uint8).contiguous().cuda()
    
    kpt1, kpt2, _, _, matches = orb_cuda.orb_match(img1_tensor, img2_tensor, 1)
    disparities = compute_disparities(kpt1, kpt2)


    print("[INFO] Valid disparities:", disparities[:10])
    


if __name__ == "__main__":
    main()