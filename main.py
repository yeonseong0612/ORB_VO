import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.dataloader import load_kitti_image_pair, count_kitti_images, read_calib_txt
from utils.make_3d_points import make_3d_points
from utils.trajectory import total_loop
from utils.eval import save_trajectory_txt
import orb_cuda

def main():
    # === 0. 초기화 및 경로 설정 ===
    orb_cuda.init_device(0)
    orb_cuda.init_detector(1241, 376)
    seq_path = "/home/yskim/projects/ORB_VO/datasets/sequences/00"
    calib_path = "/home/yskim/projects/ORB_VO/datasets/sequences/00/calib.txt"
    fx, fy, cx, cy, baseline = read_calib_txt(calib_path)
    K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]], dtype=np.float32)
    
    points_3d_list = []
    points_2d_list = []
    
    # === 1. 전체 프레임 수 확인 ===
    num_images = count_kitti_images(seq_path)
    print(f'[INFO] Number of images : {num_images}')
    
    # === 2. 루프 시작 ===
    print("[INFO] Running Loop ...")
    for t in range(500):
        # i번째, i+1번째 좌우 이미지 로드
        left_t, right_t = load_kitti_image_pair(seq_path, t)
        left_tp1, _ = load_kitti_image_pair(seq_path, t + 1)

        # PyTorch 텐서 변환
        left_t_tensor = torch.from_numpy(left_t).to(torch.uint8).contiguous().cuda()
        right_t_tensor = torch.from_numpy(right_t).to(torch.uint8).contiguous().cuda()
        left_tp1_tensor = torch.from_numpy(left_tp1).to(torch.uint8).contiguous().cuda()

        # 스테레오 매칭 → 3D 포인트 생성
        kpt_l, kpt_r, idx_l, idx_r = orb_cuda.orb_match(left_t_tensor, right_t_tensor, 10000, 0)
        pts_3d_all, valid_mask = make_3d_points(kpt_l, kpt_r, idx_l, idx_r, fx, baseline, cx, cy)
        valid_idx_l = idx_l[valid_mask]
        
        # 시간 t+1과의 2D 매칭
        _, kpt_tp1, _, idx_tp1 = orb_cuda.orb_match(left_t_tensor, left_tp1_tensor, 10000, 0)

        # numpy로 교집합 구하기
        np_valid_idx_l = valid_idx_l.cpu().numpy()
        np_idx_tp1     = idx_tp1.cpu().numpy()
        intersection, idx_3d, idx_2d = np.intersect1d(np_valid_idx_l, np_idx_tp1, return_indices=True)

        # torch로 다시 변환
        idx_3d = torch.from_numpy(idx_3d).to(valid_idx_l.device)
        idx_2d = torch.from_numpy(idx_2d).to(idx_tp1.device)

        # 매칭된 포인트 저장
        matched_3d = pts_3d_all[idx_3d].cpu().numpy()
        matched_2d = kpt_tp1[idx_tp1][idx_2d].cpu().numpy()

        points_3d_list.append(matched_3d)
        points_2d_list.append(matched_2d)

        
        if t % 100 == 0 or t == num_images - 2:
            print(f"[INFO] Processing frame {t}/{num_images - 1}")
    print("Loop end & start trajectory")
    trajectory = total_loop(num_images, points_3d_list, points_2d_list, K)
    save_trajectory_txt(trajectory, "/home/yskim/projects/ORB_VO/Results/trajectory_kitti_format.txt")


    


if __name__ == "__main__":
    main()
