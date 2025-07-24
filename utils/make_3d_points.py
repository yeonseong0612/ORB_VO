import torch
from utils.disparitiy import compute_disparities


def make_3d_points(kpts_left, kpts_right, match_idx1, match_idx2, fx, baseline, cx, cy):
    """
    좌우 키포인트와 매칭 결과를 이용하여 3D 좌표를 생성

    Args:
        kpts_left (torch.Tensor): (N, 2) 왼쪽 이미지의 키포인트
        kpts_right (torch.Tensor): (N, 2) 오른쪽 이미지의 키포인트
        match_idx1 (torch.Tensor): (M,)  왼쪽 키포인트 인덱스
        match_idx2 (torch.Tensor): (M,)  오른쪽 키포인트 인덱스
        fx (float): focal length
        baseline (float): stereo baseline (meters)

    Returns:
        points_3d (torch.Tensor): (M, 3) 3D 포인트 (X, Y, Z)
        valid_mask (torch.Tensor): (M,)  유효한 점만 True
    """
    
    # 매칭된 키포인트 좌표 추출
    pts_left = kpts_left[match_idx1]   # (M, 2)
    pts_right = kpts_right[match_idx2] # (M, 2)
    disparity = compute_disparities(pts_left, pts_right)
    
    # 깊이 계산
    eps = 1e-6
    Z = fx * baseline / (disparity + eps)
    
    # 유효한 disparity만 선택 (양수이고 너무 작지 않은 경우)
    valid_mask = (disparity > 1.0) & (Z < 80.0)  # 예: max 80m
    
    # X, Y 계산
    X = (pts_left[:, 0] - cx) * Z / fx
    Y = (pts_left[:, 1] - cy) * Z / fx

    points_3d = torch.stack((X, Y, Z), dim=1)  # (M, 3)

    return points_3d[valid_mask], valid_mask
