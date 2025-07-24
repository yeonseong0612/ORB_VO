import numpy as np
from utils.pose import pose_estimation



def accmulate_pose(R, T, prev_pose):
    """
    현재 프레임의 R, T를 이전 포즈에 누적해서 새로운 pose를 계산
    - R, T: 현재 프레임 기준 pose (camera-to-world 변환)
    - prev_pose: 이전까지 누적된 pose (4x4)
    """
        
    # 현재 프레임의 변환 행렬
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = T.reshape(-1)
        
    # 새로운 누적 포즈 = 이전 포즈 @ 현재 변환의 역행렬
    new_pose = prev_pose @ np.linalg.inv(Rt)
        
    return new_pose

def total_loop(num_frames, points_3d_list, points_2d_list, K):
    trajectory = [] # 누적 포즈 저장 리스트
    pose = np.eye(4) # 초기 포즈(월드 원점)
    trajectory.append(pose.copy()) # 첫 프레임 추가
        
    for t in range(1, num_frames):
        points_3d = points_3d_list[t - 1]
        points_2d = points_2d_list[t - 1]
        R, T = pose_estimation(points_3d, points_2d, K)
                
        pose = accmulate_pose(R, T, pose)
        trajectory.append(pose)

    return trajectory