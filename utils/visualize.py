import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_trajectory(trajectory):
    xs = []
    zs = []
    
    for pose in trajectory:
        x = pose[0, 3]
        z = pose[2, 3]
        xs.append(x)
        zs.append(z)
        
        plt.figure(figsize=(8, 6))
        plt.plot(xs, zs, marker= 'o', linestyle= '-', color='blue')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Estimated Camera Trajectory')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
def load_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            pose = np.array(vals).reshape(3, 4)
            poses.append(pose)
    return poses

def extract_positions(poses):
    return np.array([pose[:, 3] for pose in poses])  # translation 벡터만

def plot_and_save_trajectory(positions, save_path="/home/yskim/projects/ORB_VO/Results/gt_trajectory.png"):
    plt.figure(figsize=(10,6))
    plt.plot(positions[:, 0], positions[:, 2], label='GT Trajectory')  # X-Z 평면
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('KITTI GT Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"[✔] Trajectory image saved to: {save_path}")
    plt.show()

def save_trajectory_txt(positions, save_path="00_gt_trajectory_xyz.txt"):
    np.savetxt(save_path, positions, fmt="%.6f", delimiter=' ')
    print(f"[✔] Trajectory positions saved to: {save_path}")

# 사용 예시
pose_file = "/home/yskim/projects/ORB_VO/datasets/sequences/00/poses/00.txt"  # 경로는 네가 사용하는 위치로 수정
poses = load_poses(pose_file)
positions = extract_positions(poses)

plot_and_save_trajectory(positions, save_path="00_gt_trajectory.png")
save_trajectory_txt(positions, save_path="/home/yskim/projects/ORB_VO/Results/gt_trajectory_xyz.txt")