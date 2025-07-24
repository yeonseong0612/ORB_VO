

def save_trajectory_txt(trajectory, filename="trajectory.txt"):
    with open(filename, "w") as f:
        for pose in trajectory:
            # 3x4 행렬 (R | t) 추출
            Rt = pose[:3, :]
            Rt_flat = Rt.reshape(-1)  # 12개 float
            line = " ".join(f"{v:.6f}" for v in Rt_flat.tolist())
            f.write(line + "\n")