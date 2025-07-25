# 🛰️ ORB Visual Odometry

## Summary
This project implements a Stereo ORB Visual Odometry system to estimate the camera's trajectory from a stereo image sequence.

## 1. Load Stereo Images
- Load left_t, right_t, left_t+1, right_t+1 from KITTI Dataset
- Convert to grayscale

## 2. Extract ORB Features(with CUDA)
- Extract ORB KeyPoints and descriptors from each frame
- Accelerated using PyTorch C++/CUDA Extensions

## 3. Stereo Matching
### 📌 Objective
For each ORB keypoint detected in the left image, we find a corresondence on the same horizontal line in the right image to compute
disparity, which enables 3D reconstruction

### ⚙️ Processing Steps
1. Extract ORB keypoints and descriptors from teh left image
2. Extract ORB keypoints and descriptors from the right image
3. Select candidate right keypoints on the same row vL
4. Compute Hamming distance between descriptors and select the best match
5. Validate the disparityz
6. Output : (uL, vL, uR)

## 4. 3D Point Reconstruction
Given:
- uL, vL: coordinates in left image
- uR: matched x-coordinate in right image
- fx: focal length
- cx, cy: principal point
- baseline: distance between stereo cameras

Compute 3D point (X, Y, Z) as:

$$
Z = \frac{f_x \cdot \text{baseline}}{u_L - u_R}, \quad
X = \frac{(u_L - c_x) \cdot Z}{f_x}, \quad
Y = \frac{(v_L - c_y) \cdot Z}{f_x}
$$

## 5. Temporal Matching(3D-3D Correspondences)

## 6. Pose Estimation(3D-3D)

## 7. Trajectory Estimation
