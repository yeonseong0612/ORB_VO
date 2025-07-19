import cv2
import numbers as np

#points_3d : t번째 프레임에서 얻은 (N, 3)numpy array
#points_2d : t+1번째 프레임의 대응 ORB 키포인트(N, 2)
# K : 내부 카메라 행렬(fx, fx, cx, cy)
def pose_estimation(points_3d, points_2d, K):
    assert points_3d.shape[0] == points_2d.shape[0] , "3D-2D 포인트수 mismatch"
    assert points_3d.shape[1] == 3 and points_2d.shape[1] == 2, "입력 차원 오류"
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=8.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        raise RuntimeError("PnP 실패 : 충분히 정합된 매칭점이 없습니다.")
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(-1)
    
    return R, T
    