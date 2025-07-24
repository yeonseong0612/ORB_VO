import cv2
import numbers as np

#points_3d : t번째 프레임에서 얻은 (N, 3)numpy array
#points_2d : t+1번째 프레임의 대응 ORB 키포인트(N, 2)
# K : 내부 카메라 행렬(fx, fx, cx, cy)
def pose_estimation(points_3d, points_2d, K):
    if points_3d.shape[0] < 6 or points_2d.shape[0] < 6:
        raise RuntimeError("PnP 실패 : 대응점이 부족합니다. (6개 미만)")

    if points_3d.shape[0] != points_2d.shape[0]:
        raise RuntimeError(f"PnP 실패 : 3D({points_3d.shape[0]}) vs 2D({points_2d.shape[0]}) 수 불일치")

    if points_3d.shape[1] != 3 or points_2d.shape[1] != 2:
        raise RuntimeError("PnP 실패 : 입력 차원 오류 (3D는 (N,3), 2D는 (N,2) 이어야 함)")
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=10.0,     # 더 크게 해도 됨 (예: 10.0)
        iterationsCount=1000,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP 실패 : 정합된 inlier 수 부족")
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(-1)
    
    return R, T
    