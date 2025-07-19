from utils.disparitiy import compute_disparities

def depth(kpts_left, kpts_right, fx, baseline):
    depths = []
    disparities = compute_disparities(kpts_left, kpts_right)
    for i in range(disparities):
        d = disparities[i]
        if d > 0:
            z = fx * baseline / d
            depths.append(z)
        else:
            depth.append(0.0)
            
    return depths
        