import matplotlib.pyplot as plt
import numpy as np

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