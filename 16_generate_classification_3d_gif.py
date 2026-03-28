import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def main():
    print("Initializing 3D Classification GIF Generation...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    las_path = os.path.join(out_dir, "67169_5NKR_CHAKHIRASINGH_ml_classified.las")
    gif_path = os.path.join(out_dir, "rajasthan_classification_3d_rotation.gif")
    
    # Setup the Plot Figure
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off() 
    
    # 1. Load and Sample Point Cloud
    print("Loading and Sampling Point Cloud...")
    las = laspy.read(las_path)
    
    # Balanced sampling for the GIF
    ground_idx = np.where(las.classification == 2)[0]
    other_idx = np.where(las.classification != 2)[0]
    
    n_sample = 80000 # Keeping it manageable for GIF performance
    s_ground = np.random.choice(ground_idx, min(len(ground_idx), n_sample), replace=False)
    s_other = np.random.choice(other_idx, min(len(other_idx), n_sample), replace=False)
    
    z_exag = 2.0
    
    print("Plotting initial classification layers...")
    # Layer 1: Ground Points
    gx, gy, gz = las.x[s_ground], las.y[s_ground], las.z[s_ground]
    sc1 = ax.scatter(gx, gy, gz * z_exag, color='#8B4513', s=0.5, alpha=0.4, depthshade=False)
    
    # Layer 2: Non-Ground Points
    ox, oy, oz = las.x[s_other], las.y[s_other], las.z[s_other]
    sc2 = ax.scatter(ox, oy, oz * z_exag, color='#32CD32', s=0.5, alpha=0.7, depthshade=False)

    ax.set_title("3D ML Classification: Ground vs Structures", color='white', fontsize=16, fontweight='bold', pad=10)
    
    # Create the Animation update loop
    def update(frame):
        ax.view_init(elev=30, azim=frame * 3)
        if frame % 10 == 0:
            print(f"Rendering Classification Frame {frame}/120...")
        return sc1, sc2
        
    print("Rendering high-detail Classification GIF... (This will take a few minutes)")
    anim = FuncAnimation(fig, update, frames=120, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save(gif_path, writer=writer)
    print(f"SUCCESS! Classification 3D GIF generated at: {gif_path}")

if __name__ == "__main__":
    main()
