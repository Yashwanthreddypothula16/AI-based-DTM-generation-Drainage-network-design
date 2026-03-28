import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("Generating Separate 3D Ground vs Non-Ground Visualization...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    las_path = os.path.join(out_dir, "67169_5NKR_CHAKHIRASINGH_ml_classified.las")
    output_png = os.path.join(out_dir, "3D_GROUND_VS_NONGROUND.png")
    
    # 1. Load and Sample Point Cloud
    print("Loading Classified Point Cloud...")
    las = laspy.read(las_path)
    
    # Stratified sample: 50% ground, 50% non-ground for balanced separate view
    ground_idx = np.where(las.classification == 2)[0]
    other_idx = np.where(las.classification != 2)[0]
    
    # Use enough points to show the structure clearly without crashing
    n_sample = 150000 
    s_ground = np.random.choice(ground_idx, min(len(ground_idx), n_sample), replace=False)
    s_other = np.random.choice(other_idx, min(len(other_idx), n_sample), replace=False)
    
    # 2. Setup 3D Plot with Dark Theme
    print("Rendering 3D Classification View...")
    fig = plt.figure(figsize=(20, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Global Vertical Exaggeration
    z_exag = 2.0
    
    # Layer 1: Ground Points (Earthy Brown)
    gx, gy, gz = las.x[s_ground], las.y[s_ground], las.z[s_ground]
    ax.scatter(gx, gy, gz * z_exag, color='#8B4513', s=1.0, alpha=0.5, label='Class 2: Ground Terrain', depthshade=False)
    
    # Layer 2: Non-Ground Points (Vibrant Green)
    ox, oy, oz = las.x[s_other], las.y[s_other], las.z[s_other]
    ax.scatter(ox, oy, oz * z_exag, color='#32CD32', s=1.0, alpha=0.8, label='Class 1: Non-Ground / Buildings / Trees', depthshade=False)

    # Formatting 
    ax.set_title("AI/ML 3D Output Model: Separate Ground & Non-Ground Analysis", color='white', fontsize=24, fontweight='bold', pad=30)
    ax.set_axis_off()
    
    # View Angle
    ax.view_init(elev=30, azim=45)
    
    # Legend
    leg = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in leg.get_texts():
        text.set_color("white")
        text.set_fontsize(14)
    
    # Master PNG Export
    print(f"Exporting final high-res master visualization to {output_png}...")
    plt.savefig(output_png, dpi=200, facecolor='black', bbox_inches='tight')
    
    # Also save to current brain directory as the master for the UI
    brain_out = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\3D_GROUND_VS_NONGROUND.png"
    plt.savefig(brain_out, dpi=200, facecolor='black', bbox_inches='tight')
    
    plt.close()
    print("3D Classification Render complete!")

if __name__ == "__main__":
    main()
