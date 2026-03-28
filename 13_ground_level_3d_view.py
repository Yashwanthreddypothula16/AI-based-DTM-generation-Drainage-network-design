import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def main():
    print("Generating Ground-Level Perspective 3D Render...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    output_png = os.path.join(out_dir, "GROUND_LEVEL_3D_VIEW.png")
    
    # 1. Load the High-Detail DTM
    with rasterio.open(dtm_path) as src:
        # Use high detail for ground level
        dtm_data = src.read(1, out_shape=(1, int(src.height / 2), int(src.width / 2)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Setup Figure HQ
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    
    # 2. Enhanced Realistic Relief Shading
    ls = LightSource(azdeg=315, altdeg=45)
    # High vertical exaggeration to make the "ground level" feel more dramatic
    rgb = ls.shade(np.flipud(dtm_data), cmap=plt.cm.terrain, vert_exag=3.0, blend_mode='overlay')
    
    # 3. Low-Angle Rendering
    print("Final rendering of the ground-level perspective...")
    surf = ax.plot_surface(X, Y, np.flipud(dtm_data), facecolors=rgb, 
                           linewidth=0, antialiased=True, rstride=1, cstride=1)
    
    ax.set_title("Rajasthan Village: Ground-Level Topography Perspective", fontsize=20, fontweight='bold', pad=20)
    
    # CRITICAL: Set low elevation (10 degrees) to simulate walking on the ground
    # and azimuth to look diagonally across the village
    ax.view_init(elev=10, azim=225)
    
    # 4. Save
    print("Saving High-Res Ground View Export...")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    
    # Copy to artifacts directory
    brain_out = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\GROUND_LEVEL_3D_VIEW.png"
    plt.savefig(brain_out, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"SUCCESS! Ground level image saved to: {output_png}")

if __name__ == "__main__":
    main()
