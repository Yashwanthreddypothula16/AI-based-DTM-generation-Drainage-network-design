import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def main():
    print("Generating Ultra-Resolution 3D Relief Visualization (600 DPI)...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    output_png = os.path.join(out_dir, "ULTRA_HD_3D_TERRAIN.png")
    
    # 1. Load the High-Detail DTM
    print("Loading high-detail terrain model...")
    with rasterio.open(dtm_path) as src:
        # No downsampling (or very minimal if grid is too massive)
        # 1.0m resolution is usually manageable for direct DTM plotting
        dtm_data = src.read(1)
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    print(f"Data size: {dtm_data.shape[0]}x{dtm_data.shape[1]} pixels.")
    
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Setup Figure HQ
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Hide axis but keep a nice "physical" border feel
    ax.set_axis_off()
    
    # 2. Enhanced Realistic Relief Shading
    print("Computing light shadows and relief shading...")
    # Sun from North-West at 45 degree altitude
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Blend topography colors with hillshading (shaded with terrain colormap)
    # Apply vertical exaggeration of 2.0 to make the terrain "pop" and look less flat
    rgb = ls.shade(np.flipud(dtm_data), cmap=plt.cm.terrain, vert_exag=2.0, blend_mode='overlay')
    
    # 3. High-Res Mesh Rendering
    print("Final rendering of the 3D surface...")
    # rstride/cstride of 1 = full resolution (no pixel skips)
    surf = ax.plot_surface(X, Y, np.flipud(dtm_data), facecolors=rgb, 
                           linewidth=0, antialiased=False, rstride=1, cstride=1)
    
    ax.set_title("Rajasthan Village: High-Definition 3D Terrain Analysis", fontsize=24, fontweight='bold', pad=30)
    
    # Best angle to see both the terrain height and the village layout
    ax.view_init(elev=45, azim=45)
    
    # 4. Save at Ultra-High DPI (600)
    print("Saving Ultra-HD Export (600 DPI)... (Taking some time)")
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    
    # Also save to current brain directory as the master for the UI
    brain_out = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\ULTRA_HD_3D_TERRAIN.png"
    plt.savefig(brain_out, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"SUCCESS! Master 3D Image saved to: {output_png}")

if __name__ == "__main__":
    main()
