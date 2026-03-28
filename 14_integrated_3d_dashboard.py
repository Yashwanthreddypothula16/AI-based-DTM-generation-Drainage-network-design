import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
import rasterio
from scipy.interpolate import interp2d

def main():
    print("Generating Integrated 3D Point Cloud & Drainage Dashboard...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    las_path = os.path.join(out_dir, "67169_5NKR_CHAKHIRASINGH_ml_classified.las")
    shp_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    flow_acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    
    # 1. Load and Sample Point Cloud
    print("Loading Classified Point Cloud...")
    las = laspy.read(las_path)
    
    # Stratified sample: 80% ground, 20% non-ground for context
    ground_idx = np.where(las.classification == 2)[0]
    other_idx = np.where(las.classification != 2)[0]
    
    n_points = 150000 
    s_ground = np.random.choice(ground_idx, int(n_points * 0.8), replace=False)
    s_other = np.random.choice(other_idx, int(n_points * 0.2), replace=False)
    idx = np.concatenate([s_ground, s_other])
    
    px, py, pz = las.x[idx], las.y[idx], las.z[idx]
    p_classification = las.classification[idx]
    
    # 2. Load Drainage Vector & DTM for Z-lifting
    print("Lifting 2D Drainage Network into 3D...")
    drainage = gpd.read_file(shp_path)
    with rasterio.open(dtm_path) as src:
        dtm_data = src.read(1)
        transform = src.transform
        dtm_bounds = src.bounds

    # Helper function to get Z from DTM coordinates
    def get_z_height(x, y):
        row, col = src.index(x, y)
        if 0 <= row < dtm_data.shape[0] and 0 <= col < dtm_data.shape[1]:
            val = dtm_data[row, col]
            return val if val > -1000 else np.nan
        return np.nan

    # 3. Load Hotspots (Pour Points)
    print("Identifying Drainage Hotspots...")
    with rasterio.open(flow_acc_path) as f_src:
        f_data = np.ma.masked_invalid(f_src.read(1))
        f_data = np.ma.masked_less(f_data, 0)
        # Get top 20 discharge points
        f_idx = np.argsort(f_data.compressed())[-20:]
        fy, fx = np.where(f_data.mask == False)
        hotspots = []
        for h_idx in f_idx:
            r, c = fy[h_idx], fx[h_idx]
            hx, hy = f_src.transform * (c, r)
            hz = get_z_height(hx, hy)
            if not np.isnan(hz):
                hotspots.append((hx, hy, hz))

    # 4. Setup 3D Plot with Dark Theme
    print("Rendering 3D Integrated Dashboard...")
    fig = plt.figure(figsize=(24, 16), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Global Vertical Exaggeration
    z_exag = 3.0
    
    # --- LAYER 1: Point Cloud (Glow effect) ---
    # Points colored by elevation using a glowing colormap (Inferno)
    sc = ax.scatter(px, py, pz * z_exag, c=pz, s=1, cmap='inferno', alpha=0.3, depthshade=False)
    
    # --- LAYER 2: 3D Drainage Flow ---
    for geom in drainage.geometry:
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            line_x = coords[:, 0]
            line_y = coords[:, 1]
            line_z = np.array([get_z_height(x, y) for x, y in zip(line_x, line_y)]) * z_exag
            ax.plot(line_x, line_y, line_z, color='cyan', linewidth=2, alpha=0.8, zorder=10)

    # --- LAYER 3: HOTSPOT Labels (Red Diamonds) ---
    for hx, hy, hz in hotspots:
        ax.scatter(hx, hy, hz * z_exag, color='red', marker='D', s=150, zorder=20, edgecolors='white', linewidth=1.5)
        ax.text(hx, hy, (hz * z_exag) + 15, "HOTSPOT", color='white', fontsize=9, fontweight='bold', ha='center')

    # Formatting 
    ax.set_title("3D Drainage Output Model: Integrated Point Cloud & Flow Strategy", color='white', fontsize=28, fontweight='bold', pad=40)
    ax.set_axis_off()
    
    # View Angle
    ax.view_init(elev=35, azim=45)
    
    # Master PNG Export
    output_png = os.path.join(out_dir, "MASTER_INTEGRATED_3D_DASHBOARD.png")
    print(f"Exporting final high-res master viz to {output_png}...")
    plt.savefig(output_png, dpi=200, facecolor='black', bbox_inches='tight')
    
    # Also save to current brain directory as the master for the UI
    brain_out = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\MASTER_INTEGRATED_3D_DASHBOARD.png"
    plt.savefig(brain_out, dpi=200, facecolor='black', bbox_inches='tight')
    
    plt.close()
    print("Integrated 3D Dashboard render complete!")

if __name__ == "__main__":
    main()
