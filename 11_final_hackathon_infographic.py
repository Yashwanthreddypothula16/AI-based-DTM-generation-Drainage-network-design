import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, LightSource
import geopandas as gpd
from rasterio.plot import show

def main():
    print("Generating Final Hackathon Presentation Infographic...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    # Paths
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    flow_acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    flow_dir_path = os.path.join(out_dir, "raj_flow_dir.tif")
    twi_path = os.path.join(out_dir, "raj_twi_flood_risk.tif")
    network_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")

    # Load Vector Network
    drainage = gpd.read_file(network_path)

    # 1. Setup Large Figure (Infographic Style)
    fig = plt.figure(figsize=(24, 16), facecolor='#f8f9fa')
    plt.suptitle("Rajasthan Village: AI-Driven Drainage & Flood Risk Analysis Dashboard", 
                 fontsize=32, fontweight='bold', y=0.95, color='#2c3e50')
    
    # Grid Layout: 2 Rows, 2 Columns
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)

    # --- PANEL A: Realistic 3D Terrain Model (Top Left) ---
    print("Rendering Panel A: 3D Terrain...")
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    with rasterio.open(dtm_path) as src:
        # Increase detail for the master infographic by using / 4 downsampling
        dtm_data = src.read(1, out_shape=(1, int(src.height / 4), int(src.width / 4)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
        y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
        X, Y = np.meshgrid(x, y)
        ls = LightSource(azdeg=315, altdeg=45)
        # Apply vertical exaggeration
        rgb = ls.shade(np.flipud(dtm_data), cmap=plt.cm.terrain, vert_exag=2.0, blend_mode='overlay')
        ax_a.plot_surface(X, Y, np.flipud(dtm_data), facecolors=rgb, linewidth=0, antialiased=True)
        ax_a.view_init(elev=50, azim=45)
        ax_a.set_axis_off()
        ax_a.set_title("A. Realistic 3D Terrain (AI Ground Classification)", fontsize=20, fontweight='bold', pad=20)

    # --- PANEL B: Classified Flood Risk Map (Top Right) ---
    print("Rendering Panel B: Flood Risk...")
    ax_b = fig.add_subplot(gs[0, 1])
    with rasterio.open(twi_path) as src:
        twi = src.read(1)
        twi = np.ma.masked_less(twi, -100)
        p80, p95 = np.percentile(twi.compressed(), [80, 95])
        risk = np.zeros_like(twi)
        risk[twi < p80] = 1; risk[twi >= p80] = 2; risk[twi >= p95] = 3
        risk = np.ma.masked_array(risk, twi.mask)
        cmap_risk = ListedColormap(['#90EE90', '#FFD700', '#FF4500'])
        img_b = ax_b.imshow(risk, cmap=cmap_risk, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
        cbar_b = plt.colorbar(img_b, ax=ax_b, ticks=[1, 2, 3], shrink=0.8)
        cbar_b.ax.set_yticklabels(['Low Risk', 'Medium', 'High Risk'])
        ax_b.set_title("B. Classified Waterlogging & Flood Risk Zones", fontsize=20, fontweight='bold', pad=20)

    # --- PANEL C: Flow Accumulation / Stream Density (Bottom Left) ---
    print("Rendering Panel C: Flow Accumulation...")
    ax_c = fig.add_subplot(gs[1, 0])
    with rasterio.open(flow_acc_path) as src:
        acc = src.read(1)
        acc[acc <= 0] = 0.1
        img_c = ax_c.imshow(acc, cmap='Blues', norm=LogNorm(vmin=10, vmax=acc.max()), 
                           extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
        plt.colorbar(img_c, ax=ax_c, shrink=0.8, label='Cells Accumulated')
        ax_c.set_title("C. Drainage Flow Concentration (Logic Layer)", fontsize=20, fontweight='bold', pad=20)

    # --- PANEL D: Final Integrated Drainage Strategy (Bottom Right) ---
    print("Rendering Panel D: Integrated Strategy...")
    ax_d = fig.add_subplot(gs[1, 1])
    with rasterio.open(dtm_path) as src:
        show(src, ax=ax_d, cmap='terrain', alpha=0.4)
        drainage.plot(ax=ax_d, color='blue', linewidth=1.5, label='Designed Drainage Lines')
        
        # Calculate Dump Points (Top 5 Flow max)
        with rasterio.open(flow_acc_path) as f_src:
            f_data = np.ma.masked_invalid(f_src.read(1))
            f_data = np.ma.masked_less(f_data, 0)
            f_flat_idx = np.argsort(f_data.compressed())[-5:]
            fy, fx = np.where(f_data.mask == False)
            for idx in f_flat_idx:
                r, c = fy[idx], fx[idx]
                px, py = f_src.transform * (c, r)
                ax_d.scatter(px, py, color='red', s=400, marker='*', edgecolors='black', label="Recommended Strategy Outlets" if idx == f_flat_idx[-1] else "")
        
        # Add a few representative flow arrows
        with rasterio.open(flow_dir_path) as dir_src:
            d_data = dir_src.read(1)
            step = 60
            r_idx, c_idx = np.mgrid[0:d_data.shape[0]:step, 0:d_data.shape[1]:step]
            dx_map = {1:1, 2:1, 4:0, 8:-1, 16:-1, 32:-1, 64:0, 128:1}
            dy_map = {1:0, 2:-1, 4:-1, 8:-1, 16:0, 32:1, 64:1, 128:1}
            Xp, Yp = rasterio.transform.xy(dir_src.transform, r_idx, c_idx)
            Up = np.array([[dx_map.get(d_data[r, c], 0) for c in range(0, d_data.shape[1], step)] for r in range(0, d_data.shape[0], step)])
            Vp = np.array([[dy_map.get(d_data[r, c], 0) for c in range(0, d_data.shape[1], step)] for r in range(0, d_data.shape[0], step)])
            ax_d.quiver(Xp, Yp, Up, Vp, color='black', alpha=0.6, scale=40, width=0.003)

        ax_d.legend(loc='upper right', fontsize=12)
        ax_d.set_title("D. Final Strategy: Connected Network & Outlets", fontsize=20, fontweight='bold', pad=20)

    # Footer Text
    plt.figtext(0.5, 0.02, "This automated AI/ML workflow classifies point clouds, generates conditioned DTMs, and designs resilient drainage networks for densely inhabited village areas.", 
                ha="center", fontsize=18, style='italic', color='#7f8c8d')

    output_png = os.path.join(out_dir, "FINAL_SUBMISSION_INFOGRAPHIC.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final Hackathon Infographic successfully saved to {output_png}")

if __name__ == "__main__":
    main()
