import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geopandas as gpd
from matplotlib.colors import LightSource

def main():
    print("Generating Advanced 3D Drainage Output Model (Slab Edition)...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    shp_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    output_png = os.path.join(out_dir, "ADVANCED_3D_DRAINAGE_SLAB_MODEL.png")
    
    # 1. Load Data
    print("Loading Terrain and Flow Data...")
    with rasterio.open(dtm_path) as src:
        # Use a balanced resolution for the 3D mesh + labels
        dtm_data = src.read(1, out_shape=(1, int(src.height / 3), int(src.width / 3)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    drainage = gpd.read_file(shp_path)
    
    with rasterio.open(acc_path) as a_src:
        acc = a_src.read(1)
        # Identify Top 60 Hotspots for dense labeling
        valid_acc = np.ma.masked_invalid(acc)
        flat_idx = np.argsort(valid_acc.compressed())[-60:]
        fy, fx = np.where(valid_acc.mask == False)
        hotspots = []
        for idx in flat_idx:
            r, c = fy[idx], fx[idx]
            hx, hy = a_src.transform * (c, r)
            # Find closest Z in DTM
            d_row, d_col = src.index(hx, hy)
            if 0 <= d_row < dtm_data.shape[0]*3 and 0 <= d_col < dtm_data.shape[1]*3:
                # Use the loaded DTM sample for Z coordination
                s_r, s_c = int(d_row/3), int(d_col/3)
                if 0 <= s_r < dtm_data.shape[0] and 0 <= s_c < dtm_data.shape[1]:
                    hz = dtm_data[s_r, s_c]
                    if not np.isnan(hz):
                        hotspots.append((hx, hy, hz))

    # 2. Setup 3D Canvas (Dark Theme)
    fig = plt.figure(figsize=(24, 16), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()
    
    z_exag = 2.5
    min_z = np.nanmin(dtm_data) - 50 # Base of the slab
    
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = np.flipud(dtm_data)
    
    # 3. Create the "Slab" Base (The Skirt)
    print("Constructing physical Slab Base...")
    # Add vertical sides to create a "Block" look
    # North side
    for i in range(dtm_data.shape[1]-1):
        v = [[(X[0,i], Y[0,i], Z[0,i]*z_exag), (X[0,i+1], Y[0,i+1], Z[0,i+1]*z_exag), 
              (X[0,i+1], Y[0,i+1], min_z*z_exag), (X[0,i], Y[0,i], min_z*z_exag)]]
        ax.add_collection3d(Poly3DCollection(v, color='#333333', alpha=0.8))
    # South side 
    for i in range(dtm_data.shape[1]-1):
        v = [[(X[-1,i], Y[-1,i], Z[-1,i]*z_exag), (X[-1,i+1], Y[-1,i+1], Z[-1,i+1]*z_exag), 
              (X[-1,i+1], Y[-1,i+1], min_z*z_exag), (X[-1,i], Y[-1,i], min_z*z_exag)]]
        ax.add_collection3d(Poly3DCollection(v, color='#333333', alpha=0.8))
    # West side
    for i in range(dtm_data.shape[0]-1):
        v = [[(X[i,0], Y[i,0], Z[i,0]*z_exag), (X[i+1,0], Y[i+1,0], Z[i+1,0]*z_exag), 
              (X[i+1,0], Y[i+1,0], min_z*z_exag), (X[i,0], Y[i,0], min_z*z_exag)]]
        ax.add_collection3d(Poly3DCollection(v, color='#222222', alpha=0.8))
    # East side
    for i in range(dtm_data.shape[0]-1):
        v = [[(X[i,-1], Y[i,-1], Z[i,-1]*z_exag), (X[i+1,-1], Y[i+1,-1], Z[i+1,-1]*z_exag), 
              (X[i+1,-1], Y[i+1,-1], min_z*z_exag), (X[i,-1], Y[i,-1], min_z*z_exag)]]
        ax.add_collection3d(Poly3DCollection(v, color='#222222', alpha=0.8))

    # Bottom Slab
    v_bot = [[(bounds.left, bounds.bottom, min_z*z_exag), (bounds.right, bounds.bottom, min_z*z_exag), 
               (bounds.right, bounds.top, min_z*z_exag), (bounds.left, bounds.top, min_z*z_exag)]]
    ax.add_collection3d(Poly3DCollection(v_bot, color='#111111', alpha=0.9))

    # 4. Render Topography with Hillshading
    print("Rendering High-Detail Topography...")
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=plt.cm.copper, vert_exag=z_exag, blend_mode='overlay')
    surf = ax.plot_surface(X, Y, Z*z_exag, facecolors=rgb, linewidth=0, antialiased=True, rstride=1, cstride=1)

    # 5. Add Teal Drainage Overlay
    print("Overlaying Teal Drainage Network...")
    for geom in drainage.geometry:
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            lx, ly = coords[:, 0], coords[:, 1]
            # Sample Z for flow path
            lz = []
            for xi, yi in zip(lx, ly):
                r, c = src.index(xi, yi)
                # Map to sample indices for Z lookup
                sr, sc = int(r/3), int(c/3)
                if 0 <= sr < dtm_data.shape[0] and 0 <= sc < dtm_data.shape[1]:
                    lz.append(dtm_data[sr, sc] * z_exag + 2) # Slightly elevated for visibility
                else: lz.append(min_z * z_exag)
            ax.plot(lx, ly, lz, color='#00FFFF', linewidth=2, alpha=0.9, zorder=10)

    # 6. Add "HOTSPOT" Labels & Markers
    print("Labeling Hotspots...")
    for hx, hy, hz in hotspots:
        ax.scatter(hx, hy, hz*z_exag + 5, color='red', marker='D', s=100, zorder=20, edgecolors='white')
        ax.text(hx, hy, hz*z_exag + 15, "HOTSPOT", color='white', fontsize=7, fontweight='bold', ha='center', zorder=25)

    ax.set_title("3D Drainage Output Model: Physical Slab Edition", color='white', fontsize=28, fontweight='bold', pad=40)
    
    # Matching the angle in the monitor shots
    ax.view_init(elev=40, azim=40)
    
    print(f"Exporting final slab viz to {output_png}...")
    plt.savefig(output_png, dpi=200, facecolor='black', bbox_inches='tight')
    
    # Copy to artifacts
    brain_out = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\ADVANCED_3D_DRAINAGE_SLAB_MODEL.png"
    plt.savefig(brain_out, dpi=200, facecolor='black', bbox_inches='tight')
    
    plt.close()
    print("Advanced Slab Model render complete!")

if __name__ == "__main__":
    main()
