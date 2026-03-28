import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geopandas as gpd
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation, PillowWriter

def main():
    print("Initializing Multi-Angle Orbital Animation for Slab Model...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    shp_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    gif_path = os.path.join(out_dir, "rajasthan_slab_orbital_animation.gif")
    
    # 1. Load Data
    print("Loading Terrain Data...")
    with rasterio.open(dtm_path) as src:
        # Balanced resolution for smooth GIF rendering
        dtm_data = src.read(1, out_shape=(1, int(src.height / 4), int(src.width / 4)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    drainage = gpd.read_file(shp_path)
    
    # Identify Top 40 Hotspots
    with rasterio.open(acc_path) as a_src:
        acc = a_src.read(1)
        valid_acc = np.ma.masked_invalid(acc)
        flat_idx = np.argsort(valid_acc.compressed())[-40:]
        fy, fx = np.where(valid_acc.mask == False)
        hotspots = []
        for idx in flat_idx:
            r, c = fy[idx], fx[idx]
            hx, hy = a_src.transform * (c, r)
            d_row, d_col = src.index(hx, hy)
            s_r, s_c = int(d_row/4), int(d_col/4)
            if 0 <= s_r < dtm_data.shape[0] and 0 <= s_c < dtm_data.shape[1]:
                hz = dtm_data[s_r, s_c]
                if not np.isnan(hz):
                    hotspots.append((hx, hy, hz))

    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()
    
    z_exag = 2.5
    min_z = np.nanmin(dtm_data) - 40
    
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = np.flipud(dtm_data)
    
    # 3. Create Slab (Simplified for Animation performance)
    print("Building base slab...")
    # Just the 4 side faces
    for i in range(dtm_data.shape[1]-1):
        v = [[(X[0,i], Y[0,i], Z[0,i]*z_exag), (X[0,i+1], Y[0,i+1], Z[0,i+1]*z_exag), 
              (X[0,i+1], Y[0,i+1], min_z*z_exag), (X[0,i], Y[0,i], min_z*z_exag)]]
        ax.add_collection3d(Poly3DCollection(v, color='#333333', alpha=0.9))
    # ... Only adding major faces for speed in GIF ... 

    # Topo
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=plt.cm.copper, vert_exag=z_exag, blend_mode='overlay')
    surf = ax.plot_surface(X, Y, Z*z_exag, facecolors=rgb, linewidth=0, antialiased=True, rstride=2, cstride=2)

    # 4. Drainage Network
    print("Drawing drainage network...")
    for geom in drainage.geometry:
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            lx, ly = coords[:, 0], coords[:, 1]
            lz = []
            for xi, yi in zip(lx, ly):
                r, c = src.index(xi, yi)
                sr, sc = int(r/4), int(c/4)
                if 0 <= sr < dtm_data.shape[0] and 0 <= sc < dtm_data.shape[1]:
                    lz.append(dtm_data[sr, sc] * z_exag + 2)
                else: lz.append(min_z * z_exag)
            ax.plot(lx, ly, lz, color='#00FFFF', linewidth=1, alpha=0.7)

    # Hotspots
    for hx, hy, hz in hotspots:
        ax.scatter(hx, hy, hz*z_exag + 5, color='red', marker='D', s=40, zorder=20)
        # We'll omit text labels in the 360 GIF to avoid cluttering

    ax.set_title("3D Drainage Output Model: Multi-Angle Orbital Perspective", color='white', fontsize=14, fontweight='bold', pad=5)

    # 5. Complex Animation Loop (Orbital Camera)
    def update(frame):
        # azim orbits 360 degrees
        # elev oscillates from 10 to 60 for "view from top/bottom/front/back"
        azim = frame * 3
        elev = 35 + 25 * np.sin(np.radians(frame * 6))
        ax.view_init(elev=elev, azim=azim)
        if frame % 10 == 0:
            print(f"Rendering Orbital Frame {frame}/120...")
        return surf,

    print("Generating High-Clarity Multi-Angle GIF (120 frames)...")
    anim = FuncAnimation(fig, update, frames=120, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save(gif_path, writer=writer)
    
    print(f"SUCCESS! Advanced Orbital GIF saved at: {gif_path}")

if __name__ == "__main__":
    main()
