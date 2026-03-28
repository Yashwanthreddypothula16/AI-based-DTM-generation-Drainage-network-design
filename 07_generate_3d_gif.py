import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation, PillowWriter

def main():
    print("Initializing 3D GIF Generation...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    gif_path = os.path.join(out_dir, "rajasthan_3d_rotation.gif")
    
    # Setup the Plot Figure
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() # clean look
    
    # Load and downsample DTM (using / 4 for high-detail clear GIF)
    print("Loading high-detail Terrain Data for Animation...")
    with rasterio.open(dtm_path) as src:
        # Reduced downsampling from /12 to /4 for 9x more detail
        dtm_data = src.read(1, out_shape=(1, int(src.height / 4), int(src.width / 4)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    print("Building high-detail initial 3D Hillshade Mesh...")
    # Add a LightSource sun
    ls = LightSource(azdeg=315, altdeg=45)
    # Applying 3.0 vertical exaggeration for maximum visibility of drainage relief
    rgb = ls.shade(np.flipud(dtm_data), cmap=plt.cm.terrain, vert_exag=3.0, blend_mode='overlay')
    
    surf = ax.plot_surface(X, Y, np.flipud(dtm_data), facecolors=rgb, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_title("Rajasthan Village: High-Detail 360 Analysis", fontsize=18, fontweight='bold', pad=20)
    
    # Create the Animation update loop
    def update(frame):
        # Rotate azimuth angle by 3 degrees every frame (120 frames for full rotation)
        ax.view_init(elev=50, azim=frame * 3)
        if frame % 10 == 0:
            print(f"Rendering frame {frame}/120 (High-Detail)...")
        return surf,
        
    print("Rendering Ultra-Clear GIF Frames... (This will take a few minutes)")
    # 120 frames * 3 degrees = 360 full rotation
    anim = FuncAnimation(fig, update, frames=120, blit=False)
    
    # Save using Pillow at higher DPI for clarity
    writer = PillowWriter(fps=20)
    anim.save(gif_path, writer=writer)
    print(f"SUCCESS! 3D GIF successfully generated at: {gif_path}")

if __name__ == "__main__":
    main()
