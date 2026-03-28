import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.plot import show

def main():
    print("Generating 3D Visualization and 2D Drainage Overlay map...")
    
    dtm_path = r"Rajasthan_Point_Cloud\rajasthan_dtm.tif"
    network_path = r"Rajasthan_Point_Cloud\rajasthan_drainage_network.shp"
    
    # 1. Load Data
    with rasterio.open(dtm_path) as src:
        # Downsample the array by 1/4 to prevent 3D rendering engine crash (takes huge RAM)
        dtm_data = src.read(1, out_shape=(1, int(src.height / 4), int(src.width / 4)))
        # Filter NoData values (-32768)
        dtm_data[dtm_data < -1000] = np.nan
        transform = src.transform
        bounds = src.bounds
        
        # We need the full resolution reading for the 2nd (2D) plot
        full_dtm = rasterio.open(dtm_path)

    drainage_network = gpd.read_file(network_path)
    
    # Setup Figure Space
    fig = plt.figure(figsize=(20, 8))
    fig.canvas.manager.set_window_title('Rajasthan Village DTM and Drainage Analysis')
    
    # --- SUBPLOT 1: The 3D Terrain Surface ---
    # Downsampled meshgrid for speed and smoothness
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    print("Rendering 3D Surface Model...")
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, np.flipud(dtm_data), cmap='terrain', 
                            linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_title("3D Digital Terrain Model (Filtered from AI Point Cloud)", pad=15)
    ax1.set_axis_off()  # Hide axis ticks for a cleaner geographic look
    
    # View angle: Elevated looking over the village DTM
    ax1.view_init(elev=50, azim=45) 
    
    # --- SUBPLOT 2: The 2D Overlay (Clear Drainage View) ---
    print("Rendering Clear 2D Vector Drainage Map...")
    ax2 = fig.add_subplot(1, 2, 2)
    # Plot base terrain shading
    show(full_dtm, ax=ax2, cmap='terrain', title="Clear 2D Drone Drainage Map Overlay")
    # Overlay vector lines in stark contrasting Red
    drainage_network.plot(ax=ax2, color='red', linewidth=1.5, zorder=2)
    ax2.set_xlabel("Easting / Longitude")
    ax2.set_ylabel("Northing / Latitude")
    
    plt.tight_layout()
    
    # Save output snapshot to Artifact directory for the UI
    output_img = r"C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\final_rajasthan_vis.png"
    print(f"Saving final visual to {output_img}")
    plt.savefig(output_img, dpi=200, bbox_inches='tight')
    print("Visualization processing complete!")

if __name__ == "__main__":
    main()
