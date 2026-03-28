import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rasterio.plot import show

def main():
    print("Generating Detailed Flow Accumulation Insight Map...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    flow_acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")

    # Load data
    with rasterio.open(flow_acc_path) as src:
        acc = src.read(1)
        # Handle nodata/zero for log plot
        acc[acc <= 0] = 0.1
        transform = src.transform
        bounds = src.bounds

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Base DTM for context (faint)
    with rasterio.open(dtm_path) as dtm_src:
        show(dtm_src, ax=ax, cmap='Greys', alpha=0.3)

    # Detailed Flow Accumulation with Log scale for better visibility of small streams
    # Darker blue = higher accumulation
    img = ax.imshow(acc, cmap='Blues', norm=LogNorm(vmin=10, vmax=acc.max()), 
                   extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), alpha=0.8)
    
    # Annotate high-accumulation zones
    # Find the single highest accumulation point for an example label
    y_max, x_max = np.unravel_index(np.argmax(acc), acc.shape)
    x_coord, y_coord = transform * (x_max, y_max)
    
    ax.annotate('Main Drainage Sink (Highest Accumulation)', 
                xy=(x_coord, y_coord), xytext=(x_coord + 50, y_coord + 50),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Add descriptive legend/colorbar
    cbar = plt.colorbar(img, ax=ax, shrink=0.7)
    cbar.set_label('Accumulation (Number of Upstream Cells)', fontsize=12)
    
    ax.set_title("Detailed Flow Accumulation: How Water Concentrates Across the Terrain", fontsize=16, pad=20)
    ax.text(bounds.left + 20, bounds.bottom + 20, 
            "Light Blue: Minor surface runoff\nMid Blue: Secondary drainage channels\nDark Blue: Primary streams & discharge outlets", 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    output_png = os.path.join(out_dir, "map8_detailed_accumulation.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed Accumulation Insight Map successfully saved to {output_png}")

if __name__ == "__main__":
    main()
