import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd

def main():
    print("Generating Enhanced Flood Risk Map...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    twi_path = os.path.join(out_dir, "raj_twi_flood_risk.tif")
    network_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")

    # Load TWI data
    with rasterio.open(twi_path) as src:
        twi = src.read(1)
        # Mask nodata
        twi = np.ma.masked_less(twi, -100)
        transform = src.transform

    # Define Risk Categories based on TWI percentiles
    # Usually, high TWI means high flood risk
    valid_twi = twi.compressed()
    p80 = np.percentile(valid_twi, 80)
    p95 = np.percentile(valid_twi, 95)

    risk_map = np.zeros_like(twi)
    risk_map[twi < p80] = 1   # Low Risk
    risk_map[twi >= p80] = 2  # Medium Risk
    risk_map[twi >= p95] = 3  # High Risk
    risk_map = np.ma.masked_array(risk_map, twi.mask)

    # Load Drainage Network
    drainage = gpd.read_file(network_path)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom Colormap: Green (Low), Yellow (Med), Red (High)
    cmap = ListedColormap(['#90EE90', '#FFD700', '#FF4500'])
    
    img = ax.imshow(risk_map, cmap=cmap, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
    
    # Add Drainage Overlay for context
    drainage.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.5, label='Natural Drainage Paths')
    
    # Add Legend/Colorbar labels
    cbar = plt.colorbar(img, ax=ax, ticks=[1, 2, 3], shrink=0.7)
    cbar.ax.set_yticklabels(['Low Risk', 'Medium Risk', 'High Risk (Flood Zone)'])
    
    ax.set_title("Rajasthan Village: Classified Flood Risk Zones", fontsize=16, pad=20)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    
    output_png = os.path.join(out_dir, "map7_enhanced_flood_risk.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced Flood Risk Map successfully saved to {output_png}")

if __name__ == "__main__":
    main()
