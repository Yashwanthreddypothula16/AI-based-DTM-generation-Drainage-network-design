import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.plot import show

def main():
    print("Loading data...")
    dtm = rasterio.open('dtm_filled.tif')
    streams = gpd.read_file('network_vector.shp')
    
    # Check if twi_hotspots.tif exists
    try:
        twi = rasterio.open('twi_hotspots.tif')
    except Exception as e:
        twi = None

    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: DTM + Vector Streams
    ax1 = axes[0]
    dtm_img = show(dtm, ax=ax1, cmap='terrain', title='Digital Terrain Model + Connected Drainage Network')
    # Filter streams slightly if they are too heavy, or just plot
    streams.plot(ax=ax1, color='blue', linewidth=1.5, alpha=0.8)
    
    # Plot 2: TWI (Hotspots) + Streams
    ax2 = axes[1]
    if twi:
        # Plot Topographic Wetness Index using a suited colormap
        show(twi, ax=ax2, cmap='Blues', title='Waterlogging Hotspots (Topographic Wetness Index)')
        streams.plot(ax=ax2, color='red', linewidth=1, alpha=0.7)
    else:
        # Fallback to flow accumulation if TWI was somehow not generated
        facc = rasterio.open('flow_acc_wbt.tif')
        show(facc, ax=ax2, cmap='Blues', title='Flow Accumulation')
        streams.plot(ax=ax2, color='red', linewidth=1, alpha=0.7)

    # Save to Artifacts directory!
    output_path = r'C:\Users\nanir\.gemini\antigravity\brain\4c7b4fe5-df86-4403-a868-4613a865d8ad\results_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    main()
