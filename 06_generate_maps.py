import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LightSource
from whitebox import WhiteboxTools

def create_wetness_index(wbt, base_dir):
    """Generates TWI (Flood Risk Map) correctly via whitebox"""
    print("Calculating Slope and TWI (Flood Risk Map)...")
    dtm = os.path.join(base_dir, r"Rajasthan_Point_Cloud\rajasthan_dtm.tif")
    flow_acc = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_flow_acc.tif")
    slope = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_slope.tif")
    twi = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_twi_flood_risk.tif")
    
    # Needs slope and specific catchment area
    wbt.slope(dtm, slope)
    wbt.wetness_index(slope, flow_acc, twi)
    return twi

def get_dump_zones(flow_acc_path, n_points=5):
    """Finds the maximum accumulation pour points (Outlets) to identify drainage dump destinations."""
    with rasterio.open(flow_acc_path) as src:
        acc_data = src.read(1)
        transform = src.transform
        
    # Ignore NoData
    valid_data = np.ma.masked_invalid(acc_data)
    valid_data = np.ma.masked_less(valid_data, 0)
    
    # Get N largest accumulation values points
    flat_indices = np.argsort(valid_data.compressed())[-n_points:]
    coords = []
    
    # Map valid array values back to indices, and indices to coordinates
    y_idx, x_idx = np.where(valid_data.mask == False)
    
    for idx in flat_indices:
        r, c = y_idx[idx], x_idx[idx]
        x_coord, y_coord = transform * (c, r)
        coords.append((x_coord, y_coord))
        
    return coords

def main():
    print("Generating Presentation-Quality Hydrological Maps...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    # 1. TWI calculation
    wbt = WhiteboxTools()
    wbt.set_working_dir(base_dir)
    twi_path = create_wetness_index(wbt, base_dir)
    
    # Datasets
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    flow_acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    flow_dir_path = os.path.join(out_dir, "raj_flow_dir.tif")
    network_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    
    drainage_net = gpd.read_file(network_path)
    
    print("Saving Figure 1: Flow Accumulation (Log Normal)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    with rasterio.open(flow_acc_path) as src:
        acc = src.read(1)
        acc[acc <= 0] = 0.1 # Prevent log(0)
        show(acc, ax=ax, cmap='Blues', norm=LogNorm(vmin=10, vmax=acc.max()), title='Flow Accumulation (High Density Drainage Paths)')
    plt.savefig(os.path.join(out_dir, "map1_flow_accumulation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saving Figure 2: Categorical Flow Direction...")
    fig, ax = plt.subplots(figsize=(10, 8))
    with rasterio.open(flow_dir_path) as src:
        fdir = src.read(1)
        fdir = np.ma.masked_equal(fdir, -32768)
        show(fdir, ax=ax, cmap='hsv', title='D8 Flow Direction Topography')
    plt.savefig(os.path.join(out_dir, "map2_flow_direction.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saving Figure 3: Flood Risk / Waterlogging Zones (TWI)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    with rasterio.open(twi_path) as src:
        twi_data = src.read(1)
        twi_data = np.ma.masked_less(twi_data, -100)
        show(twi_data, ax=ax, cmap='YlGnBu', title='Waterlogging Flood Risk (Topographic Wetness Index)')
        drainage_net.plot(ax=ax, color='red', alpha=0.6, linewidth=0.5)
    plt.savefig(os.path.join(out_dir, "map3_flood_risk_twi.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saving Figure 4: Drainage Map & Dump Zones...")
    fig, ax = plt.subplots(figsize=(10, 8))
    dump_points = get_dump_zones(flow_acc_path, n_points=8)
    with rasterio.open(dtm_path) as dtm:
         show(dtm, ax=ax, cmap='terrain', title='Drainage Strategy with Targeted Outlet Dump Zones')
         drainage_net.plot(ax=ax, color='blue', linewidth=1.2)
         
         # Plot pour points
         for px, py in dump_points:
             ax.scatter(px, py, color='red', s=300, marker='*', edgecolors='black', label="Recommended Dump Outlets")
         
         # Deduplicate legend
         handles, labels = ax.get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.savefig(os.path.join(out_dir, "map4_drainage_dump_zones.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saving Figure 5: Realistic 3D DTM with Hillshade...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    with rasterio.open(dtm_path) as src:
        dtm_data = src.read(1, out_shape=(1, int(src.height / 8), int(src.width / 8)))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
        x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
        y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Apply realistic LightSource (Simulating sun from North-West)
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(np.flipud(dtm_data), cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
        
        ax.plot_surface(X, Y, np.flipud(dtm_data), facecolors=rgb, linewidth=0, antialiased=False, rstride=1, cstride=1)
        ax.set_title("Realistic 3D Interpolated DTM with Hillshade Overlay")
        ax.set_axis_off()
        ax.view_init(elev=60, azim=45) 
        
    plt.savefig(os.path.join(out_dir, "map5_3D_realistic_hillshade.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All Maps Successfully Generated!")

if __name__ == "__main__":
    main()
