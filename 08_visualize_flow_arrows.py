import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show

def main():
    print("Rendering Beautiful Quiver Flow Direction Map...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    flow_dir_path = os.path.join(out_dir, "raj_flow_dir.tif")
    
    # 1. Setup the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 2. Plot underlying Terrain DTM
    dtm = rasterio.open(dtm_path)
    show(dtm, ax=ax, cmap='terrain', alpha=0.5, title="Topological Flow Routing (Where water travels from -> to)")
    
    # 3. Process Flow Direction Raster into vector components
    with rasterio.open(flow_dir_path) as src:
        fdir = src.read(1)
        bounds = src.bounds
        transform = src.transform
        
    # Mask invalid data
    valid_mask = fdir > 0
    # Downsample drastically (step every 30 pixels) to prevent a massive block of black overlapping arrows
    step = 30
    
    # D8 Direction value mappings to unit vectors [DX, DY]
    d8_to_dx = {1:1, 2:1, 4:0, 8:-1, 16:-1, 32:-1, 64:0, 128:1}
    d8_to_dy = {1:0, 2:-1, 4:-1, 8:-1, 16:0, 32:1, 64:1, 128:1}
    
    # Create coordinate arrays matching raster dimensions
    rows, cols = fdir.shape
    r_coords, c_coords = np.mgrid[0:rows, 0:cols]
    
    # Subsample indices
    r_sub = r_coords[::step, ::step]
    c_sub = c_coords[::step, ::step]
    fdir_sub = fdir[::step, ::step]
    mask_sub = valid_mask[::step, ::step]
    
    # Calculate geographical X,Y from raster row,col
    X, Y = rasterio.transform.xy(transform, r_sub, c_sub, offset='center')
    X = np.array(X)
    Y = np.array(Y)
    
    # Map directions mapped onto downsampled grid
    U = np.zeros_like(fdir_sub, dtype=float)
    V = np.zeros_like(fdir_sub, dtype=float)
    
    for i in range(fdir_sub.shape[0]):
        for j in range(fdir_sub.shape[1]):
            val = fdir_sub[i, j]
            if mask_sub[i, j] and val in d8_to_dx:
                U[i, j] = d8_to_dx[val]
                V[i, j] = d8_to_dy[val]
            else:
                # Mask out areas with no direction
                U[i, j] = np.nan
                V[i, j] = np.nan
                
    # 4. Plot Arrows (Quiver)
    print("Drawing topological routing arrows...")
    # Adjust scale based on extent
    q = ax.quiver(X, Y, U, V, 
                  color='black',
                  scale=45,          # Smaller means bigger arrows
                  width=0.002,
                  headlength=4, 
                  headwidth=3)
                  
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
                  
    output_png = os.path.join(out_dir, "map6_flow_routing_arrows.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Quiver Map Success! Saved to {output_png}")

if __name__ == "__main__":
    main()
