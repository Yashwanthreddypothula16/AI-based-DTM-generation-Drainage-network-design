import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin

def main():
    print("Initializing robust DTM Generation using ML Ground Points...")
    
    input_las = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_ml_classified.las"
    output_dtm = r"Rajasthan_Point_Cloud\rajasthan_dtm.tif"
    
    print("Loading point cloud into memory...")
    las = laspy.read(input_las)
    
    # Extract only the ML-Predicted Ground Points (Class 2)
    ground_mask = las.classification == 2
    x = las.x[ground_mask]
    y = las.y[ground_mask]
    z = las.z[ground_mask]
    
    total_ground = len(x)
    print(f"Isolated {total_ground:,} ML Ground Points.")
    
    if total_ground == 0:
        print("ERROR: No ground points found. Check ML classifier output.")
        return

    # To avoid memory crashing during griddata (O(n^2) or high memory requirement), we sample.
    max_points = 1000000 
    if total_ground > max_points:
        indices = np.random.choice(total_ground, max_points, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        print(f"Sampled down to {max_points:,} points for ultra-detailed DTM interpolation.")

    # Create spatial grid with 1 meter resolution
    resolution = 1.0 
    print(f"Generating perfect {resolution}m high-detail grid structure...")
    
    xi = np.arange(np.min(x), np.max(x), resolution)
    yi = np.arange(np.min(y), np.max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    print("Interpolating ML DTM surface... (May take a minute)")
    dtm_zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Fill remaining NaNs (edges) using nearest 
    print("Filling edge gaps...")
    dtm_zi_nearest = griddata((x, y), z, (xi, yi), method='nearest')
    dtm_zi[np.isnan(dtm_zi)] = dtm_zi_nearest[np.isnan(dtm_zi)]
    
    print(f"Saving to {output_dtm} ...")
    transform = from_origin(np.min(x), np.max(y), resolution, resolution)
    with rasterio.open(
        output_dtm,
        "w",
        driver="GTiff",
        height=dtm_zi.shape[0],
        width=dtm_zi.shape[1],
        count=1,
        dtype=dtm_zi.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(dtm_zi, 1)

    print("Success! DTM Raster successfully saved.")

if __name__ == "__main__":
    main()
