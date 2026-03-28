

import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin

# -------------------------------
# STEP 1: Load LAS file
# -------------------------------

las_file = "Punjab_Point_Cloud/Dhal_Hoshiarpur_31235.las"   # <<< CHANGE to your LAS filename
las = laspy.read(las_file)

print("Total points:", len(las.x))
print("Unique classes in LAS file:", np.unique(las.classification))

# -------------------------------
# STEP 2: Estimate Ground Points
# -------------------------------
# Since your file is class 0 only, we estimate ground
threshold = np.percentile(las.z, 10)   # lowest 10% as ground

ground_mask = las.z <= threshold

x = las.x[ground_mask]
y = las.y[ground_mask]
z = las.z[ground_mask]

print("Estimated ground points:", len(x))

if len(x) == 0:
    raise ValueError("No ground points found. Adjust percentile.")

# -------------------------------
# STEP 3: Reduce Size (Important)
# -------------------------------
# Prevent memory crash
max_points = 100000

if len(x) > max_points:
    x = x[:max_points]
    y = y[:max_points]
    z = z[:max_points]
    print("Reduced to", max_points, "points for interpolation")

# -------------------------------
# STEP 4: Create Grid
# -------------------------------

resolution = 1  # 1 meter grid

xi = np.arange(min(x), max(x), resolution)
yi = np.arange(min(y), max(y), resolution)
xi, yi = np.meshgrid(xi, yi)

print("Interpolating... (this may take time)")

zi = griddata((x, y), z, (xi, yi), method='linear')

# Fill empty values
zi[np.isnan(zi)] = np.nanmean(zi)

# -------------------------------
# STEP 5: Save as GeoTIFF
# -------------------------------

transform = from_origin(min(x), max(y), resolution, resolution)

with rasterio.open(
    "dtm.tif",
    "w",
    driver="GTiff",
    height=zi.shape[0],
    width=zi.shape[1],
    count=1,
    dtype=zi.dtype,
    crs="EPSG:4326",   # Change if needed
    transform=transform,
) as dst:
    dst.write(zi, 1)

print("DTM created successfully! File saved as dtm.tif")