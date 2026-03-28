import numpy as np
import rasterio
from scipy.ndimage import sobel

# -------------------------------
# STEP 1: Load DTM
# -------------------------------
with rasterio.open("dtm.tif") as src:
    dtm = src.read(1)
    profile = src.profile
    transform = src.transform

print("DTM Loaded")

# -------------------------------
# STEP 2: Calculate Slope
# -------------------------------
dx = sobel(dtm, axis=1)
dy = sobel(dtm, axis=0)

slope = np.sqrt(dx**2 + dy**2)

print("Slope calculated")

# -------------------------------
# STEP 3: Simple Flow Accumulation
# -------------------------------
flow_acc = np.zeros_like(dtm)

for i in range(1, dtm.shape[0] - 1):
    for j in range(1, dtm.shape[1] - 1):
        neighbors = dtm[i-1:i+2, j-1:j+2]
        flow_acc[i, j] = np.sum(neighbors > dtm[i, j])

print("Flow accumulation estimated")

# -------------------------------
# STEP 4: Extract Streams
# -------------------------------
stream_threshold = 5
streams = flow_acc > stream_threshold

print("Streams extracted")

# -------------------------------
# STEP 5: Save Stream Raster
# -------------------------------
profile.update(dtype=rasterio.uint8)

with rasterio.open("streams.tif", "w", **profile) as dst:
    dst.write(streams.astype(rasterio.uint8), 1)

print("Drainage network saved as streams.tif")