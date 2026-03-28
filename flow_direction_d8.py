import numpy as np
import rasterio

# -------------------------------
# STEP 1: Load DTM
# -------------------------------
with rasterio.open("dtm.tif") as src:
    dtm = src.read(1)
    profile = src.profile
    transform = src.transform

print("DTM loaded")

rows, cols = dtm.shape
flow_dir = np.zeros((rows, cols), dtype=np.uint8)

# D8 direction codes
directions = [
    (-1, -1, 32), (-1, 0, 64), (-1, 1, 128),
    (0, -1, 16),               (0, 1, 1),
    (1, -1, 8),  (1, 0, 4),    (1, 1, 2)
]

# -------------------------------
# STEP 2: Compute D8
# -------------------------------
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        current = dtm[i, j]
        max_drop = 0
        direction_code = 0

        for di, dj, code in directions:
            neighbor = dtm[i + di, j + dj]
            drop = current - neighbor

            if drop > max_drop:
                max_drop = drop
                direction_code = code

        flow_dir[i, j] = direction_code

print("Flow direction computed")

# -------------------------------
# STEP 3: Save Output
# -------------------------------
profile.update(dtype=rasterio.uint8)

with rasterio.open("flow_direction.tif", "w", **profile) as dst:
    dst.write(flow_dir, 1)

print("Saved as flow_direction.tif")