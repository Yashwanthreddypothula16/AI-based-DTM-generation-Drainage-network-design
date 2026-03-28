import numpy as np
import rasterio

# -------------------------------
# STEP 1: Load Flow Direction
# -------------------------------
with rasterio.open("flow_direction.tif") as src:
    flow_dir = src.read(1)
    profile = src.profile

print("Flow direction loaded")

rows, cols = flow_dir.shape
flow_acc = np.ones((rows, cols), dtype=np.float32)

# D8 direction mapping (reverse lookup)
direction_map = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

# -------------------------------
# STEP 2: Accumulation Algorithm
# -------------------------------
for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        direction = flow_dir[i, j]

        if direction in direction_map:
            di, dj = direction_map[direction]
            ni = i + di
            nj = j + dj

            if 0 <= ni < rows and 0 <= nj < cols:
                flow_acc[ni, nj] += flow_acc[i, j]

print("Flow accumulation computed")

# -------------------------------
# STEP 3: Save Output
# -------------------------------
profile.update(dtype=rasterio.float32)

with rasterio.open("flow_accumulation.tif", "w", **profile) as dst:
    dst.write(flow_acc, 1)

print("Saved as flow_accumulation.tif")