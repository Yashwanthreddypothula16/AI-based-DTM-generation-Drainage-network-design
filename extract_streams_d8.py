import rasterio
import numpy as np

with rasterio.open("flow_accumulation.tif") as src:
    flow_acc = src.read(1)
    profile = src.profile

threshold = 50  # adjust this if needed

streams = flow_acc > threshold

profile.update(dtype=rasterio.uint8)

with rasterio.open("streams_d8.tif", "w", **profile) as dst:
    dst.write(streams.astype(np.uint8), 1)

print("Real D8 streams saved as streams_d8.tif")