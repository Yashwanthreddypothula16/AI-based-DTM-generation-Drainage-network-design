import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

# -------------------------------
# STEP 1: Load streams raster
# -------------------------------
with rasterio.open("streams.tif") as src:
    image = src.read(1)
    transform = src.transform
    crs = src.crs

print("Streams raster loaded")

# -------------------------------
# STEP 2: Extract shapes
# -------------------------------
geoms = []
values = []

for geom, value in shapes(image, transform=transform):
    if value == 1:
        geoms.append(shape(geom))
        values.append(value)

print("Vector features found:", len(geoms))

# -------------------------------
# STEP 3: Create GeoDataFrame
# -------------------------------
gdf = gpd.GeoDataFrame(
    {"value": values},
    geometry=geoms,
    crs=crs
)

# -------------------------------
# STEP 4: Save as Shapefile
# -------------------------------
gdf.to_file("streams.shp")

print("Drainage shapefile saved as streams.shp")