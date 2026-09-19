import os
import laspy
import rasterio
import geopandas as gpd


def get_las_metadata(filepath):
    """
    Reads metadata from a LAS or LAZ file without loading all points into memory.
    Extremely fast and memory-efficient.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with laspy.open(filepath) as fh:
            header = fh.header
            min_x, min_y, min_z = header.mins
            max_x, max_y, max_z = header.maxs
            point_count = header.point_count
            file_size = os.path.getsize(filepath)
            return {
                "filename": os.path.basename(filepath),
                "point_count": point_count,
                "bounds": {
                    "min_x": min_x,
                    "min_y": min_y,
                    "min_z": min_z,
                    "max_x": max_x,
                    "max_y": max_y,
                    "max_z": max_z,
                },
                "file_size_mb": file_size / (1024 * 1024),
                "version": f"{header.version}",
            }
    except Exception as e:
        print(f"Error reading LAS metadata for {filepath}: {e}")
        return None


def parse_stats_file(filepath):
    """
    Parses classification statistics from text files.
    """
    if not os.path.exists(filepath):
        return None
    stats = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    stats[k.strip()] = int(v.strip())
        return stats
    except Exception as e:
        print(f"Error parsing stats file {filepath}: {e}")
        return None


def get_raster_metadata(filepath):
    """
    Reads raster metadata (bounds, resolution, dimensions) from a GeoTIFF.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            res = src.res
            width = src.width
            height = src.height
            crs = src.crs
            return {
                "bounds": {
                    "min_x": bounds.left,
                    "max_x": bounds.right,
                    "min_y": bounds.bottom,
                    "max_y": bounds.top,
                },
                "resolution_x": res[0],
                "resolution_y": res[1],
                "width": width,
                "height": height,
                "crs": crs.to_string() if crs else "Unknown",
            }
    except Exception as e:
        print(f"Error reading raster metadata for {filepath}: {e}")
        return None


def get_shapefile_metadata(filepath):
    """
    Reads spatial feature counts and geometry metadata from a shapefile.
    """
    if not os.path.exists(filepath):
        return None
    try:
        gdf = gpd.read_file(filepath)
        return {
            "feature_count": len(gdf),
            "geometry_type": gdf.geom_type.iloc[0] if len(gdf) > 0 else "Unknown",
            "crs": gdf.crs.to_string() if gdf.crs else "Unknown",
        }
    except Exception as e:
        print(f"Error reading shapefile metadata for {filepath}: {e}")
        return None
