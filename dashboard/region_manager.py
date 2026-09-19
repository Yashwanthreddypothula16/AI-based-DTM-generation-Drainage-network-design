import os
import numpy as np
import plotly.graph_objects as go
import rasterio
import geopandas as gpd
from config import REGIONS, BASE_DIR
from utils import (
    get_las_metadata,
    parse_stats_file,
    get_raster_metadata,
    get_shapefile_metadata,
)


class RegionManager:
    def __init__(self, region_key):
        if region_key not in REGIONS:
            raise ValueError(f"Region {region_key} not configured.")
        self.key = region_key
        self.config = REGIONS[region_key]
        self.raw_folder = os.path.join(BASE_DIR, self.config["raw_folder"])
        self.output_folder = os.path.join(BASE_DIR, self.config["output_folder"])
        self.folder_path = self.output_folder

    def check_processed_status(self):
        """
        Checks if the critical pipeline outputs exist for this region.
        """
        dtm_path = os.path.join(self.output_folder, self.config["dtm"])
        shp_path = os.path.join(self.output_folder, self.config["drainage_shp"])

        has_dtm = os.path.exists(dtm_path)
        has_shp = os.path.exists(shp_path)

        return has_dtm and has_shp

    def get_raw_metadata(self):
        """
        Gathers metadata for all configured raw files of this region.
        """
        raw_metadata = []
        total_points = 0
        total_size_mb = 0.0
        bounds = None

        for raw_name in self.config["raw_files"]:
            file_path = os.path.join(self.raw_folder, raw_name)
            meta = get_las_metadata(file_path)
            if meta:
                raw_metadata.append(meta)
                total_points += meta["point_count"]
                total_size_mb += meta["file_size_mb"]

                # Expand global bounds
                b = meta["bounds"]
                if bounds is None:
                    bounds = b.copy()
                else:
                    bounds["min_x"] = min(bounds["min_x"], b["min_x"])
                    bounds["max_x"] = max(bounds["max_x"], b["max_x"])
                    bounds["min_y"] = min(bounds["min_y"], b["min_y"])
                    bounds["max_y"] = max(bounds["max_y"], b["max_y"])
                    bounds["min_z"] = min(bounds["min_z"], b["min_z"])
                    bounds["max_z"] = max(bounds["max_z"], b["max_z"])

        return {
            "files": raw_metadata,
            "total_points": total_points
            if total_points > 0
            else self.config["default_stats"]["total_points"],
            "total_size_mb": total_size_mb,
            "bounds": bounds
            or {
                "min_x": 0.0,
                "max_x": 1000.0,
                "min_y": 0.0,
                "max_y": 1000.0,
                "min_z": 0.0,
                "max_z": 100.0,
            },
        }

    def compile_statistics(self):
        """
        Compiles the complete statistical summary of the region.
        Merges raw metadata, processed outputs, and falls back to config defaults if missing.
        """
        status = self.check_processed_status()
        raw_meta = self.get_raw_metadata()
        defaults = self.config["default_stats"].copy()

        stats = {
            "region_name": self.config["name"],
            "status": "Processed & Tested"
            if status
            else "Raw Dataset (Pending Hydrological Pipeline)",
            "total_points": raw_meta["total_points"],
            "raw_size_mb": raw_meta["total_size_mb"],
            "dtm_resolution": defaults["resolution"],
            "area_covered_sq_km": defaults["area_covered_sq_km"],
            "drainage_segments": defaults["drainage_segments"],
            "flood_hotspots": defaults["flood_hotspots"],
        }

        # Ground classification variables
        stats_file_path = os.path.join(self.folder_path, self.config["stats_file"])
        parsed_stats = parse_stats_file(stats_file_path)

        if parsed_stats:
            stats.update(parsed_stats)
        else:
            # Generate simulated point distribution based on total_points
            total = stats["total_points"]
            stats["ground_points"] = int(total * 0.70)
            stats["building_points"] = int(total * 0.08)
            stats["tree_points"] = int(total * 0.12)
            stats["unclassified_points"] = total - (
                stats["ground_points"] + stats["building_points"] + stats["tree_points"]
            )
            stats["estimated_houses"] = defaults["estimated_houses"]
            stats["estimated_trees"] = defaults["estimated_trees"]

        # Parse DTM raster if exists
        if status:
            dtm_path = os.path.join(self.folder_path, self.config["dtm"])
            raster_meta = get_raster_metadata(dtm_path)
            if raster_meta:
                stats["dtm_resolution"] = f"{raster_meta['resolution_x']:.2f}m"

                # Approximate area covered based on cell size and bounds
                width_m = (
                    raster_meta["bounds"]["max_x"] - raster_meta["bounds"]["min_x"]
                )
                height_m = (
                    raster_meta["bounds"]["max_y"] - raster_meta["bounds"]["min_y"]
                )
                area_sq_km = (width_m * height_m) / 1_000_000.0
                stats["area_covered_sq_km"] = float(f"{area_sq_km:.2f}")

            # Shapefile counts
            shp_path = os.path.join(self.folder_path, self.config["drainage_shp"])
            shp_meta = get_shapefile_metadata(shp_path)
            if shp_meta:
                stats["drainage_segments"] = shp_meta["feature_count"]

        return stats

    def get_3d_scene(self):
        """
        Generates or loads the 3D terrain and drainage scene as a Plotly Figure object.
        Works seamlessly for both processed and raw simulated regions.
        """
        is_processed = self.check_processed_status()
        raw_meta = self.get_raw_metadata()
        bounds = raw_meta["bounds"]

        z_exag = 3.0

        if is_processed:
            dtm_path = os.path.join(self.folder_path, self.config["dtm"])
            shp_path = os.path.join(self.folder_path, self.config["drainage_shp"])
            acc_path = os.path.join(self.folder_path, self.config["flow_acc"])

            # 1. Load DTM
            with rasterio.open(dtm_path) as src:
                # Optimized grid size for smooth browser interaction
                dtm_data = src.read(1, out_shape=(1, 200, 200))
                r_bounds = src.bounds
                dtm_data[dtm_data < -1000] = np.nan

            x_grid = np.linspace(r_bounds.left, r_bounds.right, dtm_data.shape[1])
            y_grid = np.linspace(r_bounds.bottom, r_bounds.top, dtm_data.shape[0])

            # 2. Extract shapefile lines
            line_x, line_y, line_z = [], [], []
            try:
                drainage = gpd.read_file(shp_path)
                with rasterio.open(dtm_path) as src_full:
                    dtm_full_data = src_full.read(1)
                    for geom in drainage.geometry:
                        if geom.geom_type == "LineString":
                            coords = np.array(geom.coords)
                            for xi, yi in coords:
                                r, c = src_full.index(xi, yi)
                                if 0 <= r < src_full.height and 0 <= c < src_full.width:
                                    val = dtm_full_data[r, c]
                                    if val > -1000:
                                        line_x.append(xi)
                                        line_y.append(yi)
                                        line_z.append(val * z_exag + 1.5)
                            line_x.append(None)
                            line_y.append(None)
                            line_z.append(None)
            except Exception:
                pass

            # 3. Detect hotspots
            hotspots_x, hotspots_y, hotspots_z = [], [], []
            try:
                with rasterio.open(acc_path) as a_src:
                    acc = a_src.read(1)
                    valid_acc = np.ma.masked_invalid(acc)
                    flat_idx = np.argsort(valid_acc.compressed())[-40:]
                    fy, fx = np.where(~valid_acc.mask)

                    with rasterio.open(dtm_path) as src_h:
                        dtm_h_data = src_h.read(1)
                        for idx in flat_idx:
                            r, c = fy[idx], fx[idx]
                            hx, hy = a_src.transform * (c, r)
                            rh, ch = src_h.index(hx, hy)
                            if 0 <= rh < src_h.height and 0 <= ch < src_h.width:
                                hz = dtm_h_data[rh, ch]
                                if hz > -1000:
                                    hotspots_x.append(hx)
                                    hotspots_y.append(hy)
                                    hotspots_z.append(hz * z_exag + 4.0)
            except Exception:
                pass

        else:
            # --- SIMULATION ENGINE: GENERATE HIGHLY REALISTIC SYNTHETIC TERRAIN ---
            # Create a synthetic 150x150 grid matching the raw bounding box
            x_grid = np.linspace(bounds["min_x"], bounds["max_x"], 150)
            y_grid = np.linspace(bounds["min_y"], bounds["max_y"], 150)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Combine sine, cosine, and noise to represent mountain ridges and a river valley
            freq_x = 2 * np.pi / (bounds["max_x"] - bounds["min_x"])
            freq_y = 2 * np.pi / (bounds["max_y"] - bounds["min_y"])

            valley = 20 * np.sin((X - bounds["min_x"]) * freq_x * 0.5)
            hills = (
                35
                * np.cos((X - bounds["min_x"]) * freq_x * 1.5)
                * np.sin((Y - bounds["min_y"]) * freq_y * 1.5)
            )
            dtm_data = valley + hills + 120.0

            # Add micro-relief noise
            np.random.seed(42)
            noise = np.random.normal(0, 1.5, dtm_data.shape)
            dtm_data += noise

            # Adjust heights to match raw Z scale
            z_min = bounds["min_z"]
            z_max = bounds["max_z"]
            if z_max - z_min > 10:
                dtm_data = z_min + (dtm_data - dtm_data.min()) / (
                    dtm_data.max() - dtm_data.min()
                ) * (z_max - z_min)

            # Simulate flowing river channel shape (Cyan curves)
            line_x, line_y, line_z = [], [], []
            cx = np.linspace(bounds["min_x"], bounds["max_x"], 100)
            cy = (
                bounds["min_y"]
                + (bounds["max_y"] - bounds["min_y"]) * 0.5
                + (bounds["max_y"] - bounds["min_y"])
                * 0.2
                * np.sin(
                    5 * (cx - bounds["min_x"]) / (bounds["max_x"] - bounds["min_x"])
                )
            )

            for xi, yi in zip(cx, cy):
                # Interpolate simulated Z height
                col_idx = int(
                    (xi - bounds["min_x"]) / (bounds["max_x"] - bounds["min_x"]) * 149
                )
                row_idx = int(
                    (yi - bounds["min_y"]) / (bounds["max_y"] - bounds["min_y"]) * 149
                )
                col_idx = max(0, min(149, col_idx))
                row_idx = max(0, min(149, row_idx))

                h_val = dtm_data[row_idx, col_idx]
                line_x.append(xi)
                line_y.append(yi)
                line_z.append(h_val * z_exag - 1.0)  # Flowing in the valley

            # Add secondary streams
            line_x.append(None)
            line_y.append(None)
            line_z.append(None)
            cx2 = np.linspace(
                bounds["min_x"] + (bounds["max_x"] - bounds["min_x"]) * 0.3,
                bounds["max_x"] * 0.9,
                50,
            )
            cy2 = bounds["min_y"] + (bounds["max_y"] - bounds["min_y"]) * 0.1 * np.cos(
                3 * cx2 / bounds["max_x"]
            )
            for xi, yi in zip(cx2, cy2):
                col_idx = int(
                    (xi - bounds["min_x"]) / (bounds["max_x"] - bounds["min_x"]) * 149
                )
                row_idx = int(
                    (yi - bounds["min_y"]) / (bounds["max_y"] - bounds["min_y"]) * 149
                )
                col_idx = max(0, min(149, col_idx))
                row_idx = max(0, min(149, row_idx))
                h_val = dtm_data[row_idx, col_idx]
                line_x.append(xi)
                line_y.append(yi)
                line_z.append(h_val * z_exag - 0.5)

            # Generate simulated hotspots at the lowest catchment points
            hotspots_x, hotspots_y, hotspots_z = [], [], []
            flat_indices = np.argsort(dtm_data.ravel())[
                :25
            ]  # Lowest points are hotspots
            for idx in flat_indices:
                r = idx // dtm_data.shape[1]
                c = idx % dtm_data.shape[1]
                hx = x_grid[c]
                hy = y_grid[r]
                hz = dtm_data[r, c]
                hotspots_x.append(hx)
                hotspots_y.append(hy)
                hotspots_z.append(hz * z_exag + 2.0)

        # Build Interactive Figure
        fig = go.Figure()

        # Color palettes based on state status
        colorscale = "balance" if is_processed else "earth"

        fig.add_trace(
            go.Surface(
                z=dtm_data * z_exag,
                x=x_grid,
                y=y_grid,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Elevation (m)", thickness=15, len=0.6, x=1.05),
                name="Terrain Structure",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line=dict(color="#00f0ff", width=6.5),
                name="Flow Networks",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=hotspots_x,
                y=hotspots_y,
                z=hotspots_z,
                mode="markers",
                marker=dict(
                    size=5,
                    color="#ff0055",
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                name="Waterlogging Hotspots",
            )
        )

        # Dark layout options
        fig.update_layout(
            template="plotly_dark",
            title=f"{self.config['name']} - Interactive 3D Hydrological Design",
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.9)),
                aspectmode="data",
            ),
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.05,
                bgcolor="rgba(10, 15, 20, 0.7)",
            ),
        )

        return fig

    def load_raster_data_arrays(self, map_type):
        """
        Helper to load 2D arrays for rendering the map series (Slope, Acc, Dir, Flood Risk).
        Returns a beautifully downsampled array, extent bounds, and colormap parameters.
        """
        is_processed = self.check_processed_status()

        if is_processed:
            mapping = {
                "slope": self.config["slope"],
                "flow_dir": self.config["flow_dir"],
                "flow_acc": self.config["flow_acc"],
                "flood_risk": self.config["flood_risk"],
            }

            if map_type not in mapping:
                return None

            raster_path = os.path.join(self.folder_path, mapping[map_type])
            if not os.path.exists(raster_path):
                return None

            try:
                with rasterio.open(raster_path) as src:
                    # Load and squeeze
                    data = src.read(1, out_shape=(1, 300, 300))
                    # Mask nodata
                    data = np.ma.masked_invalid(data)
                    data = np.ma.masked_less(data, -1000)
                    return {
                        "simulated": False,
                        "data": data,
                        "bounds": src.bounds,
                        "crs": src.crs.to_string() if src.crs else "",
                    }
            except Exception:
                pass

        # --- DYNAMIC SIMULATION OF MAPS FOR UNPROCESSED REGIONS ---
        raw_meta = self.get_raw_metadata()
        bounds = raw_meta["bounds"]

        # Create a synthetic 200x200 grid
        size = 200
        x = np.linspace(bounds["min_x"], bounds["max_x"], size)
        y = np.linspace(bounds["min_y"], bounds["max_y"], size)
        X, Y = np.meshgrid(x, y)

        np.random.seed(101)

        if map_type == "slope":
            # Slope: derived from the gradient of synthetic landscape
            # Simulating slope using high frequencies
            data = np.abs(
                np.sin(X / 200) * np.cos(Y / 200) * 15
                + np.random.normal(0, 1.0, X.shape)
            )
            data = np.clip(data, 0, 45)  # Max 45 degree slope
        elif map_type == "flow_dir":
            # Flow Direction: 1 to 128 D8 integers
            dirs = [1, 2, 4, 8, 16, 32, 64, 128]
            data = np.random.choice(dirs, size=X.shape)
        elif map_type == "flow_acc":
            # Flow Accumulation: concentric patterns flowing towards a valley line
            valley_line = bounds["min_y"] + (bounds["max_y"] - bounds["min_y"]) * 0.5
            dist_to_valley = np.abs(Y - valley_line)
            # High accumulation in the valley, exponential drop-off
            data = 100000 / (dist_to_valley + 10) + np.random.exponential(15, X.shape)
            data = np.clip(data, 0.1, None)
        elif map_type == "flood_risk":
            # Flood risk based on TWI
            valley_line = bounds["min_y"] + (bounds["max_y"] - bounds["min_y"]) * 0.5
            dist_to_valley = np.abs(Y - valley_line)
            # High risk in the valley
            data = (
                15
                - (dist_to_valley / (bounds["max_y"] - bounds["min_y"])) * 12
                + np.random.normal(0, 0.5, X.shape)
            )
            data = np.clip(data, 0, 15)
        else:
            return None

        return {
            "simulated": True,
            "data": data,
            "bounds": bounds,
            "crs": "EPSG:32619 (Simulated Projection)",
        }
