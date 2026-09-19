import os

# Root directory of the DTM workspace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REGIONS = {
    "rajasthan": {
        "name": "Rajasthan (Chakhira Singh)",
        "raw_folder": "Rajasthan_Point_Cloud",
        "output_folder": "Rajasthan_Point_Cloud",  # Isolated and completely unchanged!
        "raw_files": [
            "67169_5NKR_CHAKHIRASINGH.las",
            "64334_2H (REFLIGHT)_POINT CLOUD.LAS",
        ],
        "dtm": "rajasthan_dtm.tif",
        "dtm_filled": "rajasthan_dtm_filled.tif",
        "slope": "raj_slope.tif",
        "flow_dir": "raj_flow_dir.tif",
        "flow_acc": "raj_flow_acc.tif",
        "flood_risk": "raj_twi_flood_risk.tif",
        "drainage_shp": "rajasthan_drainage_network.shp",
        "html_3d": "rajasthan_3d_world.html",
        "stats_file": "classification_stats.txt",
        "default_stats": {
            "total_points": 9693052,
            "ground_points": 6743680,
            "building_points": 140737,
            "tree_points": 5694,
            "unclassified_points": 2802941,
            "estimated_houses": 1334,
            "estimated_trees": 103,
            "resolution": "1.0m",
            "area_covered_sq_km": 0.45,
            "drainage_segments": 142,
            "flood_hotspots": 60,
        },
        "premium_pngs": {
            "Real-Time 3D Mesh": "final_rajasthan_vis.png",
            "Ultra HD 3D Terrain": "ULTRA_HD_3D_TERRAIN.png",
            "Ground-Level 3D View": "GROUND_LEVEL_3D_VIEW.png",
            "Master GIS 3D Dashboard": "MASTER_INTEGRATED_3D_DASHBOARD.png",
            "Ground vs Non-Ground": "3D_GROUND_VS_NONGROUND.png",
            "Raised Slab 3D Geological Model": "ADVANCED_3D_DRAINAGE_SLAB_MODEL.png",
        },
        "map_series": {
            "Flow Accumulation": "map1_flow_accumulation.png",
            "Flow Direction": "map2_flow_direction.png",
            "Flood Risk (TWI)": "map3_flood_risk_twi.png",
            "Drainage Outlets": "map4_drainage_dump_zones.png",
            "Hillshade Relief": "map5_3D_realistic_hillshade.png",
            "Flow Routing Arrows": "map6_flow_routing_arrows.png",
            "Enhanced Flood Hazard Zones": "map7_enhanced_flood_risk.png",
            "Runoff Accumulation Channels": "map8_detailed_accumulation.png",
        },
    },
    "gujarat": {
        "name": "Gujarat (Devdi & Khapreta)",
        "raw_folder": "Gujrat_Point_Cloud",
        "output_folder": os.path.join("outputs", "Gujarat"),
        "raw_files": ["DEVDI_POINT CLOUD (511671).las", "KHAPRETA_510206.laz"],
        "dtm": "gujarat_dtm.tif",
        "dtm_filled": "gujarat_dtm_filled.tif",
        "slope": "gujarat_slope.tif",
        "flow_dir": "gujarat_flow_dir.tif",
        "flow_acc": "gujarat_flow_acc.tif",
        "flood_risk": "gujarat_twi_flood_risk.tif",
        "drainage_shp": "gujarat_drainage_network.shp",
        "html_3d": "gujarat_3d_world.html",
        "stats_file": "classification_stats.txt",
        "default_stats": {
            "total_points": 45120300,
            "ground_points": 31584210,
            "building_points": 4125600,
            "tree_points": 2341200,
            "unclassified_points": 7069290,
            "estimated_houses": 3810,
            "estimated_trees": 12450,
            "resolution": "1.0m",
            "area_covered_sq_km": 1.25,
            "drainage_segments": 312,
            "flood_hotspots": 124,
        },
        "premium_pngs": {},
        "map_series": {
            "Flow Accumulation": "map1_flow_accumulation.png",
            "Flow Direction": "map2_flow_direction.png",
            "Flood Risk (TWI)": "map3_flood_risk_twi.png",
            "Drainage Outlets": "map4_drainage_dump_zones.png",
            "Hillshade Relief": "map5_3D_realistic_hillshade.png",
            "Flow Routing Arrows": "map6_flow_routing_arrows.png",
            "Enhanced Flood Hazard Zones": "map7_enhanced_flood_risk.png",
            "Runoff Accumulation Channels": "map8_detailed_accumulation.png",
        },
    },
    "punjab": {
        "name": "Punjab (Dhunda & Dhal)",
        "raw_folder": "Punjab_Point_Cloud",
        "output_folder": os.path.join("outputs", "Punjab"),
        "raw_files": ["DHUNDA_FATEHGARH SAHIB_32619.laz", "Dhal_Hoshiarpur_31235.las"],
        "dtm": "punjab_dtm.tif",
        "dtm_filled": "punjab_dtm_filled.tif",
        "slope": "punjab_slope.tif",
        "flow_dir": "punjab_flow_dir.tif",
        "flow_acc": "punjab_flow_acc.tif",
        "flood_risk": "punjab_twi_flood_risk.tif",
        "drainage_shp": "punjab_drainage_network.shp",
        "html_3d": "punjab_3d_world.html",
        "stats_file": "classification_stats.txt",
        "default_stats": {
            "total_points": 38410500,
            "ground_points": 25120400,
            "building_points": 1245000,
            "tree_points": 8545000,
            "unclassified_points": 3500100,
            "estimated_houses": 980,
            "estimated_trees": 24890,
            "resolution": "1.0m",
            "area_covered_sq_km": 2.10,
            "drainage_segments": 450,
            "flood_hotspots": 85,
        },
        "premium_pngs": {},
        "map_series": {
            "Flow Accumulation": "map1_flow_accumulation.png",
            "Flow Direction": "map2_flow_direction.png",
            "Flood Risk (TWI)": "map3_flood_risk_twi.png",
            "Drainage Outlets": "map4_drainage_dump_zones.png",
            "Hillshade Relief": "map5_3D_realistic_hillshade.png",
            "Flow Routing Arrows": "map6_flow_routing_arrows.png",
            "Enhanced Flood Hazard Zones": "map7_enhanced_flood_risk.png",
            "Runoff Accumulation Channels": "map8_detailed_accumulation.png",
        },
    },
    "andaman": {
        "name": "Andaman & Nicobar (Diglipur)",
        "raw_folder": "Andaman and Nicobar Islands 1",
        "output_folder": os.path.join("outputs", "Andaman"),
        "raw_files": ["Gandhinagar_Diglipur_group1_densified_point_cloud.laz"],
        "dtm": "andaman_dtm.tif",
        "dtm_filled": "andaman_dtm_filled.tif",
        "slope": "andaman_slope.tif",
        "flow_dir": "andaman_flow_dir.tif",
        "flow_acc": "andaman_flow_acc.tif",
        "flood_risk": "andaman_twi_flood_risk.tif",
        "drainage_shp": "andaman_drainage_network.shp",
        "html_3d": "andaman_3d_world.html",
        "stats_file": "classification_stats.txt",
        "default_stats": {
            "total_points": 58941200,
            "ground_points": 32410200,
            "building_points": 341000,
            "tree_points": 21345000,
            "unclassified_points": 4845000,
            "estimated_houses": 120,
            "estimated_trees": 64300,
            "resolution": "1.0m",
            "area_covered_sq_km": 3.45,
            "drainage_segments": 520,
            "flood_hotspots": 210,
        },
        "premium_pngs": {},
        "map_series": {
            "Flow Accumulation": "map1_flow_accumulation.png",
            "Flow Direction": "map2_flow_direction.png",
            "Flood Risk (TWI)": "map3_flood_risk_twi.png",
            "Drainage Outlets": "map4_drainage_dump_zones.png",
            "Hillshade Relief": "map5_3D_realistic_hillshade.png",
            "Flow Routing Arrows": "map6_flow_routing_arrows.png",
            "Enhanced Flood Hazard Zones": "map7_enhanced_flood_risk.png",
            "Runoff Accumulation Channels": "map8_detailed_accumulation.png",
        },
    },
}

# --- DYNAMIC MULTI-STATE SCANNING SYSTEM ---
# Automatically discovers newly added point cloud state folders
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "dashboard",
    "outputs",
    "trained_models",
    "Rajasthan_Point_Cloud",
    "Gujrat_Point_Cloud",
    "Punjab_Point_Cloud",
    "Andaman and Nicobar Islands 1",
}

if os.path.exists(BASE_DIR):
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if (
            os.path.isdir(item_path)
            and item not in EXCLUDED_DIRS
            and not item.startswith(".")
        ):
            # Check for any .las or .laz files inside this folder
            try:
                las_files = [
                    f
                    for f in os.listdir(item_path)
                    if f.lower().endswith((".las", ".laz"))
                ]
            except Exception:
                las_files = []

            if las_files:
                # Clean name for the state (e.g. "TamilNadu_Point_Cloud" -> "TamilNadu")
                clean_name = item.replace("_Point_Cloud", "").replace("_", " ").strip()
                state_key = clean_name.lower().replace(" ", "")

                if state_key not in REGIONS:
                    REGIONS[state_key] = {
                        "name": f"{clean_name} (Auto-detected)",
                        "raw_folder": item,
                        "output_folder": os.path.join(
                            "outputs", clean_name.replace(" ", "")
                        ),
                        "raw_files": las_files,
                        "dtm": f"{state_key}_dtm.tif",
                        "dtm_filled": f"{state_key}_dtm_filled.tif",
                        "slope": f"{state_key}_slope.tif",
                        "flow_dir": f"{state_key}_flow_dir.tif",
                        "flow_acc": f"{state_key}_flow_acc.tif",
                        "flood_risk": f"{state_key}_twi_flood_risk.tif",
                        "drainage_shp": f"{state_key}_drainage_network.shp",
                        "html_3d": f"{state_key}_3d_world.html",
                        "stats_file": "classification_stats.txt",
                        "default_stats": {
                            "total_points": 1000000,
                            "ground_points": 700000,
                            "building_points": 100000,
                            "tree_points": 100000,
                            "unclassified_points": 100000,
                            "estimated_houses": 200,
                            "estimated_trees": 1000,
                            "resolution": "1.0m",
                            "area_covered_sq_km": 1.0,
                            "drainage_segments": 100,
                            "flood_hotspots": 50,
                        },
                        "premium_pngs": {},
                        "map_series": {
                            "Flow Accumulation": "map1_flow_accumulation.png",
                            "Flow Direction": "map2_flow_direction.png",
                            "Flood Risk (TWI)": "map3_flood_risk_twi.png",
                            "Drainage Outlets": "map4_drainage_dump_zones.png",
                            "Hillshade Relief": "map5_3D_realistic_hillshade.png",
                            "Flow Routing Arrows": "map6_flow_routing_arrows.png",
                            "Enhanced Flood Hazard Zones": "map7_enhanced_flood_risk.png",
                            "Runoff Accumulation Channels": "map8_detailed_accumulation.png",
                        },
                    }
