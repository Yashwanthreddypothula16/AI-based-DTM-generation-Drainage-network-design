import os
from config import REGIONS, BASE_DIR


class StateRegistry:
    @staticmethod
    def get_all_states():
        """Returns lists of configured state identifiers."""
        return list(REGIONS.keys())

    @staticmethod
    def get_state_config(state_key):
        """Returns the dictionary configuration for a given state key."""
        if state_key not in REGIONS:
            raise ValueError(f"State '{state_key}' is not registered.")
        return REGIONS[state_key]

    @staticmethod
    def get_raw_folder(state_key):
        """Returns the absolute path to raw datasets folder."""
        cfg = StateRegistry.get_state_config(state_key)
        return os.path.join(BASE_DIR, cfg["raw_folder"])

    @staticmethod
    def get_output_folder(state_key):
        """Returns the absolute path to processed outputs folder."""
        cfg = StateRegistry.get_state_config(state_key)
        return os.path.join(BASE_DIR, cfg["output_folder"])

    @staticmethod
    def check_raw_exists(state_key):
        """Verifies if the configured raw point clouds exist on the disk."""
        cfg = StateRegistry.get_state_config(state_key)
        raw_dir = StateRegistry.get_raw_folder(state_key)
        if not os.path.exists(raw_dir):
            return False

        # Verify at least one configured raw file is present
        for rf in cfg["raw_files"]:
            rf_path = os.path.join(raw_dir, rf)
            if os.path.exists(rf_path):
                return True
        return False

    @staticmethod
    def get_output_filepaths(state_key):
        """Returns absolute target paths for all 11 required output categories."""
        cfg = StateRegistry.get_state_config(state_key)
        out_dir = StateRegistry.get_output_folder(state_key)

        # Establish prefix names
        prefix = state_key

        return {
            "cleaned_las": os.path.join(out_dir, f"{prefix}_cleaned.las"),
            "classified_las": os.path.join(out_dir, f"{prefix}_ml_classified.las"),
            "dtm": os.path.join(out_dir, cfg["dtm"]),
            "dtm_filled": os.path.join(out_dir, cfg["dtm_filled"]),
            "slope": os.path.join(out_dir, cfg["slope"]),
            "flow_dir": os.path.join(out_dir, cfg["flow_dir"]),
            "flow_acc": os.path.join(out_dir, cfg["flow_acc"]),
            "flood_risk": os.path.join(out_dir, cfg["flood_risk"]),
            "drainage_shp": os.path.join(out_dir, cfg["drainage_shp"]),
            "html_3d": os.path.join(out_dir, cfg["html_3d"]),
            "stats_txt": os.path.join(out_dir, cfg["stats_file"]),
            "stats_html": os.path.join(out_dir, "state_statistics.html"),
            "validation_json": os.path.join(out_dir, "validation_report.json"),
            # Map Series PNGs
            "png_accumulation": os.path.join(out_dir, "map1_flow_accumulation.png"),
            "png_direction": os.path.join(out_dir, "map2_flow_direction.png"),
            "png_flood_risk": os.path.join(out_dir, "map3_flood_risk_twi.png"),
            "png_drainage_outlets": os.path.join(
                out_dir, "map4_drainage_dump_zones.png"
            ),
            "png_hillshade": os.path.join(out_dir, "map5_3D_realistic_hillshade.png"),
            "png_quiver_arrows": os.path.join(out_dir, "map6_flow_routing_arrows.png"),
            "png_enhanced_flood": os.path.join(out_dir, "map7_enhanced_flood_risk.png"),
            "png_detailed_acc": os.path.join(out_dir, "map8_detailed_accumulation.png"),
        }

    @staticmethod
    def check_processed_status(state_key):
        """Checks if all critical output products exist for this state."""
        paths = StateRegistry.get_output_filepaths(state_key)

        # Core processed raster & vector indicators
        critical_outputs = [
            "dtm",
            "slope",
            "flow_dir",
            "flow_acc",
            "flood_risk",
            "drainage_shp",
            "html_3d",
        ]

        for k in critical_outputs:
            if not os.path.exists(paths[k]):
                return False
        return True
