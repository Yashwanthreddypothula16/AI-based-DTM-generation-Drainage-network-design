#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
================================================================================
   ___ ___ ___   ___ ___ ___ ___ _    ___ _  _ ___
  | _ \_ _| _ \ | _ \_ _| _ \ __| |  |_ _| \| | __|
  |  _/| ||  _/ |  _/| ||  _/ _|| |__ | || .` | _|
  |_| |___|_|   |_| |___|_| |___|____|___|_|\\_|___|
    Hydrological AI/ML & 3D Drainage Design Suite
================================================================================
Developed for Advanced AI-based DTM Generation & Drainage Network Design
Target Audience: IIT Tirupati Hackathon Judges
Author: Antigravity AI Partner
"""

import os
import sys
import time
import argparse
import subprocess
import shutil

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Color Codes for Stunning Console Graphics
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Header Banner
BANNER = f"""
{CYAN}{BOLD}  ___ ___ ___   ___ ___ ___ ___ _    ___ _  _ ___ 
 | _ \\_ _| _ \\ | _ \\_ _| _ \\ __| |  |_ _| \\| | __|
 |  _/| ||  _/ |  _/| ||  _/ _|| |__ | || .` | _| 
 |_| |___|_|   |_| |___|_| |___|____|___|_|\\_|___|
   {RESET}{YELLOW}Hydrological AI/ML & 3D Drainage Design Suite{RESET}
{BLUE}======================================================={RESET}
"""

# Categorize and define scripts for the Rajasthan Point Cloud pipeline
RAJASTHAN_STEPS = [
    {
        "step": "1",
        "script": "01_clean_rajasthan_data.py",
        "name": "Point Cloud Cleaning",
        "type": "CORE",
        "desc": "Removes Drone LiDAR noise outliers beyond 3 standard deviations.",
    },
    {
        "step": "2",
        "script": "02_ml_classifier.py",
        "name": "ML Ground Classifier",
        "type": "CORE",
        "desc": "Trains a Random Forest classifier to identify ground vs roof/vegetation.",
    },
    {
        "step": "3",
        "script": "03_generate_dtm.py",
        "name": "DTM Generation",
        "type": "CORE",
        "desc": "Linear grid interpolation of ML ground points to generate a 1m DTM raster.",
    },
    {
        "step": "4",
        "script": "04_extract_drainage.py",
        "name": "Drainage Extraction",
        "type": "CORE",
        "desc": "Runs pit filling, D8 routing, flow accumulation, and polyline shapefile tracing.",
    },
    {
        "step": "5",
        "script": "05_visualize_3d.py",
        "name": "3D Terrain Overlay Map",
        "type": "VISUALIZATION",
        "desc": "Generates 3D surface relief mesh with stark 2D red vector drainage overlay.",
    },
    {
        "step": "6",
        "script": "06_generate_maps.py",
        "name": "Hydrological Map Series",
        "type": "CORE",  # Set as CORE because it computes raj_twi_flood_risk.tif needed by Steps 9 and 11
        "desc": "Calculates TWI & slope rasters, saves flow accumulation/direction maps.",
    },
    {
        "step": "7",
        "script": "07_generate_3d_gif.py",
        "name": "3D Rotating Terrain GIF",
        "type": "VISUALIZATION",
        "desc": "Animates rotating 3D hillshade terrain model using matplotlib FuncAnimation.",
    },
    {
        "step": "8",
        "script": "08_visualize_flow_arrows.py",
        "name": "Flow Direction Quiver Map",
        "type": "VISUALIZATION",
        "desc": "Overlays downsampled D8 flow direction vector arrows on terrain mesh.",
    },
    {
        "step": "9",
        "script": "09_generate_enhanced_flood_map.py",
        "name": "TWI Flood Risk Mapping",
        "type": "VISUALIZATION",
        "desc": "Categorizes TWI values into Low, Medium, and High flood risk hazard zones.",
    },
    {
        "step": "10",
        "script": "10_detail_flow_accumulation.py",
        "name": "Flow Accumulation Analysis",
        "type": "VISUALIZATION",
        "desc": "Plots high-contrast log flow accumulation to pinpoint runoff concentration channels.",
    },
    {
        "step": "11",
        "script": "11_final_hackathon_infographic.py",
        "name": "Hackathon 4-Panel Infographic",
        "type": "VISUALIZATION",
        "desc": "Consolidates terrain, overlays, TWI risk, and final strategy into a master dashboard.",
    },
    {
        "step": "12",
        "script": "12_ultra_res_3d_render.py",
        "name": "HQ 3D Relief PNG",
        "type": "VISUALIZATION",
        "desc": "Renders high-DPI 3D terrain relief at full native raster grid resolution.",
    },
    {
        "step": "13",
        "script": "13_ground_level_3d_view.py",
        "name": "Ground-Level Perspective Render",
        "type": "VISUALIZATION",
        "desc": "Renders low-angle scenic 3D view showcasing the micro-topological relief.",
    },
    {
        "step": "14",
        "script": "14_integrated_3d_dashboard.py",
        "name": "Point Cloud & GIS 3D Dashboard",
        "type": "VISUALIZATION",
        "desc": "Lifts 2D shapefile drainage lines directly onto a 3D point cloud mesh.",
    },
    {
        "step": "15",
        "script": "15_3d_ground_vs_nonground.py",
        "name": "Ground vs Non-Ground 3D Render",
        "type": "VISUALIZATION",
        "desc": "Plots point cloud classes in earthy/organic styles with vertical exaggeration.",
    },
    {
        "step": "16",
        "script": "16_generate_classification_3d_gif.py",
        "name": "Classification Rotation GIF",
        "type": "VISUALIZATION",
        "desc": "Generates 360-degree rotating GIF of the classified LiDAR point cloud.",
    },
    {
        "step": "17",
        "script": "17_advanced_3d_drainage_model.py",
        "name": "Raised Slab 3D Model",
        "type": "VISUALIZATION",
        "desc": "Renders terrain as a solid 3D geological block with underlaid base and labeled outlets.",
    },
    {
        "step": "18",
        "script": "18_generate_advanced_slab_rotation.py",
        "name": "Slab Model Orbital Animation",
        "type": "VISUALIZATION",
        "desc": "Generates elegant orbital spinning GIF of the slab geological block.",
    },
    {
        "step": "19",
        "script": "19_interactive_3d_world.py",
        "name": "Interactive Plotly 3D HTML",
        "type": "VISUALIZATION",
        "desc": "Creates lightweight, responsive browser-ready Plotly HTML 3D scene.",
    },
    {
        "step": "20",
        "script": "20_refined_classification.py",
        "name": "DBSCAN Refined Classification",
        "type": "CORE",
        "desc": "Applies spatial DBSCAN clustering to separate building points from vegetation.",
    },
    {
        "step": "21",
        "script": "21_statistics_dashboard.py",
        "name": "Premium HTML Stats Dashboard",
        "type": "VISUALIZATION",
        "desc": "Generates gorgeous Plotly HTML dashboard showing categorization percentages.",
    },
]

# Reference workflow (Gujarat/Sample in the root folder)
REFERENCE_STEPS = [
    {
        "step": "Ref-1",
        "script": "02_hydrology_network.py",
        "name": "Reference Hydrology Flow Network",
        "type": "CORE",
        "desc": "Runs pit filling, flow directions, accumulations, slope, and TWI on root dtm.tif.",
    },
    {
        "step": "Ref-2",
        "script": "03_vectorize_network.py",
        "name": "Reference Stream Vectorizer",
        "type": "CORE",
        "desc": "Traces streams from streams_wbt.tif into network_vector.shp.",
    },
]

# Set the active conversation directory to copy visualizations dynamically
ACTIVE_BRAIN_DIR = r"C:\Users\nanir\OneDrive\Desktop\DTM"
CONVERSATION_BRAIN_DIR = (
    r"C:\Users\nanir\.gemini\antigravity\brain\e25be27b-f581-4821-a67d-2789942db877"
)


def get_python_interpreter():
    """Locate and return the best python interpreter to use (preferring local .venv)."""
    venv_path = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if sys.platform.startswith("win"):
        if os.path.exists(venv_path):
            return venv_path
    else:
        # Unix/macOS fallback
        venv_path_unix = os.path.join(os.getcwd(), ".venv", "bin", "python")
        if os.path.exists(venv_path_unix):
            return venv_path_unix
    return sys.executable


def check_dependencies(interpreter):
    """Verify all critical libraries required for the pipeline are installed on target interpreter."""
    print(f"\n{YELLOW}{BOLD}[DEPENDENCY VERIFICATION]{RESET}")
    print(f"  Target Interpreter: {CYAN}{interpreter}{RESET}")
    required = {
        "laspy": "LiDAR point cloud parsing",
        "rasterio": "Raster GIS dataset processing",
        "whitebox": "Hydrological modeling suite",
        "geopandas": "Vector shapefile spatial geometry",
        "plotly": "Premium interactive HTML plotting",
        "sklearn": "Machine learning (Random Forest, DBSCAN)",
        "scipy": "Spatial queries (cKDTree) & interpolation",
        "matplotlib": "2D and 3D rendering engines",
    }

    missing = []
    for pkg, desc in required.items():
        try:
            # Run import check command using target interpreter
            res = subprocess.run(
                [interpreter, "-c", f"import {pkg}"], capture_output=True
            )
            if res.returncode == 0:
                print(f"  {GREEN}✔{RESET} {pkg:<12} : Installed successfully ({desc})")
            else:
                print(f"  {RED}✘{RESET} {pkg:<12} : {RED}MISSING{RESET} ({desc})")
                missing.append(pkg)
        except Exception:
            print(f"  {RED}✘{RESET} {pkg:<12} : {RED}MISSING{RESET} ({desc})")
            missing.append(pkg)

    if missing:
        print(f"\n{RED}{BOLD}WARNING: Missing packages detected!{RESET}")
        print("To install them, please run:")
        print(f"pip install {' '.join(missing)}")
        return False

    print(
        f"{GREEN}{BOLD}All essential systems are operational! Ready to execute.{RESET}\n"
    )
    return True


def copy_visualizations_to_brain(script_name):
    """Copies premium visualization output files directly to the Gemini chat view so they render in real-time."""
    if not os.path.exists(CONVERSATION_BRAIN_DIR):
        return

    mapping = {
        "05_visualize_3d.py": ["final_rajasthan_vis.png"],
        "12_ultra_res_3d_render.py": ["ULTRA_HD_3D_TERRAIN.png"],
        "13_ground_level_3d_view.py": ["GROUND_LEVEL_3D_VIEW.png"],
        "14_integrated_3d_dashboard.py": ["MASTER_INTEGRATED_3D_DASHBOARD.png"],
        "15_3d_ground_vs_nonground.py": ["3D_GROUND_VS_NONGROUND.png"],
        "17_advanced_3d_drainage_model.py": ["ADVANCED_3D_DRAINAGE_SLAB_MODEL.png"],
    }

    if script_name in mapping:
        source_dir = os.path.join(os.getcwd(), "Rajasthan_Point_Cloud")
        for filename in mapping[script_name]:
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(CONVERSATION_BRAIN_DIR, filename)
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dest_file)
                    print(
                        f"  {CYAN}⮞ Real-Time UI Render Sync: Copied {filename} to Gemini Chat View{RESET}"
                    )
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Master Execution Pipeline for AI DTM & Drainage Design Suite"
    )
    parser.add_argument(
        "state",
        type=str,
        nargs="?",
        default=None,
        help="Select the state to run (e.g. Rajasthan, Gujarat, Punjab, Andaman)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["rajasthan", "reference"],
        default="rajasthan",
        help="Select pipeline to run (default: rajasthan, ignored if positional state is provided)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Production Mode: Skips heavy visual-only rendering tasks and rotating animations",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated step numbers to execute specific stages (e.g. --steps 1,2,3 or 1-4)",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only run dependency validation diagnostics",
    )

    args = parser.parse_args()

    # 1. Print banner
    print(BANNER)

    # 1.5 Determine target python interpreter (prefer local .venv if exists)
    interpreter = get_python_interpreter()

    # 2. Perform dependency check
    deps_ok = check_dependencies(interpreter)
    if args.check_deps:
        sys.exit(0 if deps_ok else 1)

    # Determine the target state key
    if args.state:
        state_key = args.state.lower().strip()
    else:
        state_key = args.pipeline.lower().strip()

    # 3. Determine pipeline scripts or dispatch to generalized pipeline
    if state_key not in ["rajasthan", "reference"]:
        from config import REGIONS

        if state_key not in REGIONS:
            print(
                f"{RED}ERROR: State '{args.state}' is not registered and no matching folder found.{RESET}"
            )
            print(f"Registered states: {', '.join(REGIONS.keys())}")
            sys.exit(1)

        print(
            f"\n{CYAN}{BOLD}--- DISPATCHING TO SCALABLE INFERENCE PIPELINE FOR {state_key.upper()} ---{RESET}"
        )
        from pipeline_executor import run_state_pipeline

        success = run_state_pipeline(state_key)
        sys.exit(0 if success else 1)

    if state_key == "reference":
        pipeline_name = "REFERENCE ROOT DTM WORKFLOW"
        selected_steps = REFERENCE_STEPS
    else:
        pipeline_name = "PRIMARY RAJASTHAN LI-DAR AUTOMATED PIPELINE"
        selected_steps = RAJASTHAN_STEPS

    # 4. Filter by steps if requested
    run_list = []
    if args.steps:
        target_steps = []
        # Expand ranges like 1-4
        for chunk in args.steps.split(","):
            if "-" in chunk:
                try:
                    start, end = map(int, chunk.split("-"))
                    target_steps.extend(map(str, range(start, end + 1)))
                except ValueError:
                    print(f"{RED}Invalid step range: {chunk}{RESET}")
                    sys.exit(1)
            else:
                target_steps.append(chunk.strip())

        for step in selected_steps:
            if step["step"] in target_steps:
                run_list.append(step)
    else:
        run_list = selected_steps

    # 5. Filter by production mode (skips VISUALIZATION tasks)
    if args.prod:
        original_count = len(run_list)
        run_list = [s for s in run_list if s["type"] == "CORE"]
        skipped_count = original_count - len(run_list)
        print(
            f"{YELLOW}★ PRODUCTION MODE ACTIVE: Skipping {skipped_count} Visualization-only tasks to optimize throughput.{RESET}"
        )

    if not run_list:
        print(f"{RED}No stages selected or matching filter criteria!{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}{BOLD}--- INITIATING {pipeline_name} ---{RESET}")
    print(f"Total Steps Selected: {len(run_list)}")
    print(f"Working Directory   : {os.getcwd()}")
    print("-" * 60)

    pipeline_start = time.time()
    success_count = 0
    failures = []

    # Execute Pipeline
    for idx, step in enumerate(run_list, start=1):
        step["step"]
        script = step["script"]
        name = step["name"]
        step_type = step["type"]
        desc = step["desc"]

        type_lbl = (
            f"{GREEN}[CORE]{RESET}"
            if step_type == "CORE"
            else f"{MAGENTA}[VISUAL]{RESET}"
        )

        print(f"\n{BOLD}{YELLOW}STAGE {idx}/{len(run_list)} ({type_lbl} {name}){RESET}")
        print(f"  File  : {CYAN}{script}{RESET}")
        print(f"  About : {desc}")
        print(f"  Status: {YELLOW}Executing subprocess...{RESET}")

        script_path = os.path.join(os.getcwd(), script)
        if not os.path.exists(script_path):
            print(f"  {RED}✘ FAILED: File not found at {script_path}{RESET}")
            failures.append((name, "Script file missing"))
            break

        step_start = time.time()

        # Trigger the execution process
        try:
            # Set environment variable to dynamic brain directory to ensure outputs copy correctly
            env = os.environ.copy()
            env["BRAIN_DIR"] = CONVERSATION_BRAIN_DIR

            result = subprocess.run(
                [interpreter, script],
                env=env,
                capture_output=False,  # Keep standard output visible so judges can see logs real-time!
                text=True,
            )

            elapsed = time.time() - step_start

            if result.returncode == 0:
                print(
                    f"  {GREEN}✔ SUCCESS! Stage '{name}' completed in {elapsed:.1f}s{RESET}"
                )
                success_count += 1

                # Copy Premium outputs to Gemini workspace
                copy_visualizations_to_brain(script)
            else:
                print(
                    f"  {RED}✘ ERROR: Subprocess exited with code {result.returncode} ({elapsed:.1f}s){RESET}"
                )
                failures.append(
                    (name, f"Subprocess returned error {result.returncode}")
                )
                break

        except Exception as e:
            elapsed = time.time() - step_start
            print(f"  {RED}✘ CRITICAL CRASH: {str(e)} ({elapsed:.1f}s){RESET}")
            failures.append((name, str(e)))
            break

    # Final Pipeline Statistics & Dashboard
    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"{CYAN}{BOLD}              PIPELINE EXECUTION REPORT{RESET}")
    print("=" * 60)
    print(f"  Active Workflow    : {pipeline_name}")
    print(f"  Completed Stages   : {success_count}/{len(run_list)}")
    print(
        f"  Total Duration     : {total_elapsed / 60:.2f} minutes ({total_elapsed:.1f}s)"
    )

    if failures:
        print(f"\n  {RED}{BOLD}PIPELINE RUN FAILED!{RESET}")
        for name, err in failures:
            print(f"    - Failed Stage: {RED}{name}{RESET} (Reason: {err})")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}PIPELINE COMPLETED SUCCESSFULLY!{RESET}")
        print("  All data processing, ML classifications, DTM interpolations,")
        print("  hydrology routing models, and 3D visual outputs are GIS-ready.")
        print("=" * 60)


if __name__ == "__main__":
    main()
