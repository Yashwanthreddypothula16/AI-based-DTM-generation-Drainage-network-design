import os
from whitebox import WhiteboxTools

def main():
    print("Setting up Hydrological Network Analysis...")
    wbt = WhiteboxTools()
    
    # Set the working directory to the current folder where files are
    wbt.set_working_dir(os.path.abspath(os.getcwd()))
    
    # 1. Fill Pits/Depressions
    # This is a critical step to ensure continuous flow paths.
    dtm_input = "dtm.tif"
    dtm_filled = "dtm_filled.tif"
    
    print(f"1. Filling depressions in {dtm_input}...")
    wbt.fill_depressions(dtm_input, dtm_filled, fix_flats=True)
    
    # 2. D8 Pointer (Flow Direction)
    # Calculates where each drop of water will flow
    flow_dir = "flow_dir_wbt.tif"
    print("2. Computing D8 flow direction...")
    wbt.d8_pointer(dtm_filled, flow_dir)
    
    # 3. D8 Flow Accumulation
    # Simulates how much water flows into each cell based on flow direction
    flow_acc = "flow_acc_wbt.tif"
    print("3. Computing D8 flow accumulation...")
    wbt.d8_flow_accumulation(dtm_filled, flow_acc, out_type="cells")
    
    # 4. Extract Streams
    # Extract only cells with high flow accumulation (e.g., more than 50 cells)
    threshold = 50
    streams_raster = "streams_wbt.tif"
    print(f"4. Extracting streams with >{threshold} cell accumulation...")
    wbt.extract_streams(flow_acc, streams_raster, threshold=threshold)
    
    # 5. Optional: Topographic Wetness Index (TWI) for finding hotspots
    # TWI helps pinpoint "waterlogging hotspots"
    twi = "twi_hotspots.tif"
    twi_slope = "slope.tif"
    print("5. Calculating Slope and Topographic Wetness Index (for hotspot mapping)...")
    wbt.slope(dtm_filled, twi_slope)
    wbt.wetness_index(twi_slope, flow_acc, twi)

    print(f"Hydrology network complete. Final stream raster saved to: {streams_raster}")

if __name__ == "__main__":
    main()
