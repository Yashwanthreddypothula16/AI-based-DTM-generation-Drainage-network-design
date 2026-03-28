import os
from whitebox import WhiteboxTools

def main():
    print("Initializing Hydrology Pipeline for Rajasthan DTM...")
    wbt = WhiteboxTools()
    wbt.verbose = True
    base_dir = os.path.abspath(os.getcwd())
    wbt.set_working_dir(base_dir)
    
    dtm = os.path.join(base_dir, r"Rajasthan_Point_Cloud\rajasthan_dtm.tif")
    dtm_filled = os.path.join(base_dir, r"Rajasthan_Point_Cloud\rajasthan_dtm_filled.tif")
    flow_dir = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_flow_dir.tif")
    flow_acc = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_flow_acc.tif")
    streams_raster = os.path.join(base_dir, r"Rajasthan_Point_Cloud\raj_streams.tif")
    streams_vector = os.path.join(base_dir, r"Rajasthan_Point_Cloud\rajasthan_drainage_network.shp")
    
    # 1. Pit Filling (Ensure network stays connected despite ground artifacts)
    print("Filling DTM depressions...")
    wbt.fill_depressions(dtm, dtm_filled, fix_flats=True)
    
    # 2. Flow Direction
    print("Computing D8 Flow Direction...")
    wbt.d8_pointer(dtm_filled, flow_dir)
    
    # 3. Flow Accumulation
    print("Computing D8 Flow Accumulation...")
    wbt.d8_flow_accumulation(dtm_filled, flow_acc, out_type="cells")
    
    # 4. Extract Streams
    threshold = 500  # Higher threshold because the village is bigger, we only want main drains
    print(f"Extracting Stream networks with accumulation > {threshold} cells...")
    wbt.extract_streams(flow_acc, streams_raster, threshold=threshold)
    
    # 5. Vectorize (GIS Ready LineStrings)
    print("Tracing raster streams into fully connected 2D Polylines...")
    wbt.raster_streams_to_vector(
        streams=streams_raster,
        d8_pntr=flow_dir,
        output=streams_vector
    )
    
    print(f"Drainage Extraction Complete! GIS network saved: {streams_vector}")

if __name__ == "__main__":
    main()
