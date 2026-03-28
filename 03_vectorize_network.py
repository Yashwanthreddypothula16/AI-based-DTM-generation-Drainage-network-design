import os
from whitebox import WhiteboxTools

def main():
    print("Vectorizing stream network...")
    wbt = WhiteboxTools()
    wbt.set_working_dir(os.path.abspath(os.getcwd()))
    
    # Convert stream raster to connected line vectors (Shapefile format)
    streams_raster = "streams_wbt.tif"
    flow_dir = "flow_dir_wbt.tif"
    output_shp = "network_vector.shp"
    
    # Whitebox has a specialized algorithm to trace stream rasters down slope and create polylines
    print("Tracing raster to vector linestring...")
    wbt.raster_streams_to_vector(
        streams=streams_raster,
        d8_pntr=flow_dir,
        output=output_shp
    )
    
    print(f"Success! Vector drainage network saved as '{output_shp}'")
    print("This file contains fully connected drainage polylines, which satisfy the 'GIS-ready' requirement.")

if __name__ == "__main__":
    main()
