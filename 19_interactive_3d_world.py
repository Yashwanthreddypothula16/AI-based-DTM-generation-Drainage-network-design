import os
import rasterio
import numpy as np
import plotly.graph_objects as go
import geopandas as gpd


def main():
    print("Generating Interactive 3D World (Plotly Edition)...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    
    dtm_path = os.path.join(out_dir, "rajasthan_dtm.tif")
    shp_path = os.path.join(out_dir, "rajasthan_drainage_network.shp")
    acc_path = os.path.join(out_dir, "raj_flow_acc.tif")
    output_html = os.path.join(out_dir, "rajasthan_3d_world.html")
    
    # Check existence
    if not os.path.exists(dtm_path):
        print(f"ERROR: DTM file not found at {dtm_path}")
        return

    # 1. Load Terrain Data (Optimized for Browser)
    print("Loading Terrain Surface...")
    with rasterio.open(dtm_path) as src:
        # 400x400 is very safe for smooth browser interaction
        dtm_data = src.read(1, out_shape=(1, 400, 400))
        bounds = src.bounds
        dtm_data[dtm_data < -1000] = np.nan
        
    x = np.linspace(bounds.left, bounds.right, dtm_data.shape[1])
    y = np.linspace(bounds.bottom, bounds.top, dtm_data.shape[0])
    
    z_exag = 3.0
    
    # 2. Extract 3D Drainage Network
    print("Extracting 3D Drainage Lines...")
    drainage = gpd.read_file(shp_path)
    line_x, line_y, line_z = [], [], []
    
    # Load full DTM for height sampling
    with rasterio.open(dtm_path) as src_full:
        for geom in drainage.geometry:
            if geom.geom_type == 'LineString':
                coords = np.array(geom.coords)
                for xi, yi in coords:
                    r, c = src_full.index(xi, yi)
                    if 0 <= r < src_full.height and 0 <= c < src_full.width:
                        # Direct height lookup
                        val = src_full.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0][0]
                        if val > -1000:
                            line_x.append(xi)
                            line_y.append(yi)
                            line_z.append(val * z_exag + 2)
                line_x.append(None); line_y.append(None); line_z.append(None)

    # 3. Detect Top 60 Hotspots
    print("Detecting Labeled Hotspots...")
    with rasterio.open(acc_path) as a_src:
        acc = a_src.read(1)
        valid_acc = np.ma.masked_invalid(acc)
        flat_idx = np.argsort(valid_acc.compressed())[-60:]
        fy, fx = np.where(valid_acc.mask == False)
        hotspots_x, hotspots_y, hotspots_z = [], [], []
        
        # We'll use the already open src_full if it were open, 
        # but let's just open dtm again briefly for hotspots
        with rasterio.open(dtm_path) as src_h:
            for idx in flat_idx:
                r, c = fy[idx], fx[idx]
                hx, hy = a_src.transform * (c, r)
                rh, ch = src_h.index(hx, hy)
                if 0 <= rh < src_h.height and 0 <= ch < src_h.width:
                    hz = src_h.read(1, window=rasterio.windows.Window(ch, rh, 1, 1))[0][0]
                    if hz > -1000:
                        hotspots_x.append(hx)
                        hotspots_y.append(hy)
                        hotspots_z.append(hz * z_exag + 8)

    # 4. Build Plotly Interactive Figure
    print("Building 3D World Canvas...")
    fig = go.Figure()
    fig.add_trace(go.Surface(z=dtm_data * z_exag, x=x, y=y, colorscale='balance', showscale=False, name='Terrain'))
    fig.add_trace(go.Scatter3d(x=line_x, y=line_y, z=line_z, mode='lines', line=dict(color='cyan', width=4), name='Flow'))
    fig.add_trace(go.Scatter3d(x=hotspots_x, y=hotspots_y, z=hotspots_z, mode='markers+text',
                               marker=dict(size=6, color='red', symbol='diamond'),
                               text=["HOTSPOT"] * len(hotspots_x), textfont=dict(color="white", size=9), name='Hotspots'))

    fig.update_layout(template="plotly_dark", title="Rajasthan Interactive 3D World",
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'))
    
    print(f"Saving interactive 3D world to {output_html}...")
    fig.write_html(output_html)
    print("SUCCESS!")

if __name__ == "__main__":
    main()
