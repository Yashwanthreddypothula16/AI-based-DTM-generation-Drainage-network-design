import os
import sys

# Dynamic path resolution to support Streamlit runtime context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zipfile
import io
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from config import REGIONS
from region_manager import RegionManager

# Page Configuration for Wide Hackathon Showcase Layout
st.set_page_config(
    page_title="AI DTM & Drainage Design Suite",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphism / Cyberpunk Dark Mode Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&family=Roboto:wght@300;400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0b0f17 !important;
        background-image: radial-gradient(circle at 50% 50%, #141b26 0%, #0b0f17 100%) !important;
        color: #c9d1d9 !important;
        font-family: 'Roboto', sans-serif !important;
    }
    
    /* Header Typography styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #ffffff !important;
    }
    
    /* Title Banners */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-align: center !important;
        margin-bottom: 2px !important;
        text-shadow: 0 0 25px rgba(0, 240, 255, 0.45) !important;
        letter-spacing: 2px !important;
    }
    
    .sub-title {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.15rem !important;
        color: #00f0ff !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        letter-spacing: 2.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e141f !important;
        border-right: 1px solid rgba(0, 240, 255, 0.1) !important;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(18, 25, 38, 0.75) !important;
        border: 1px solid rgba(0, 240, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        text-align: center !important;
        backdrop-filter: blur(10px) !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45) !important;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) !important;
        border-color: #00f0ff !important;
        box-shadow: 0 10px 25px rgba(0, 240, 255, 0.2) !important;
    }
    
    .metric-val {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-top: 5px !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1) !important;
    }
    
    .metric-lbl {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.85rem !important;
        color: #8b949e !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        font-weight: 700 !important;
    }
    
    /* Active Region Banner */
    .region-banner {
        background: linear-gradient(90deg, rgba(0, 240, 255, 0.08) 0%, rgba(22, 27, 34, 0) 100%) !important;
        border-left: 4px solid #00f0ff !important;
        padding: 15px 20px !important;
        border-radius: 4px !important;
        margin-bottom: 25px !important;
    }
    
    /* Streamlit Tabs Customization */
    button[data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        color: #8b949e !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 20px !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #00f0ff !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom-color: #00f0ff !important;
    }
    
    /* Custom Alerts */
    .alert-banner {
        background-color: rgba(255, 152, 0, 0.1) !important;
        border: 1px solid rgba(255, 152, 0, 0.3) !important;
        color: #ffa726 !important;
        border-radius: 8px !important;
        padding: 12px 18px !important;
        font-size: 0.95rem !important;
        margin-bottom: 20px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Helper function to zip shapefiles
def zip_shapefile_bundle(folder_path, shapefile_basename):
    zip_buffer = io.BytesIO()
    base_prefix = os.path.splitext(shapefile_basename)[0]

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        files_added = 0
        for filename in os.listdir(folder_path):
            if filename.startswith(base_prefix):
                file_path = os.path.join(folder_path, filename)
                zip_file.write(file_path, arcname=filename)
                files_added += 1

        if files_added == 0:
            return None

    zip_buffer.seek(0)
    return zip_buffer


# Main Dashboard App Setup
def main():
    # 1. Main Header Branding
    st.markdown(
        '<div class="main-title">🗺️ AI DTM & DRAINAGE DESIGN SUITE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Multi-Region LiDAR Topographical Web Dashboard</div>',
        unsafe_allow_html=True,
    )

    # 2. Sidebar Navigation Panel
    st.sidebar.markdown(
        "<h3 style='text-align:center;'>🕹️ CONTROL PANEL</h3>", unsafe_allow_html=True
    )

    region_options = {k: v["name"] for k, v in REGIONS.items()}
    selected_key = st.sidebar.selectbox(
        "SELECT ACTIVE REGION",
        options=list(region_options.keys()),
        format_func=lambda x: region_options[x],
    )

    # Instantiate the active Region Controller
    rm = RegionManager(selected_key)
    stats = rm.compile_statistics()
    is_processed = rm.check_processed_status()

    # Display Active Region Banner
    status_label = (
        "PROCESSED & READY" if is_processed else "SIMULATED / RAW BROWSER MODE"
    )
    st.markdown(
        f"""
        <div class="region-banner">
            <h4 style="margin: 0; padding: 0;">🌐 REGION FOCUS: {stats["region_name"]}</h4>
            <span style="font-size: 0.85rem; color: {"#4caf50" if is_processed else "#ffb74d"}; font-weight: bold; letter-spacing: 1px;">
                STATUS: {status_label}
            </span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if this is a simulated/fallback view, show warning banner
    if not is_processed:
        st.markdown(
            f"""
            <div class="alert-banner">
                ⚠️ <strong>Demonstration Fallback Active:</strong> Full hydrological outputs for 
                <strong>{stats["region_name"]}</strong> are pending pipeline processing. 
                A high-fidelity dynamic synthetic terrain and matching spatial indicators have been generated 
                based on raw point cloud boundaries to enable complete visual previewing.
            </div>
        """,
            unsafe_allow_html=True,
        )

    # 3. High-Density Statistics Panel Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Processed Points</div>
                <div class="metric-val">{stats["total_points"]:,}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Isolated Ground Points</div>
                <div class="metric-val">{stats["ground_points"]:,}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">DTM Cell Resolution</div>
                <div class="metric-val">{stats["dtm_resolution"]}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Catchment Area Covered</div>
                <div class="metric-val">{stats["area_covered_sq_km"]} km²</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Flood Hotspot Sinks</div>
                <div class="metric-val" style="color: #ff3366;">{stats["flood_hotspots"]}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Tab Structure Setup
    tab_overview, tab_dtm, tab_maps, tab_drainage, tab_3d = st.tabs(
        [
            "📊 OVERVIEW & ML STATISTICS",
            "⛰️ DIGITAL TERRAIN MODEL",
            "🗺️ HYDROLOGICAL MAP SERIES",
            "🌊 DRAINAGE VECTOR DESIGN",
            "🕹️ INTERACTIVE 3D WORLD",
        ]
    )

    # TAB 1: OVERVIEW & ML STATISTICS
    with tab_overview:
        c1, c2 = st.columns([3, 2])

        with c1:
            st.subheader("🤖 Random Forest Point Cloud Classification")

            # Pie Chart
            classes = ["Ground", "Vegetation", "Buildings", "Unclassified"]
            counts = [
                stats["ground_points"],
                stats["tree_points"],
                stats["building_points"],
                stats["unclassified_points"],
            ]
            colors = ["#4CAF50", "#8BC34A", "#FF9800", "#607D8B"]

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=classes,
                        values=counts,
                        hole=0.4,
                        marker=dict(colors=colors),
                        textinfo="percent+label",
                    )
                ]
            )
            fig_pie.update_layout(
                template="plotly_dark",
                margin=dict(t=30, b=0, l=0, r=0),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5
                ),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("📁 LiDAR File Registry Details")
            raw_data = rm.get_raw_metadata()

            for f in raw_data["files"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 12px;">
                        <span style="color: #00f0ff; font-weight: bold; font-family: monospace;">📄 {f["filename"]}</span><br/>
                        <span style="font-size: 0.9rem;">
                            ⚖️ File Size: <strong>{f["file_size_mb"]:.1f} MB</strong><br/>
                            🔢 Points: <strong>{f["point_count"]:,} pts</strong><br/>
                            🏷️ Las Specification: <strong>v{f["version"]}</strong>
                        </span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.subheader("💡 Estimated Biological & Structural Metrics")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric(
                    "Estimated Structures Found", f"{stats['estimated_houses']:,}"
                )
            with sc2:
                st.metric(
                    "Estimated Canopy Trees Identified", f"{stats['estimated_trees']:,}"
                )

    # TAB 2: DIGITAL TERRAIN MODEL (DTM)
    with tab_dtm:
        st.subheader("⛰️ Interpolated Ground Surface Elevation Map")
        st.markdown("""
            A 2D high-detail representation of the linear-interpolated ground level (ML ground predictions). 
            Values represent absolute orthometric elevation above mean sea level.
        """)

        # Plot 2D Elevation
        fig_3d = rm.get_3d_scene()

        # Expose the surface map data
        surface_trace = [t for t in fig_3d.data if t.type == "surface"][0]

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=surface_trace.z
                / 3.0,  # Remove vertical exaggeration for actual values
                x=surface_trace.x,
                y=surface_trace.y,
                colorscale="balance",
                colorbar=dict(title="Height (m)"),
            )
        )

        fig_heat.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=40, t=10, b=40),
            xaxis=dict(showgrid=False, title="UTM Easting Coordinate (m)"),
            yaxis=dict(showgrid=False, title="UTM Northing Coordinate (m)"),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # TAB 3: HYDROLOGICAL MAP SERIES
    with tab_maps:
        st.subheader("🗺️ Hydrological & Drainage Flow Series Maps")

        map_types = {
            "slope": "Terrain Slope Angle (Degrees)",
            "flow_dir": "D8 Flow Direction Topography (8 Cardinal Catchments)",
            "flow_acc": "Downstream Log Flow Accumulation Sinks",
            "flood_risk": "Waterlogging Hazards & Flood Risk Zones (TWI)",
        }

        selected_map = st.selectbox(
            "SELECT RASTER ANALYSIS LAYER",
            options=list(map_types.keys()),
            format_func=lambda x: map_types[x],
        )

        # Load map details
        map_res = rm.load_raster_data_arrays(selected_map)

        if map_res:
            mc1, mc2 = st.columns([3, 1])

            with mc1:
                # Decide styling colormaps
                colormaps = {
                    "slope": "earth",
                    "flow_dir": "twilight",
                    "flow_acc": "Blues",
                    "flood_risk": "YlGnBu",
                }

                # Dynamic Plotly Heatmap
                fig_map = go.Figure(
                    data=go.Heatmap(
                        z=map_res["data"],
                        colorscale=colormaps[selected_map],
                        colorbar=dict(title=map_types[selected_map]),
                    )
                )

                fig_map.update_layout(
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=10, b=40),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    height=520,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig_map, use_container_width=True)

            with mc2:
                # Display explanations
                st.markdown(f"**Layer Scope**: `{map_res['crs']}`")

                if selected_map == "slope":
                    st.markdown("""
                        **Terrain Slope Map**: Calculates local cell slope in degrees. 
                        Red and warm colors indicate steep cliffs/embankments, crucial for calculating stream runoff velocity.
                    """)
                elif selected_map == "flow_dir":
                    st.markdown("""
                        **D8 Flow Pointer**: Directs cell runoff toward the steepest neighbor. 
                        Useful to map topological flow direction angles (represented using cyclic hue scales).
                    """)
                elif selected_map == "flow_acc":
                    st.markdown("""
                        **Flow Accumulation**: Pinpoints the number of upstream cells draining into individual channels. 
                        Identifies water accumulation networks and micro-drain lines.
                    """)
                elif selected_map == "flood_risk":
                    st.markdown("""
                        **Topographic Wetness Index (TWI)**: Models potential water accumulation based on topography. 
                        Identifies high-risk marshy basins and village waterlogging dump sinks.
                    """)

                # If pre-baked presentation PNG map exists, display it for perfect display
                if is_processed:
                    png_mapping = {
                        "slope": "map5_3D_realistic_hillshade.png",  # hillshade is realistic representation
                        "flow_dir": "map2_flow_direction.png",
                        "flow_acc": "map1_flow_accumulation.png",
                        "flood_risk": "map3_flood_risk_twi.png",
                    }

                    img_path = os.path.join(rm.folder_path, png_mapping[selected_map])
                    if os.path.exists(img_path):
                        st.markdown("---")
                        st.markdown("**HQ Presentation Overlay:**")
                        st.image(
                            Image.open(img_path),
                            caption="Presentation Render View",
                            use_column_width=True,
                        )

        # High-Fidelity 8-Map Series Presentation Gallery
        if is_processed:
            st.markdown("---")
            st.subheader("🖼️ High-Fidelity 8-Map Series Gallery")
            st.markdown("""
                Access all 8 presentation-quality maps generated by the hydrological analysis engine.
                These maps are optimized for direct integration into engineering reports and hackathon evaluation.
            """)

            map_gallery_options = {
                "map1_flow_accumulation.png": "Map 1: Flow Accumulation (Downstream Runoff Concentrator)",
                "map2_flow_direction.png": "Map 2: D8 Flow Direction Pointer Topography",
                "map3_flood_risk_twi.png": "Map 3: Topographic Wetness Index (TWI) & Drainage Overlay",
                "map4_drainage_dump_zones.png": "Map 4: Drainage Network & Outlets Dump Zones",
                "map5_3D_realistic_hillshade.png": "Map 5: Realistic 3D Hillshade Terrain Relief",
                "map6_flow_routing_arrows.png": "Map 6: Flow Gradient Vector Routing",
                "map7_enhanced_flood_risk.png": "Map 7: Classified Enhanced Flood Risk Zones",
                "map8_detailed_accumulation.png": "Map 8: Runoff Accumulation Channels",
            }

            selected_gallery_map = st.selectbox(
                "SELECT PRESENTATION MAP TO VIEW",
                options=list(map_gallery_options.keys()),
                format_func=lambda x: map_gallery_options[x],
                key="gallery_selector",
            )

            gallery_img_path = os.path.join(rm.folder_path, selected_gallery_map)
            if os.path.exists(gallery_img_path):
                st.image(
                    Image.open(gallery_img_path),
                    caption=map_gallery_options[selected_gallery_map],
                    use_column_width=True,
                )
            else:
                st.info(
                    "💡 Run the integrated pipeline to generate this presentation map for the active region."
                )

    # TAB 4: DRAINAGE VECTOR DESIGN
    with tab_drainage:
        st.subheader("🌊 Polyline Drainage Network Shapefile Exporter")

        c_net1, c_net2 = st.columns([3, 2])

        with c_net1:
            st.markdown(
                f"""
                This module handles the export of vectorized drainage network shapefiles parsed directly 
                from D8 routing and GIS stream extraction algorithms. 
                
                - Estimated Total Drainage Segments: <strong>{stats["drainage_segments"]} channels</strong>
                - Design System: <strong>ESRI Shapefile (.shp, .dbf, .shx, .prj)</strong>
                - Integration Ready: <strong>QGIS, ArcGIS, CAD Systems</strong>
            """,
                unsafe_allow_html=True,
            )

            # Zip and offer download button for processed regions
            if is_processed:
                shp_basename = rm.config["drainage_shp"]
                zip_data = zip_shapefile_bundle(rm.folder_path, shp_basename)

                if zip_data:
                    st.success("✅ GIS Polyline Shapefiles bundled successfully!")
                    st.download_button(
                        label="📥 DOWNLOAD DRAINAGE NETWORK SHAPEFILES (ZIP)",
                        data=zip_data,
                        file_name=f"{selected_key}_drainage_network_shp.zip",
                        mime="application/zip",
                    )
                else:
                    st.error("Error: Could not pack shapefile bundle.")
            else:
                st.warning(
                    "⚠️ Shapefile download is restricted in dynamic simulation mode. Run hydrological pipeline to generate physical GIS shapefiles."
                )

        with c_net2:
            # Display high DPI vector map preview if exists
            if is_processed:
                img_path = os.path.join(rm.folder_path, "map4_drainage_dump_zones.png")
                if os.path.exists(img_path):
                    st.image(
                        Image.open(img_path),
                        caption="Vector Drainage Network & Dump Outlets Map",
                        use_column_width=True,
                    )

    # TAB 5: INTERACTIVE 3D WORLD
    with tab_3d:
        st.subheader("🕹️ Interactive 3D WebGL Terrain Explorer")
        st.markdown("""
            Rotate, zoom, and explore the 3D surface mesh elevation directly in the browser window.
            Blue lines trace the **topological water flow channels**, and red diamonds highlight critical **waterlogging flood hotspot sinks**.
        """)

        # Display Plotly 3D Figure directly
        fig_3d = rm.get_3d_scene()
        st.plotly_chart(fig_3d, use_container_width=True)

        # Display additional premium 3D screenshots if available in config
        if selected_key == "rajasthan" and "premium_pngs" in rm.config:
            st.markdown("---")
            st.subheader("📸 High-DPI Presentation Scenic Renders")

            # Show images side by side
            pngs = rm.config["premium_pngs"]
            cols = st.columns(3)

            image_keys = [
                "Ultra HD 3D Terrain",
                "Ground-Level 3D View",
                "Raised Slab 3D Geological Model",
            ]
            for idx, img_k in enumerate(image_keys):
                if img_k in pngs:
                    img_path = os.path.join(rm.folder_path, pngs[img_k])
                    if os.path.exists(img_path):
                        with cols[idx % 3]:
                            st.image(
                                Image.open(img_path),
                                caption=img_k,
                                use_column_width=True,
                            )


if __name__ == "__main__":
    main()
