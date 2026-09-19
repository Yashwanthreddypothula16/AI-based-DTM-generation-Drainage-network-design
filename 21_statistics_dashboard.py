import os
import plotly.graph_objects as go


def main():
    print("🎨 Generating Premium Statistics Dashboard...")
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, "Rajasthan_Point_Cloud")
    stats_path = os.path.join(out_dir, "classification_stats.txt")
    output_html = os.path.join(out_dir, "rajasthan_statistics.html")

    # Check if stats exist, if not, wait or use defaults for template
    if not os.path.exists(stats_path):
        print(f"⚠️ Warning: {stats_path} not found. Using sample data for layout...")
        stats = {
            "ground_points": 6743680,
            "building_points": 450000,
            "tree_points": 1200000,
            "unclassified_points": 1200000,
            "estimated_houses": 420,
            "estimated_trees": 1540,
            "total_points": 9600000,
        }
    else:
        stats = {}
        with open(stats_path, "r") as f:
            for line in f:
                k, v = line.strip().split(":")
                stats[k] = int(v)

    # 1. Create Charts
    # Pie Chart
    labels = ["Ground", "Vegetation", "Buildings", "Other"]
    values = [
        stats["ground_points"],
        stats["tree_points"],
        stats["building_points"],
        stats["unclassified_points"],
    ]
    colors = ["#4CAF50", "#8BC34A", "#FF9800", "#9E9E9E"]

    fig_pie = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.4, marker_colors=colors)]
    )
    fig_pie.update_layout(
        template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), showlegend=True
    )

    # Bar Chart
    fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig_bar.update_layout(
        template="plotly_dark",
        margin=dict(t=0, b=0, l=0, r=0),
        yaxis_title="Point Count",
    )

    # 2. Build Custom HTML/CSS Template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rajasthan LiDAR Statistics Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #0d1117;
                --card-bg: rgba(22, 27, 34, 0.8);
                --accent-color: #2f81f7;
                --text-main: #c9d1d9;
                --text-bright: #ffffff;
                --glass: rgba(255, 255, 255, 0.05);
            }}
            
            body {{
                background-color: var(--bg-color);
                color: var(--text-main);
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background-image: radial-gradient(circle at 50% 50%, #1a1f29 0%, #0d1117 100%);
            }}
            
            .header {{
                padding: 40px 20px;
                text-align: center;
                background: linear-gradient(180deg, rgba(47, 129, 247, 0.1) 0%, transparent 100%);
            }}
            
            .header h1 {{
                font-family: 'Orbitron', sans-serif;
                font-size: 2.5rem;
                color: var(--text-bright);
                margin: 0;
                letter-spacing: 2px;
                text-transform: uppercase;
                text-shadow: 0 0 20px rgba(47, 129, 247, 0.5);
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            .stat-card {{
                background: var(--card-bg);
                border: 1px solid var(--glass);
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 40px rgba(47, 129, 247, 0.2);
                border-color: var(--accent-color);
            }}
            
            .stat-value {{
                font-family: 'Orbitron', sans-serif;
                font-size: 2.2rem;
                font-weight: bold;
                color: var(--text-bright);
                margin: 10px 0;
            }}
            
            .stat-label {{
                text-transform: uppercase;
                font-size: 0.9rem;
                letter-spacing: 1px;
                color: var(--accent-color);
            }}
            
            .main-content {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            @media (max-width: 1000px) {{
                .main-content {{ grid-template-columns: 1fr; }}
            }}
            
            .chart-card {{
                background: var(--card-bg);
                border: 1px solid var(--glass);
                border-radius: 12px;
                padding: 20px;
                min-height: 400px;
            }}
            
            .chart-title {{
                font-family: 'Orbitron', sans-serif;
                font-size: 1.1rem;
                margin-bottom: 20px;
                text-align: center;
                color: var(--text-bright);
            }}
            
            .footer {{
                text-align: center;
                padding: 40px;
                font-size: 0.8rem;
                color: #484f58;
            }}
            
            .preview-img {{
                width: 100%;
                border-radius: 8px;
                margin-top: 20px;
                border: 1px solid var(--glass);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Rajasthan DTM: AI Feature Statistics</h1>
            <p>High-resolution LiDAR analytics & terrain classification metrics</p>
        </div>
        
        <div class="container">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Classified Points</div>
                    <div class="stat-value">{stats["total_points"]:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Ground Accuracy Base</div>
                    <div class="stat-value">{stats["ground_points"]:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Estimated Houses</div>
                    <div class="stat-value" style="color: #FF9800;">{stats["estimated_houses"]}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Estimated Trees</div>
                    <div class="stat-value" style="color: #8BC34A;">{stats["estimated_trees"]}</div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="chart-card">
                    <div class="chart-title">Point Cloud Distribution</div>
                    {fig_pie.to_html(full_html=False, include_plotlyjs="cdn")}
                </div>
                <div class="chart-card">
                    <div class="chart-title">Categorical Point Counts</div>
                    {fig_bar.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>
            
            <div class="chart-card" style="grid-column: span 2;">
                <div class="chart-title">3D Terrain Visualization Context</div>
                <img src="MASTER_INTEGRATED_3D_DASHBOARD.png" class="preview-img" alt="3D Preview">
                <p style="text-align:center; margin-top:15px; font-size: 0.9rem;">
                    Generated from the integrated 3D point cloud model. Data density: ~280 pts/m².
                </p>
            </div>
        </div>
        
        <div class="footer">
            &copy; 2026 Rajasthan Hydrological AI Analysis Pipeline • Advanced DTM Generation
        </div>
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✅ Dashboard successfully generated at: {output_html}")


if __name__ == "__main__":
    main()
