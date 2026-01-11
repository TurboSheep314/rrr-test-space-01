import folium
from folium.plugins import HeatMap

def create_zip_heatmap(gdf, value_col, center=(42.3, -71.1), zoom=9):
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    heat_data = [
        [
            row.geometry.centroid.y,
            row.geometry.centroid.x,
            row[value_col]
        ]
        for _, row in gdf.iterrows()
        if row.geometry is not None
    ]

    HeatMap(
        heat_data,
        radius=25,
        blur=18,
        min_opacity=0.4
    ).add_to(m)

    return m