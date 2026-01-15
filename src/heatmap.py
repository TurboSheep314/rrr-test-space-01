import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
import branca.colormap as cm




def create_zip_heatmap(gdf, value_col, center=(42.3, -71.1), zoom=9):
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    colormap = cm.LinearColormap(
    colors=["blue", "cyan", "yellow", "orange", "red"],
    vmin=float(gdf[value_col].min()),
    vmax=float(gdf[value_col].max()),
    caption="Composite Score")

    # — OPTIONAL: choropleth fill by score — colors ZIP areas
    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=["Zip", value_col],
        key_on="feature.properties.Zip",
        fill_color="colormap",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=value_col,
    ).add_to(m)

    # Heat layer at centroids (optional)
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

    # Hover tooltips (popups) on top
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {"fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["Zip", "Composite Score"],
            aliases=["ZIP Code", "Composite Score"],
            localize=True
        ),
    ).add_to(m)

    # LEGEND (simple HTML box)
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background-color: white;
        padding: 8px;
        font-size: 14px;
        border: 1px solid #777;
    ">
        <strong>Composite Score</strong><br>
        Shaded = Higher Score
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(template)
    m.get_root().add_child(legend)

    return m