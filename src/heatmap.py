import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement

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

    folium.GeoJson(
    gdf,
    style_function=lambda feature: {"fillOpacity": 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["Zip", "Composite Score"],
        aliases=["ZIP Code", "Composite Score"],
        localize=True
    ),
    ).add_to(m)
    
    

    # legend HTML
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
        Hotter = Higher Score
    </div>
    {% endmacro %}
    """

    legend = MacroElement()
    legend._template = Template(template)

    m.get_root().add_child(legend)

   

    return m