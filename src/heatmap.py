import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
import math


def create_zip_heatmap(gdf, value_col, center=(42.3, -71.1), zoom=9, featured_homes=None, selected_zips=None):
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    selected_zips = {str(zip_code).zfill(5) for zip_code in (selected_zips or [])}

    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=["Zip", value_col],
        key_on="feature.properties.Zip",
        fill_color="PuOr",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=value_col,
    ).add_to(m)

    # Heat layer at centroids (optional)
    # heat_data = [
    #     [
    #         row.geometry.centroid.y,
    #         row.geometry.centroid.x,
    #         row[value_col]
    #     ]
    #     for _, row in gdf.iterrows()
    #     if row.geometry is not None
    # ]
    heat_data = []
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        val = row.get(value_col)

        # Skip if score is None or NaN
        if val is None:
            continue
        try:
            # this will catch NaNs
            if math.isnan(float(val)):
                continue
        except Exception:
            continue

        heat_data.append([lat, lon, float(val)])
    HeatMap(
        heat_data,
        radius=0.25,
        blur=18,
        min_opacity=0.4
    ).add_to(m)

    # Hover tooltips (popups) on top
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {"fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["Zip", value_col],
            aliases=["ZIP Code", value_col],
            localize=True
        ),
    ).add_to(m)

    click_layer = folium.GeoJson(
        gdf,
        name="zip-click-layer",
        style_function=lambda feature: {
            "fillColor": "#9ca3af" if selected_zips and feature["properties"]["Zip"] not in selected_zips else "transparent",
            "color": "#2563eb" if feature["properties"]["Zip"] in selected_zips else "#00000000",
            "weight": 3 if feature["properties"]["Zip"] in selected_zips else 1,
            "fillOpacity": 0.45 if selected_zips and feature["properties"]["Zip"] not in selected_zips else 0.0,
        },
        highlight_function=lambda feature: {
            "weight": 3,
            "color": "#2563eb",
            "fillOpacity": 0.08,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["Zip", value_col],
            aliases=["ZIP Code", value_col],
            localize=True
        ),
    )
    click_layer.add_to(m)

    if featured_homes:
        for home in featured_homes:
            lat = home.get("latitude")
            lon = home.get("longitude")
            if lat is None or lon is None:
                continue

            popup_lines = [
                f"<strong>{home.get('address', 'Home Match')}</strong>",
                f"{home.get('city', '')} {home.get('zip', '')}".strip(),
                f"Price: ${home.get('price', 0):,.0f}" if home.get("price") is not None else "",
                f"{home.get('beds', '?')} bd | {home.get('baths', '?')} ba",
                f"{home.get('square_feet', 0):,.0f} sqft" if home.get("square_feet") is not None else "",
            ]
            if home.get("match_score") is not None:
                popup_lines.append(f"Match score: {home['match_score']:.3f}")

            popup_html = "<br>".join([line for line in popup_lines if line])
            folium.Marker(
                location=[lat, lon],
                tooltip=home.get("address", "Home Match"),
                popup=folium.Popup(popup_html, max_width=280),
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
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
        <strong>__VALUE_COL__</strong><br>
        Shaded = Higher Score
    </div>
    {% endmacro %}
    """
    template = template.replace("__VALUE_COL__", value_col)
    legend = MacroElement()
    legend._template = Template(template)
    m.get_root().add_child(legend)

    map_name = m.get_name()
    click_template = """
    {% macro script(this, kwargs) %}
    __CLICK_LAYER__.eachLayer(function(layer) {
        layer.on("click", function(e) {
            var zipCode = null;
            if (layer.feature && layer.feature.properties) {
                zipCode = layer.feature.properties.Zip;
            }
            if (!zipCode || !window.parent) return;
            window.parent.postMessage({
                type: "mapZipSelected",
                zip: zipCode
            }, "*");
        });
    });
    {% endmacro %}
    """
    click_template = click_template.replace("__MAP_NAME__", map_name)
    click_template = click_template.replace("__CLICK_LAYER__", click_layer.get_name())
    click_bridge = MacroElement()
    click_bridge._template = Template(click_template)
    m.get_root().add_child(click_bridge)

    return m
