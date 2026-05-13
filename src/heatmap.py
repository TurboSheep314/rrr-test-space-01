import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
import branca.colormap as cm
import math


def create_zip_heatmap(gdf, value_col, center=(42.3, -71.1), zoom=9, featured_homes=None, selected_zips=None):
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    selected_zips = {str(zip_code).zfill(5) for zip_code in (selected_zips or [])}

    tooltip_fields = ["Zip", value_col]
    tooltip_aliases = ["ZIP Code", value_col]
    for field, alias in [
        ("market_signal", "Market Signal"),
        ("market_interpretation", "Interpretation"),
    ]:
        if field in gdf.columns:
            tooltip_fields.append(field)
            tooltip_aliases.append(alias)

    valid_scores = []
    if value_col in gdf.columns:
        for value in gdf[value_col].tolist():
            try:
                value = float(value)
                if not math.isnan(value):
                    valid_scores.append(value)
            except Exception:
                continue

    max_abs = max((abs(v) for v in valid_scores), default=1.0)
    if max_abs == 0:
        max_abs = 1.0

    palette = [
        "#b35806",
        "#e67e22",
        "#f5c58d",
        "#f5f2ea",
        "#c9e3f2",
        "#5aa2d1",
        "#5b2a86",
    ]
    bounds = [(-max_abs + (2 * max_abs * i / 7.0)) for i in range(8)]
    score_colormap = cm.StepColormap(
        colors=palette,
        index=bounds,
        vmin=-max_abs,
        vmax=max_abs,
        caption=f"{value_col} (Centered on Home Match)",
    )

    def base_style(feature):
        value = feature["properties"].get(value_col)
        try:
            value = float(value)
            if math.isnan(value):
                raise ValueError
            fill = score_colormap(value)
        except Exception:
            fill = "#d1d5db"
        return {
            "fillColor": fill,
            "fillOpacity": 0.7,
            "color": "#6b7280",
            "weight": 0.7,
            "opacity": 0.45,
        }

    folium.GeoJson(
        gdf,
        name="zip-base-layer",
        style_function=base_style,
    ).add_to(m)
    score_colormap.add_to(m)

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

    pricing_fields = []
    pricing_aliases = []
    for field, alias in [
        ("sale_count", "Sale Count"),
        ("price_q1", "Price Q1"),
        ("price_median", "Price Median"),
        ("price_mean", "Price Mean"),
        ("price_skew_direction", "Price Skew Direction"),
        ("affordability_skew_index", "Affordability Skew Index"),
        ("market_signal", "Market Signal"),
        ("market_interpretation", "Interpretation"),
        ("market_type", "Market Type"),
    ]:
        if field in gdf.columns:
            pricing_fields.append(field)
            pricing_aliases.append(alias)

    # Hover tooltips on top
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {"fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
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
            fields=tooltip_fields,
            aliases=tooltip_aliases,
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
