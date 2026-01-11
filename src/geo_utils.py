import geopandas as gpd

# def load_zip_shapes(path: str, zip_col="ZCTA5CE10"):
#     gdf = gpd.read_file(path)

#     gdf["Zip"] = gdf[zip_col].astype(str).str.zfill(5)
#     return gdf[["Zip", "geometry"]]


def load_zip_shapes(path: str):
    gdf = gpd.read_file(path)

    # Auto-detect ZIP/ZCTA field across Census years/products
    for candidate in ["ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID"]:
        if candidate in gdf.columns:
            zip_col = candidate
            break
    else:
        raise ValueError(f"No ZIP/ZCTA column found. Columns: {list(gdf.columns)}")

    gdf["Zip"] = gdf[zip_col].astype(str).str.zfill(5)
    return gdf[["Zip", "geometry"]]