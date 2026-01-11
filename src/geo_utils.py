import geopandas as gpd

def load_zip_shapes(path: str, zip_col="ZCTA5CE10"):
    gdf = gpd.read_file(path)

    gdf["Zip"] = gdf[zip_col].astype(str).str.zfill(5)
    return gdf[["Zip", "geometry"]]