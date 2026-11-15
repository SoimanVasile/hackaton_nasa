import os
import ee
import pandas as pd

# Initialize Google Earth Engine
GEE_PROJECT_ID = "hazel-env-513413-n9"
ee.Initialize(project=GEE_PROJECT_ID)

# Overwrites ONLY urban_aquisition.parquet (Las Vegas + Lake Mead), keeps Aral & Alaska intact
FORCE_OVERWRITE = {"urban_aquisition"}

REGIONS = {
    "sea_aquisition": {
        "bbox": [58.00, 43.30, 61.80, 46.80],
        "scale": 1500  # Aral Sea: water desiccation (NDWI) & salinity
    },
    "urban_aquisition": {
        # Centered on Las Vegas / Henderson metro + full Lake Mead & Overton Arm
        "bbox": [-115.48, 35.82, -114.05, 36.75],
        "scale": 750   # ~23,000 cells/yr (~1M rows total)
    },
    "snow_ice_aquisition": {
        "bbox": [-148.05, 60.90, -146.65, 61.50],
        "scale": 600   # Harvard & Columbia Glaciers, Alaska: ice retreat (NDSI)
    }
}


def mask_clouds(img):
    """Mask clouds and cloud shadows using QA_PIXEL bitmask."""
    qa = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)


def prep_l57(img):
    """Harmonize Landsat 5/7 Collection 2 surface reflectance and thermal bands."""
    img = mask_clouds(img)
    sr = img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'],
        ['blue', 'green', 'red', 'nir', 'swir1']
    ).multiply(0.0000275).add(-0.2)
    lst = img.select(['ST_B6'], ['lst_c']).multiply(0.00341802).add(149.0).subtract(273.15)
    return sr.addBands(lst).copyProperties(img, ['system:time_start'])


def prep_l89(img):
    """Harmonize Landsat 8/9 Collection 2 surface reflectance and thermal bands."""
    img = mask_clouds(img)
    sr = img.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
        ['blue', 'green', 'red', 'nir', 'swir1']
    ).multiply(0.0000275).add(-0.2)
    lst = img.select(['ST_B10'], ['lst_c']).multiply(0.00341802).add(149.0).subtract(273.15)
    return sr.addBands(lst).copyProperties(img, ['system:time_start'])


def add_spectral_indices(img):
    """Compute NDWI (water), NDBI (urban infrastructure), NDSI (ice), and Salinity."""
    ndwi = img.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndbi = img.normalizedDifference(['swir1', 'nir']).rename('ndbi')
    ndsi = img.normalizedDifference(['green', 'swir1']).rename('ndsi')
    salinity = img.select('blue').multiply(img.select('red')).sqrt().rename('salinity_index')

    keep_bands = ['ndwi', 'ndbi', 'ndsi', 'salinity_index', 'lst_c']
    return img.addBands([ndwi, ndbi, ndsi, salinity]).select(keep_bands)


def get_merged_collection(geom):
    """Merge Landsat 5, 7, 8, and 9 collections over a bounding box."""
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(geom).map(prep_l57)
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(geom).map(prep_l57)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom).map(prep_l89)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(geom).map(prep_l89)
    return l5.merge(l7).merge(l8).merge(l9).map(add_spectral_indices)


def export_region_to_parquet(dataset_name, bbox, scale):
    """Sample annual summer median grids and export to a Parquet file."""
    out_file = f"{dataset_name}.parquet"
    if dataset_name not in FORCE_OVERWRITE and os.path.exists(out_file):
        print(f"Skipping {out_file} (already exists).")
        return

    geom = ee.Geometry.Rectangle(bbox)
    col = get_merged_collection(geom)
    yearly_dfs = []

    print(f"Starting extraction for: {out_file}...")
    for year in range(1984, 2027):
        yearly_col = col.filterDate(f"{year}-06-01", f"{year}-09-30")
        composite = yearly_col.median().clip(geom).addBands(ee.Image.pixelLonLat())

        samples = composite.sample(
            region=geom,
            scale=scale,
            projection='EPSG:4326',
            geometries=False
        )

        df_year = ee.data.computeFeatures({
            'expression': samples,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        if not df_year.empty and 'ndwi' in df_year.columns:
            df_year['year'] = year
            if 'geo' in df_year.columns:
                df_year = df_year.drop(columns=['geo'])
            yearly_dfs.append(df_year)

        print(f"  [{dataset_name}] Year {year} -> {len(df_year)} cells")

    full_df = pd.concat(yearly_dfs, ignore_index=True)
    full_df.to_parquet(out_file, index=False)
    print(f"Saved {out_file} ({len(full_df)} total rows)!\n")


if __name__ == "__main__":
    for name, cfg in REGIONS.items():
        export_region_to_parquet(name, cfg["bbox"], cfg["scale"])