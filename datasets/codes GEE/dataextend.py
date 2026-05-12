
import ee
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECTRAL_BANDS = ["Blue", "Green", "Red", "RE1", "RE2", "RE3", "RE4", "NIR", "SWIR1", "SWIR2"]
N_TIMESTEPS    = 36
N_SPECTRAL     = len(SPECTRAL_BANDS) * N_TIMESTEPS   # 360

GROWING_SEASON_START = "2021-04-10"   # ~DOY 100
GROWING_SEASON_END   = "2021-10-27"   # ~DOY 300

# Sampling scales — matched to each dataset's native resolution
ERA5_SCALE      = 11132  # ERA5-Land native pixel size (m)
SOIL_SCALE      = 250    # OpenLandMap native resolution (m)
TOPO_ELEV_SCALE = 1855   # ETOPO1 native pixel size (m)
TOPO_LF_SCALE   = 30     # CSP/ERGo US landforms (based on 10m NED DEM)

KELVIN_OFFSET   = 273.15


# ---------------------------------------------------------------------------
# Earth Engine initialisation
# ---------------------------------------------------------------------------
def init_ee(project=None):
    """Initialise Earth Engine, optionally specifying a cloud project."""
    try:
        if project:
            ee.Initialize(project='dataset-projet-1')
        else:
            ee.Initialize(project='dataset-projet-1')
        log.info("Earth Engine initialised successfully.")
    except Exception as exc:
        log.error(
            "EE initialisation failed. Run `earthengine authenticate` first.\n%s", exc
        )
        raise


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def parse_geo_column(geo_series):
    """
    Parse the .geo column (GeoJSON Point strings) into lon/lat columns.
    Accepts both string and dict representations.
    Returns a DataFrame with columns ['lon', 'lat'].
    """
    records = []
    for val in geo_series:
        if isinstance(val, str):
            val = val.replace("'", '"')
            geo = json.loads(val)
        else:
            geo = val
        lon, lat = geo["coordinates"]
        records.append({"lon": float(lon), "lat": float(lat)})
    return pd.DataFrame(records)


def df_to_feature_collection(df, coords):
    """
    Build an EE FeatureCollection from a DataFrame.
    Only sample_id, cropland, and row_index are carried as properties
    to keep serialisation fast — spectral columns are NOT sent to GEE.
    """
    features = []
    for i, (_, row) in enumerate(df.iterrows()):
        lon = coords.iloc[i]["lon"]
        lat = coords.iloc[i]["lat"]
        props = {
            "sample_id": int(row.get("sample_id", df.index[i])),
            "cropland":  int(row.get("cropland", -1)),
            "row_index": int(df.index[i]),
        }
        feat = ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), props)
        features.append(feat)
    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------------
# Image builders
# ---------------------------------------------------------------------------
def build_climate_image():
    """
    ERA5-Land Daily Aggregated — growing season 2021 (DOY 100-300).

    Dataset:  ECMWF/ERA5_LAND/DAILY_AGGR
    Coverage: 1950-01-02 to present  (~11 132 m resolution)

    Previous dataset ECMWF/ERA5/DAILY only covers up to 2020-07-09 and uses
    different band names — do NOT use it for 2021 data.

    Band name mapping (old ECMWF/ERA5/DAILY  →  new ERA5_LAND/DAILY_AGGR):
        mean_2m_air_temperature   →  temperature_2m
        total_precipitation       →  total_precipitation_sum
        dewpoint_2m_temperature   →  dewpoint_temperature_2m

    Output bands:
        mean_temp_growing      – growing-season mean 2-m air temperature (°C)
        total_precip_growing   – growing-season total precipitation      (mm)
        mean_dewpoint_growing  – growing-season mean dewpoint temp       (°C)
    """
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(GROWING_SEASON_START, GROWING_SEASON_END)
    )

    # Guard: fail immediately rather than letting arithmetic blow up on a
    # 0-band image (the symptom that caused the original error).
    count = era5.size().getInfo()
    if count == 0:
        raise RuntimeError(
            f"ERA5_LAND/DAILY_AGGR returned 0 images for "
            f"{GROWING_SEASON_START}–{GROWING_SEASON_END}. "
            "Check the date range and your EE project access."
        )
    log.info("  ERA5-Land collection: %d daily images in growing season.", count)

    mean_temp = (
        era5.select("temperature_2m")           # K
        .mean()
        .subtract(KELVIN_OFFSET)               # → °C
        .rename("mean_temp_growing")
    )
    total_precip = (
        era5.select("total_precipitation_sum")  # m (daily sum)
        .sum()
        .multiply(1000)                          # → mm
        .rename("total_precip_growing")
    )
    mean_dewpoint = (
        era5.select("dewpoint_temperature_2m")  # K
        .mean()
        .subtract(KELVIN_OFFSET)               # → °C
        .rename("mean_dewpoint_growing")
    )

    return ee.Image.cat([mean_temp, total_precip, mean_dewpoint])


def build_soil_image():
    """
    OpenLandMap static soil properties (250 m resolution).

    IMPORTANT: GEE asset IDs use a forward-slash before the version number:
        Correct:  OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02
        Wrong:    OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M_v02   ← underscore breaks lookup

    Output bands:
        soil_ph             – topsoil pH, 0-30 cm mean  (pH units, 0–14 scale)
        soil_organic_carbon – topsoil OC, 0-30 cm mean  (g/kg)
        soil_texture_class  – topsoil USDA texture class at 0 cm (categories 1–12)

    Depth layers used:
        b0  = 0 cm
        b10 = 10 cm
        b30 = 30 cm
    """
    # pH: native unit is pH × 10 → divide by 10
    ph = (
        ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
        .select(["b0", "b10", "b30"])
        .reduce(ee.Reducer.mean())
        .divide(10)
        .rename("soil_ph")
    )
    # Organic carbon: native unit is g/kg × 5 → multiply by 5 reverses that... wait,
    # native stored value = actual_g_per_kg / 5, so actual = stored × 5
    oc = (
        ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
        .select(["b0", "b10", "b30"])
        .reduce(ee.Reducer.mean())
        .multiply(5)
        .rename("soil_organic_carbon")
    )
    # Texture: categorical 1–12 (USDA classes); use topsoil layer only
    texture = (
        ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
        .select("b0")
        .rename("soil_texture_class")
    )

    return ee.Image.cat([ph, oc, texture])


def build_topo_image():
    """
    Static topography: ETOPO1 elevation + CSP/ERGo US landforms.

    Elevation:
        Dataset:  NOAA/NGDC/ETOPO1  (static image, ~1 855 m resolution)
        Band:     bedrock  (elevation in metres)

    Landforms:
        Dataset:  CSP/ERGo/1_0/US/landforms  (preferred over Global/ALOS_landforms
                  for CONUS because it uses the higher-resolution 10m USGS NED DEM)
        Band:     constant
        Values:   11–42  (NOT 1–11 — common mistake)
                  11 = Peak/ridge top
                  21 = Cliff
                  22 = Upper slope
                  23 = Middle slope
                  24 = Lower slope / flat slope
                  31 = Dry flat
                  32 = Moist flat
                  41 = Closed depression
                  42 = Wet flat / valley bottom

    Output bands:
        elevation      – bedrock elevation (m)
        landform_class – landform category (11–42)
    """
    elevation = (
        ee.Image("NOAA/NGDC/ETOPO1")
        .select("bedrock")
        .rename("elevation")
    )
    landforms = (
        ee.Image("CSP/ERGo/1_0/US/landforms")   # US NED-based; better than Global ALOS for CONUS
        .select("constant")
        .rename("landform_class")
    )

    return ee.Image.cat([elevation, landforms])


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------
def _sample_with_retry(image, fc, scale, band_names, description, max_retries=5, backoff=10.0):
    """
    Call image.sampleRegions on fc with retry / exponential back-off.
    Returns a DataFrame indexed by row_index.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result   = image.sampleRegions(collection=fc, scale=scale, geometries=False)
            features = result.getInfo()["features"]
            rows = []
            for feat in features:
                props = feat["properties"]
                row   = {"row_index": props["row_index"]}
                for band in band_names:
                    row[band] = props.get(band, np.nan)
                rows.append(row)
            return pd.DataFrame(rows).set_index("row_index").sort_index()
        except Exception as exc:
            log.warning("  Attempt %d/%d failed (%s): %s", attempt, max_retries, description, exc)
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                log.error("  All retries exhausted for %s. Filling with NaN.", description)
                n = fc.size().getInfo()
                df_nan = pd.DataFrame(np.nan, index=range(n), columns=band_names)
                df_nan.index.name = "row_index"
                return df_nan


def sample_image_batched(image, df, coords, scale, band_names, description, batch_size=1500):
    """
    Extract pixel values in batches to stay within GEE payload/memory limits.
    Returns a DataFrame indexed by the original df index.
    """
    n = len(df)
    parts = []

    for start in tqdm(range(0, n, batch_size), desc=f"  {description}"):
        end          = min(start + batch_size, n)
        idx_slice    = df.index[start:end]
        batch_df     = df.loc[idx_slice].copy()
        batch_coords = coords.iloc[start:end].copy()
        batch_coords.index = idx_slice

        fc_batch = df_to_feature_collection(batch_df, batch_coords)
        part     = _sample_with_retry(image, fc_batch, scale, band_names, description)
        parts.append(part)

    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
EXPECTED_RANGES = {
    "mean_temp_growing":    (-10,   45),
    "total_precip_growing": (  0, 3000),
    "mean_dewpoint_growing":(-20,   35),
    "soil_ph":              (  2,   11),
    "soil_organic_carbon":  (  0,  600),
    "soil_texture_class":   (  1,   12),
    "elevation":            (-500, 8850),
    "landform_class":       ( 11,   42),  # CSP/ERGo classes run 11–42, NOT 1–11
}


def validate_covariates(df, state):
    """Print a validation report for all environmental covariate columns."""
    log.info("── Validation: %s ──────────────────────────", state)
    env_cols = [c for c in EXPECTED_RANGES if c in df.columns]
    total    = len(df)

    for col in env_cols:
        n_miss = df[col].isnull().sum()
        pct    = n_miss / total * 100
        flag   = "⚠ " if pct > 1 else "✓ "
        log.info("  %s%-30s  missing=%d/%d (%.1f%%)", flag, col, n_miss, total, pct)

    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        out = ((df[col] < lo) | (df[col] > hi)).sum()
        if out:
            log.warning("  ⚠  %-30s  %d values outside expected [%s, %s]", col, out, lo, hi)

    log.info("  Statistics:\n%s", df[env_cols].describe().to_string())


# ---------------------------------------------------------------------------
# Main pipeline per state
# ---------------------------------------------------------------------------
def process_state(csv_path, state_name, batch_size, output_dir, drive_folder):
    """
    Full pipeline for one state:
      1. Load existing spectral CSV
      2. Parse coordinates from .geo column
      3. Extract 8 environmental covariates via GEE
      4. Merge, validate, and save
    """
    log.info("═══════════════════════════════════════════")
    log.info("Processing: %s  (%s)", state_name, csv_path)
    log.info("═══════════════════════════════════════════")

    # 1. Load
    df = pd.read_csv(csv_path)
    log.info("  Loaded %d rows × %d columns.", len(df), len(df.columns))
    if ".geo" not in df.columns:
        raise ValueError(f"'.geo' column not found in {csv_path}.")

    # 2. Parse coordinates
    log.info("  Parsing coordinates …")
    coords = parse_geo_column(df[".geo"])
    coords.index = df.index

    # 3. Build EE images (lazy server-side definitions — no data transferred yet)
    climate_img = build_climate_image()
    soil_img    = build_soil_image()
    topo_img    = build_topo_image()

    # 4. Extract covariates
    climate_bands = ["mean_temp_growing", "total_precip_growing", "mean_dewpoint_growing"]
    soil_bands    = ["soil_ph", "soil_organic_carbon", "soil_texture_class"]
    topo_bands    = ["elevation", "landform_class"]

    log.info("  ── Climate (ERA5-Land) ──────────────────")
    df_climate = sample_image_batched(
        climate_img, df, coords, ERA5_SCALE, climate_bands, "Climate", batch_size
    )

    log.info("  ── Soil (OpenLandMap) ───────────────────")
    df_soil = sample_image_batched(
        soil_img, df, coords, SOIL_SCALE, soil_bands, "Soil", batch_size
    )

    log.info("  ── Topography ───────────────────────────")
    # Elevation and landforms have very different native scales; extract separately
    # to avoid resampling artefacts.
    df_elev = sample_image_batched(
        topo_img.select("elevation"), df, coords,
        TOPO_ELEV_SCALE, ["elevation"], "Elevation", batch_size
    )
    df_lf = sample_image_batched(
        topo_img.select("landform_class"), df, coords,
        TOPO_LF_SCALE, ["landform_class"], "Landforms", batch_size
    )
    df_topo = pd.concat([df_elev, df_lf], axis=1)

    # 5. Merge
    log.info("  Merging …")
    spectral_cols = [c for c in df.columns if any(c.startswith(b + "_t") for b in SPECTRAL_BANDS)]
    valid_cols    = [c for c in df.columns if c.startswith("valid_t")]
    
    geo_col       = [".geo"] if ".geo" in df.columns else []
    meta_cols = [c for c in ["label", "sample_id", "system:index"] if c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("dummy")], errors="ignore")
    keep = meta_cols + spectral_cols + valid_cols + geo_col
    df_merged = df[keep].join(df_climate).join(df_soil).join(df_topo)

    n_env = sum(1 for c in climate_bands + soil_bands + topo_bands if c in df_merged.columns)
    log.info(
        "  Result: %d rows × %d columns  (%d spectral + %d env covariates)",
        len(df_merged), len(df_merged.columns), len(spectral_cols), n_env,
    )

    # 6. Validate
    validate_covariates(df_merged, state_name)

    # 7. Save locally
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"{state_name}_2021_Extended.csv"
    out_path = Path(output_dir) / out_name
    df_merged.to_csv(out_path, index=False)
    log.info("  Saved → %s", out_path)

    # 8. Optional GEE Drive export (env covariates only — spectral data stays local)
    if drive_folder:
        export_full_to_drive(df_merged, state_name, drive_folder)

    return df_merged


# ---------------------------------------------------------------------------
# Optional Drive export
# ---------------------------------------------------------------------------
def export_full_to_drive(df_merged, state_name, drive_folder, chunk_size=3334):
    """Export complete merged dataset to Drive in 3 chunks to stay under GEE's 10MB limit."""

    tasks = []
    chunks = [df_merged.iloc[i:i+chunk_size] for i in range(0, len(df_merged), chunk_size)]
    log.info("  Splitting into %d chunks of ~%d rows …", len(chunks), chunk_size)

    for chunk_idx, chunk in enumerate(chunks, 1):
        log.info("  Building FeatureCollection for chunk %d/%d …", chunk_idx, len(chunks))

        features = []
        for _, row in chunk.iterrows():
            try:
                geo = json.loads(str(row[".geo"]).replace("'", '"'))
                lon, lat = geo["coordinates"]
            except Exception:
                continue

            props = {}
            for col in df_merged.columns:
                if col == ".geo":
                    continue
                v = row[col]
                props[col] = None if pd.isna(v) else v

            features.append(ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), props))

        fc = ee.FeatureCollection(features)
        file_prefix = f"{state_name}_2021_Extended_part{chunk_idx}"

        task = ee.batch.Export.table.toDrive(
            collection     = fc,
            description    = file_prefix,
            folder         = drive_folder,
            fileNamePrefix = file_prefix,
            fileFormat     = "CSV",
        )
        task.start()
        tasks.append((chunk_idx, task))
        log.info("  ✓ Chunk %d export task started: %s", chunk_idx, task.id)

    # Wait for all 3 tasks
    log.info("  Waiting for all chunks to finish …")
    log.info("  Monitor: https://code.earthengine.google.com/tasks")

    while tasks:
        still_running = []
        for chunk_idx, task in tasks:
            state = task.status()["state"]
            if state == "COMPLETED":
                log.info("  ✓ Chunk %d complete → Drive/%s/%s_2021_Extended_part%d.csv",
                         chunk_idx, drive_folder, state_name, chunk_idx)
            elif state in ("FAILED", "CANCELLED"):
                log.error("  ✗ Chunk %d %s: %s",
                          chunk_idx, state, task.status().get("error_message", ""))
            else:
                still_running.append((chunk_idx, task))

        tasks = still_running
        if tasks:
            log.info("  Still running: chunks %s — checking again in 30s …",
                     [i for i, _ in tasks])
            time.sleep(30)

    log.info("  All chunks done. Files in Drive folder: %s", drive_folder)
    log.info("  %s_2021_Extended_part1.csv  (~3334 rows)", state_name)
    log.info("  %s_2021_Extended_part2.csv  (~3334 rows)", state_name)
    log.info("  %s_2021_Extended_part3.csv  (~3332 rows)", state_name)
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate extended crop-mapping dataset with environmental covariates."
    )
    p.add_argument("--arkansas",    default="MCTNet_Arkansas_2021.csv")
    p.add_argument("--california",  default="MCTNet_California_2021.csv")
    p.add_argument("--output_dir",  default="output")
    p.add_argument("--drive_folder",default=None,
                   help="Google Drive folder for optional GEE export (env covariates only)")
    p.add_argument("--batch_size",  type=int, default=1500,
                   help="Points per GEE sampleRegions call (default 1500)")
    p.add_argument("--project",     default=None,
                   help="GEE cloud project ID")
    p.add_argument("--states",      nargs="+", default=["arkansas", "california"],
                   choices=["arkansas", "california"])
    return p.parse_args()


def main():
    args = parse_args()
    init_ee(project=args.project)

    state_map = {
        "arkansas":   ("Arkansas",   args.arkansas),
        "california": ("California", args.california),
    }

    results = {}
    for key in args.states:
        label, csv_path = state_map[key]
        if not Path(csv_path).exists():
            log.warning("File not found, skipping %s: %s", label, csv_path)
            continue
        results[label] = process_state(
            csv_path     = csv_path,
            state_name   = label,
            batch_size   = args.batch_size,
            output_dir   = args.output_dir,
            drive_folder = args.drive_folder,
        )

    # Summary
    log.info("══════════════ Summary ══════════════")
    env_cols = [
        "mean_temp_growing", "total_precip_growing", "mean_dewpoint_growing",
        "soil_ph", "soil_organic_carbon", "soil_texture_class",
        "elevation", "landform_class",
    ]
    for state, df in results.items():
        spectral = sum(1 for c in df.columns if any(c.startswith(b + "_t") for b in SPECTRAL_BANDS))
        env      = sum(1 for c in env_cols if c in df.columns)
        log.info(
            "  %-12s  rows=%-6d  spectral=%-4d  env=%-2d  total_features=%d",
            state, len(df), spectral, env, spectral + env,
        )
    log.info("Done.")


if __name__ == "__main__":
    main()