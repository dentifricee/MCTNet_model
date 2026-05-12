import ee

ee.Initialize(project='dataset-projet-1')

YEAR        = 2021
START_DATE  = ee.Date.fromYMD(YEAR, 1, 1)
N_STEPS     = 36
STEP_DAYS   = 10
CHUNK_SIZE  = 12
N_SAMPLES   = 10000
RANDOM_SEED = 123
SCALE       = 10

BANDS       = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
BAND_LABELS = ['Blue','Green','Red','RE1','RE2','RE3','NIR','RE4','SWIR1','SWIR2']

states     = ee.FeatureCollection('TIGER/2018/States')
arkansas   = states.filter(ee.Filter.eq('NAME', 'Arkansas')).first().geometry()
california = states.filter(ee.Filter.eq('NAME', 'California')).first().geometry()

def mask_s2_clouds(image):
    scl  = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return (image
            .updateMask(mask)
            .select(BANDS)
            .divide(10000)
            .toFloat()
            .copyProperties(image, ['system:time_start']))

end_date = START_DATE.advance(N_STEPS * STEP_DAYS, 'day')

s2_global = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate(START_DATE, end_date)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
               .map(mask_s2_clouds))

worldcover    = ee.ImageCollection('ESA/WorldCover/v200').first()
cropland_mask = worldcover.eq(40)

cdl_2021  = (ee.ImageCollection('USDA/NASS/CDL')
               .filter(ee.Filter.calendarRange(2021, 2021, 'year'))
               .first())
cdl_crop  = cdl_2021.select('cropland')
cdl_conf  = cdl_2021.select('confidence')
conf_95   = cdl_conf.gte(95)

ARK_FROM  = [5, 3, 1, 2]
ARK_TO    = [0, 1, 2, 3]
ARK_OTHER = 4

CAL_FROM  = [69, 3, 36, 75, 204]
CAL_TO    = [0,  1,  2,  3,   4]
CAL_OTHER = 5

def sample_locations(aoi, cdl_from, cdl_to, other_class):
    label = cdl_crop.remap(cdl_from, cdl_to, other_class).rename('label')
    valid_area_mask = conf_95.And(cropland_mask)
    label = label.updateMask(valid_area_mask).toByte()
    dummy = ee.Image.constant(1).rename('dummy').updateMask(label.mask())
    to_sample = label.addBands(dummy)
    pts = to_sample.stratifiedSample(
        numPoints   = N_SAMPLES,
        classBand   = 'dummy',
        region      = aoi,
        scale       = 30,
        seed        = RANDOM_SEED,
        geometries  = True,
        dropNulls   = True
    )
    pts_with_id = pts.map(lambda f: f.set('sample_id', f.id()))
    return pts_with_id

def extract_chunk(aoi, sample_pts, step_start, step_end):
    s2_aoi       = s2_global.filterBounds(aoi)
    step_indices = ee.List.sequence(step_start, step_end)

    def build_step_image(step_idx):
        step_idx  = ee.Number(step_idx)
        t0        = START_DATE.advance(step_idx.multiply(STEP_DAYS), 'day')
        t1        = t0.advance(STEP_DAYS, 'day')
        median    = s2_aoi.filterDate(t0, t1).median()
        valid     = median.select('B2').mask().unmask(0).rename('valid')
        spectral = median.rename(BANDS)  # keeps NaN where no data
        step_img  = (spectral
                     .addBands(valid)
                     .updateMask(cropland_mask)
                     .toFloat())
        sfx       = ee.String('t').cat(step_idx.add(1).format('%02d'))
        new_names = (ee.List(BAND_LABELS)
                       .add('valid')
                       .map(lambda lbl: ee.String(lbl).cat('_').cat(sfx)))
        return step_img.rename(new_names)

    step_images     = step_indices.map(build_step_image)
    col             = ee.ImageCollection.fromImages(step_images)
    stacked         = col.toBands()
    old_names       = stacked.bandNames()
    new_names_clean = old_names.map(
        lambda name: ee.String(name).replace('^[^_]+_', '')
    )
    stacked = stacked.rename(new_names_clean)

    return stacked.reduceRegions(
        collection = sample_pts,
        reducer    = ee.Reducer.mean(),
        scale      = SCALE,
        tileScale  = 4
    )

ark_pts = sample_locations(arkansas, ARK_FROM, ARK_TO, ARK_OTHER)
cal_pts = sample_locations(california, CAL_FROM, CAL_TO, CAL_OTHER)

ark_chunk1 = extract_chunk(arkansas, ark_pts,  0, 11)
ark_chunk2 = extract_chunk(arkansas, ark_pts, 12, 23)
ark_chunk3 = extract_chunk(arkansas, ark_pts, 24, 35)

cal_chunk1 = extract_chunk(california, cal_pts,  0, 11)
cal_chunk2 = extract_chunk(california, cal_pts, 12, 23)
cal_chunk3 = extract_chunk(california, cal_pts, 24, 35)

export_tasks = [
    (ark_chunk1, 'MCTNet_Arkansas_2021_Chunk1'),
    (ark_chunk2, 'MCTNet_Arkansas_2021_Chunk2'),
    (ark_chunk3, 'MCTNet_Arkansas_2021_Chunk3'),
    (cal_chunk1, 'MCTNet_California_2021_Chunk1'),
    (cal_chunk2, 'MCTNet_California_2021_Chunk2'),
    (cal_chunk3, 'MCTNet_California_2021_Chunk3'),
]

for collection, description in export_tasks:
    task = ee.batch.Export.table.toDrive(
        collection  = collection,
        description = description,
        folder      = 'EarthEngine_Exports',
        fileFormat  = 'CSV'
    )
    task.start()
    print(f'Started: {description}')