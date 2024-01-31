import os
import numpy as np
import rasterio
from cropmaps.get_creodias import get_data, eodata_path_creator
from cropmaps.sts import sentimeseries
from cropmaps.cube import generate_cube_paths, make_cube
from cropmaps.models import random_forest_train, random_forest_predict, save_model, LandCover_Masking, load_model, random_forest_predict_patches
from cropmaps.prepare_vector import burn

def local_DIAS_path_creator(image):
    if (image.satellite == "Sentinel-2A") or (image.satellite == "Sentinel-2B"):
        satellite = "Sentinel-2"
    else:
        raise ValueError("Satellite name unkwown!")

    if image.processing_level == "Level-2A" or image.processing_level == "Level-2Ap":
        level = "L2A"
    elif image.processing_level == "Level-1C":
        level = "L1C"
    else:
        raise ValueError("Processing level unkwown!")

    date = image.datetime
    year = str(date.year)
    month = str(date.month).zfill(2)
    day = str(date.day).zfill(2)

    return os.path.join(satellite, level, year, month, day)

search = "D:\Work\Development_Projects\cropmaps\France\Raw"
store = "D:\Work\Development_Projects\cropmaps\France\eodata"
AOI = "D:\Work\Development_Projects\cropmaps\France\AOI\AOI.geojson"

# Get data
eodata = sentimeseries("S2-timeseries")
eodata.find(search)
eodata.sort_images(date=True)

creodias_paths = []
for image in eodata.data:
    src = os.path.join(image.path, image.name)
    DIAS_path = local_DIAS_path_creator(image)
    creodias_paths.append(os.path.join(store, DIAS_path, image.name))

# Reproducing a DIAS enviroment where we can not write inside the data folder from the moment we have the data
products_path = "D:\Work\Development_Projects\cropmaps\France\eodata_local"
if not os.path.exists(products_path):
    os.makedirs(products_path)

# Create a timeseries with the data from query
eodata = sentimeseries("S2-timeseries")
eodata.find_DIAS(creodias_paths)

eodata.sort_images(date=True)

eodata.filter_dates("01102018", min_date = "01112017")


dates_to_remove = ["29112017", "24122017", "23012018", "12022018", "12022018", "07072018", "27072018",  "06082018", "16082018", "21082018", "31082018", "31082018"]
eodata.remove_date(dates_to_remove)

# First do the masking for all the available bands
# | Band | Resolution (m) | Resize | Final Resolution |
# | 02   | 10             | True   | 10               |
# | 03   | 10             | True   | 10               |
# | 04   | 10             | True   | 10               |
# | 05   | 20             | True   | 10               |
# | 06   | 20             | True   | 10               |
# | 07   | 20             | True   | 10               |
# | 08   | 10             | True   | 10               |
# | 8A   | 20             | True   | 10               |
# | 11   | 20             | True   | 10               |
# | 12   | 20             | True   | 10               |

eodata.clipbyMask(shapefile = AOI, store = products_path)
eodata.clipbyMask(shapefile = AOI, store = products_path, resize = True)
# If you want to perform the analysis on less bands just add the band name to the band argument
# as in here: eodata.clipbyMask(shapefile = mask, store = products_path, band = "B05", resize = True)

# Calculate vegetation indexes
eodata.getVI("NDVI", store = products_path, subregion = "AOI") # Subregion is the name of the mask shapefile
eodata.getVI("NDWI", store = products_path, subregion = "AOI")
eodata.getVI("NDBI", store = products_path, subregion = "AOI") # This is in 20m

# Do the same for the vegetation indexes
# NOTE: CLIPPED DATA BY DEFAULT ARE SAVED AS AN ATTRIBUTE OF THE S2IMAGE OBJECT AS image.BAND[RESOLUTION][MASK_NAME]
# SO IN THIS CASE TO ACCESS THE ATTRIBUTE OF IMAGE 0, BAND B12 YOU CAN write: eodata.data[0].B12["10"]["AOI"]
eodata.upsample(store = products_path, band = "NDBI", subregion = "AOI") # Transforming to 10m

# Apply SCL cloud masks
eodata.apply_SCL(store = products_path, resolution="highest", subregion="AOI") # This performs only the default bands
eodata.apply_SCL(store = products_path, band = "NDVI", resolution="highest", subregion="AOI")
eodata.apply_SCL(store = products_path, band = "NDWI", resolution="highest", subregion="AOI")
eodata.apply_SCL(store = products_path, band = "NDBI", resolution="highest", subregion="AOI")

# Get the paths of all available data for making the cube
#bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI"]
bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDBI", "NDWI"]
paths = generate_cube_paths(eodata, bands, mask = "AOI")


# Create the cube
cube_name = "cube"
cube_path = make_cube(paths, products_path, cube_name, gap_fill = True, dtype = np.float32, force_new = True)

# Prepare training ground truth data
shapefile = "D:\Work\Development_Projects\cropmaps\France\GT\GT_FR_2018_Final.shp" # Shapefile of the vector training data
classes = "EC_hcat_n" # Name of the column with the crops
base = eodata.data[0].NDVI["10"]["AOI"] # Path of a base image to burn
metadata = rasterio.open(base).meta.copy()
gt_data_raster = burn(shapefile, classes, metadata)

model = random_forest_train(cube_path = cube_path,
    gt_fpath = gt_data_raster, 
    test_size = 0.2,
    gridsearch = False,
    parameters = {'max_depth': 50, 'n_estimators': 200},
    results_to = "D:\Work\Development_Projects\cropmaps\France\Results")

saved_model = save_model(model, spath = "D:\Work\Development_Projects\cropmaps\France\Results") # Save model to disk

#model = load_model("D:\Work\Development_Projects\cropmaps\France\Results\model.save")

predicted = random_forest_predict_patches(cube_path = cube_path, model = model, results_to = "D:\Work\Development_Projects\cropmaps\France\Results", patch_size=(512, 512))
landcover_data = "D:\Work\Development_Projects\cropmaps\France\Results\LandCover"
landcover_image = eodata.LandCover(store = landcover_data)

predicted = LandCover_Masking(landcover_image, predicted, results_to = "D:\Work\Development_Projects\cropmaps\France\Results")
