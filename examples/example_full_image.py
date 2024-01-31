import os
import numpy as np
import rasterio
from cropmaps.get_creodias import get_data, eodata_path_creator
from cropmaps.sts import sentimeseries
from cropmaps.cube import generate_cube_paths, make_cube
from cropmaps.models import random_forest_train, random_forest_predict, save_model, LandCover_Masking
from cropmaps.prepare_vector import burn

def local_DIAS_path_creator(image):
    if (image.satellite == "Sentinel-2A") or (image.satellite == "Sentinel-2B"):
        satellite = "Sentinel-2"
    else:
        raise ValueError("Satellite name unkwown!")

    if image.processing_level == "Level-2A":
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

search = "D:\Work\Development_Projects\cropmaps\Raw"
store = "D:\Work\Development_Projects\cropmaps\eodata"
AOI = "D:\Work\Development_Projects\cropmaps\AOI\AOI.geojson"

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
products_path = "D:\Work\Development_Projects\cropmaps\eodata_local"
if not os.path.exists(products_path):
    os.makedirs(products_path)

# Create a timeseries with the data from query
eodata = sentimeseries("S2-timeseries")
eodata.find_DIAS(creodias_paths)
eodata.remove_orbit("22")
eodata.sort_images(date=True)

eodata.remove_cloudy(max_cloud = 5)

dates_to_remove = ["12092022", "27102022", "16112022", "21112022", "26122022", "19022023", "04072023", "09072023", "14072023", "19072023", "24072023", "03082023", "08082023", "13082023", "18082023"] # DDMMYYYY

for date in dates_to_remove:
    eodata.remove_date(date=date)

eodata.upsample()
eodata.getVI("NDVI", store = products_path)
eodata.getVI("NDWI", store = products_path)
eodata.getVI("NDBI", store = products_path)
eodata.upsample(band = "NDBI")


print(eodata.show_metadata())

# Apply SCL cloud masks
eodata.apply_SCL(store = products_path, resolution="highest") # This performs only the default bands
eodata.apply_SCL(store = products_path, band = "NDVI", resolution="highest")
eodata.apply_SCL(store = products_path, band = "NDWI", resolution="highest")
eodata.apply_SCL(store = products_path, band = "NDBI", resolution="highest")

# Get the paths of all available data for making the cube
bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI"]
paths = generate_cube_paths(eodata, bands)
print(paths)
# Create the cube
cube_name = "cube"
cube_path = make_cube(paths, products_path, cube_name, gap_fill = True, dtype = np.float32)

# Prepare training ground truth data
shapefile = "D:\Work\Development_Projects\cropmaps\GT\GT_data_22-23_test.shp" # Shapefile of the vector training data
classes = "Crop" # Name of the column with the crops
base = eodata.data[0].NDVI["10"]["raw"] # Path of a base image to burn
metadata = rasterio.open(base).meta.copy()
gt_data_raster = burn(shapefile, classes, metadata)

model = random_forest_train(cube_path = cube_path,
    gt_fpath = gt_data_raster, 
    results_to = "D:\Work\Development_Projects\cropmaps\Results")

saved_model = save_model(model, spath = "D:\Work\Development_Projects\cropmaps\Results") # Save model to disk

predicted = random_forest_predict(cube_path = cube_path, model = model, results_to = "D:\Work\Development_Projects\cropmaps\Results")
predicted = "D:\Work\Development_Projects\cropmaps\Results\Predictions.tif"

landcover_data = "D:\Work\Development_Projects\cropmaps\Results\LandCover"
landcover_image = eodata.LandCover(store = landcover_data)

predicted = LandCover_Masking(landcover_image, predicted, results_to = "D:\Work\Development_Projects\cropmaps\Results")
