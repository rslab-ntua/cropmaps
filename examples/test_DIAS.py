import os
import numpy as np
import rasterio
from cropmaps.sts import sentimeseries
from cropmaps.get_creodias import get_data, check_L2, eodata_path_creator
from cropmaps.cube import generate_cube_paths, make_cube
from cropmaps.models import random_forest_train, random_forest_predict, save_model, LandCover_Masking
from cropmaps.prepare_vector import burn

# Mask the data and resize them to 10m resolution to match the highest possible resolution
AOI = "/home/eouser/uth/Cap_Bon/AOI/AOI.geojson"

user = "****"
password = "****"
start_date = "20210910"
end_date = "20220910"

# Check here for more: https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/FullTextSearch?redirectedfrom=SciHubUserGuide.3FullTextSearch
data = get_data(AOI, start_date, end_date, user, password, producttype = "S2MSI2A", cloudcoverpercentage = "[0 TO 5]")
data = check_L2(data)
creodias_paths = eodata_path_creator(data)
print(creodias_paths)
# Reproducing a DIAS enviroment where we can not write inside the data folder from the moment we have the data
# This step is required because there are no permissions to write inside the /eodata DIAS folder
products_path = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/eodata_local"
if not os.path.exists(products_path):
    os.makedirs(products_path)

# Create a timeseries with all the available data from query
eodata = sentimeseries("S2-timeseries")
eodata.find_DIAS(creodias_paths)
eodata.sort_images(date=True)

# Keep only images with specific orbit 122
# Over the region of interest there are two available orbits the the one contains missing data
unique_orbits = list(set(eodata.orbits))
print(f"Orbits: {unique_orbits}")
unique_orbits.remove("122")
to_be_removed = unique_orbits
for orbit in to_be_removed:
    eodata.remove_orbit(orbit)

#Remove specific dates with bad data
eodata.remove_date("11112021")

# OK no extra orbits (Can use remove_orbit if an orbit exists)

# Mask the data and resize them to 10m resolution to match the highest possible resolution
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
eodata.clipbyMask(shapefile = AOI, store = products_path) # Raw Clipping
eodata.clipbyMask(shapefile = AOI, store = products_path, resize = True) # Resize 20m to 10m

# If you want to perform the analysis on less bands just add the band name to the band argument
# as in here: eodata.clipbyMask(shapefile = mask, store = products_path, band = "B05", resize = True)

# Calculate vegetation indexes
eodata.getVI("NDVI", store = products_path, subregion = "AOI") # Subregion is the name of the mask shapefile
eodata.getVI("NDWI", store = products_path, subregion = "AOI")
eodata.getVI("NDBI", store = products_path, subregion = "AOI")

# Do the same for the vegetation indexes
# NOTE: CLIPPED DATA BY DEFAULT ARE SAVED AS AN ATTRIBUTE OF THE S2IMAGE OBJECT AS image.BAND[RESOLUTION][MASK_NAME]
# SO IN THIS CASE TO ACCESS THE ATTRIBUTE OF IMAGE 0, BAND B12 YOU CAN write: eodata.data[0].B12["10"]["AOI"] 
eodata.upsample(store = products_path, band = "NDBI", subregion = "AOI")

# Get the paths of all available data for making the cube
bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI"]
paths = generate_cube_paths(eodata, bands, mask = "AOI")

# Create the cube
cube_name = "cube"
cube_path = make_cube(paths, products_path, cube_name, dtype = np.float32)

# Prepare training ground truth data
shapefile = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/Ground_Truth/GT_data_selected_balanced_buffered.shp" # Shapefile of the vector training data
classes = "Crop" # Name of the column with the crops
base_path = eodata.data[0].NDVI["10"]["AOI"] # Path of a base image to burn
metadata = rasterio.open(base_path).meta.copy() # Copy metadata dictionary to save the image using the same CRS, transform, shape etc
gt_data_raster = burn(shapefile, classes, metadata) # Save the ground truth data as raster

# Train the model using the cube and the burned raster image with the ground truth data
model = random_forest_train(cube_path = cube_path,
    gt_fpath = gt_data_raster, 
    results_to = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/")

#saved_model = save_model(model, spath = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/") # Save model to disk

predicted = random_forest_predict(cube_path = cube_path, model = model, results_to = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/") # Use the model to predict over the cube


landcover_data = "/home/eouser/uth/Cap_Bon/LandCover" # Path to store LandCover data
landcover_image = eodata.LandCover(store = landcover_data) # Download ESA worldcover data

predicted = LandCover_Masking(landcover_image, predicted, results_to = "/home/eouser/vol1/crop-type-mapping/Cap_Bon/") # Mask the result using the third party LandCover product
