import os
import pytest
import numpy as np
import rasterio
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.cube import generate_cube_paths, make_cube

for directory, _, _ in os.walk(os.path.join(os.path.dirname(__file__), "data")):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))

if not os.path.exists(os.path.join(os.path.dirname(__file__), "data/eodata_local")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "data/eodata_local"))

search_params = [(os.path.join(os.path.dirname(__file__), "data/eodata_local/hypercube"), "cube.tif", np.float32, -9999, True, True, 0.0001, 0., True, False, do_not_raise()),
                ]

@pytest.mark.parametrize("searchPath, newFilename, dtype, nodata, gap_fill, harmonize, alpha, beta, force_new, compress, exception", search_params)
def test_make_cube(searchPath, newFilename, dtype, nodata, gap_fill, harmonize, alpha, beta, force_new, compress, exception):
    if not os.path.exists(searchPath):
        os.makedirs(searchPath)

    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(os.path.join(os.path.dirname(__file__), "data/eodata"))
        # To AOI
        eodata.clipbyMask(os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"), store = os.path.join(os.path.dirname(__file__), "data/eodata_local"))
        eodata.clipbyMask(os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"), store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), resize = True)
        eodata.getVI("NDVI", store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), subregion = "AOI") # Subregion is the name of the mask shapefile
        eodata.getVI("NDWI", store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), subregion = "AOI")
        eodata.getVI("NDBI", store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), subregion = "AOI") # This is in 20m
        eodata.upsample(store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), band = "NDBI", subregion = "AOI", new = "AOI_10_Upsampled") # Transforming to 10m
        eodata.apply_SCL(store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), resolution="highest", subregion="AOI") # This performs only the default bands
        eodata.apply_SCL(store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), band = "NDVI", resolution="highest", subregion="AOI")
        eodata.apply_SCL(store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), band = "NDWI", resolution="highest", subregion="AOI")
        eodata.apply_SCL(store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), band = "NDBI", resolution="highest", subregion="AOI")
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDBI", "NDWI"]

        paths = generate_cube_paths(eodata, bands, mask = "AOI")

        cube = make_cube(paths, searchPath, newFilename, dtype, nodata, gap_fill, harmonize, alpha, beta, force_new, compress)

        with rasterio.open(cube) as src:
            assert src.meta["count"] == len(paths)
            assert src.meta["dtype"] == "float32"
