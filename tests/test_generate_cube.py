import os
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.cube import generate_cube_paths

for directory, _, _ in os.walk("./data"):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))

if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")

search_params = [("AOI", do_not_raise()),
                 (None, do_not_raise()),
                ]

@pytest.mark.parametrize("mask, exception", search_params)
def test_generate_cube_paths(mask, exception):
    with exception:
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDBI", "NDWI"]

        eodata = sentimeseries("S2-timeseries")
        eodata.find("./data/eodata")
        # To AOI
        eodata.clipbyMask("./data/AOI/AOI.geojson", store = "./data/eodata_local")
        eodata.clipbyMask("./data/AOI/AOI.geojson", store = "./data/eodata_local", resize = True)
        eodata.getVI("NDVI", store = "./data/eodata_local", subregion = "AOI") # Subregion is the name of the mask shapefile
        eodata.getVI("NDWI", store = "./data/eodata_local", subregion = "AOI")
        eodata.getVI("NDBI", store = "./data/eodata_local", subregion = "AOI") # This is in 20m
        eodata.upsample(store = "./data/eodata_local", band = "NDBI", subregion = "AOI", new = "AOI_10_Upsampled") # Transforming to 10m
        eodata.apply_SCL(store = "./data/eodata_local", resolution="highest", subregion="AOI") # This performs only the default bands
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDVI", resolution="highest", subregion="AOI")
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDWI", resolution="highest", subregion="AOI")
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDBI", resolution="highest", subregion="AOI")

        # To raw
        eodata.upsample(store = "./data/eodata_local") # Transforming all 20m bands to 10m
        eodata.getVI("NDVI", store = "./data/eodata_local") # Subregion is the name of the mask shapefile
        eodata.getVI("NDWI", store = "./data/eodata_local")
        eodata.getVI("NDBI", store = "./data/eodata_local") # This is in 20m
        eodata.upsample(store = "./data/eodata_local", band = "NDBI") # Transforming to 10m
        eodata.apply_SCL(store = "./data/eodata_local", resolution="highest") # This performs only the default bands
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDVI", resolution="highest")
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDWI", resolution="highest")
        eodata.apply_SCL(store = "./data/eodata_local", band = "NDBI", resolution="highest")
        paths = generate_cube_paths(eodata, bands, mask)
        assert isinstance(paths, list)
        assert len(paths) == len(bands)*eodata.total