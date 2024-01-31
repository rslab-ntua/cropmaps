import os
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
import rasterio

for directory, _, _ in os.walk("./data"):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))


if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")

search_params = [("NDVI", "./data/eodata_local", None, "AOI", False, do_not_raise()),
                 ("NDWI", "./data/eodata_local", None, "AOI", False, do_not_raise()),
                 ("NDBI", "./data/eodata_local", None, "AOI", False, do_not_raise()),
                 ("NDVI", "./data/eodata_local", None, None, False, do_not_raise()),
                 ("NDWI", "./data/eodata_local", None, None, False, do_not_raise()),
                 ("NDBI", "./data/eodata_local", None, None, False, do_not_raise()),
                 ("NDVI", None, None, None, False, do_not_raise()),
                 ("NDWI", None, None, None, False, do_not_raise()),
                 ("NDBI", None, None, None, False, do_not_raise()),
                 ("NDVI", None, None, "AOI", False, do_not_raise()),
                 ("NDWI", None, None, "AOI", False, do_not_raise()),
                 ("NDBI", None, None, "AOI", False, do_not_raise()),
                ]

@pytest.mark.parametrize("index, store, image, subregion, verbose, exception", search_params)
def test_getVI(index, store, image, subregion, verbose, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find("./data/eodata")
        eodata.clipbyMask("./data/AOI/AOI.geojson", store = "./data/eodata_local")
        eodata.clipbyMask("./data/AOI/AOI.geojson")
        eodata.getVI(index, store, image, subregion, verbose)
        if subregion is None:
            level = "raw"
        else:
            level = subregion
        
        if image is None:
            for im in eodata.data:
                resolution = im.setResolution(index)
                # Check if path exists
                assert getattr(im, index)[resolution][level]
                # Open image to check
                src = rasterio.open(getattr(im, index)[resolution][level])
                # Check datatype
                dtype = "float32"
                assert src.meta["dtype"] == dtype
                
                output_format = "GTiff"
                assert output_format == src.meta["driver"]
                src.close()
