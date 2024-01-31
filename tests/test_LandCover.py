import os
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
import rasterio

for directory, _, _ in os.walk(os.path.join(os.path.dirname(__file__), "data")):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))

if not os.path.exists(os.path.join(os.path.dirname(__file__), "data/eodata_local")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "data/eodata_local"))

landcover_data = os.path.join(os.path.dirname(__file__), "data/LandCover")

search_params = [(landcover_data, None, "LC_mosaic.tif", do_not_raise()),
                 (None, None, "LC_mosaic.tif", pytest.raises(TypeError)),
                 (landcover_data, os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"), "LC_mosaic.tif", do_not_raise()),
                ]

@pytest.mark.parametrize("store, aoi, outname, exception", search_params)
def test_LandCover(store, aoi, outname, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(os.path.join(os.path.dirname(__file__), "data/eodata"))
        landcover_image = eodata.LandCover(store, aoi, outname)
        src = rasterio.open(landcover_image)
        assert src.meta["dtype"] == "uint8"
        assert src.meta["driver"] == "GTiff"
        
