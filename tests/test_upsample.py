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

search_params = [(True, "NDBI", None, None, os.path.join(os.path.dirname(__file__), "data/eodata_local"), "AOI", do_not_raise()),
                 (True, "NDVI", None, None, os.path.join(os.path.dirname(__file__), "data/eodata_local"), "AOI", do_not_raise()),
                 (True, "SCL", None, None, None, None, do_not_raise()),
                 (False, "B07", None, None, os.path.join(os.path.dirname(__file__), "data/eodata_local"), "AOI", do_not_raise()),
                 (False, None, None, None, os.path.join(os.path.dirname(__file__), "data/eodata_local"), "AOI", do_not_raise()),
                 (False, None, None, None, None, None, do_not_raise()),
                ]

@pytest.mark.parametrize("series, band, method, new, store, subregion, exception", search_params)
def test_upsample(series, band, method, new, store, subregion, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(os.path.join(os.path.dirname(__file__), "data/eodata"))
        eodata.clipbyMask(os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"), store = os.path.join(os.path.dirname(__file__), "data/eodata_local"))
        eodata.clipbyMask(os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"))
        eodata.getVI("NDVI", store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), subregion = "AOI") # Subregion is the name of the mask shapefile
        eodata.getVI("NDBI", store = os.path.join(os.path.dirname(__file__), "data/eodata_local"), subregion = "AOI") # This is in 20m
        if series:
            eodata.upsample(None, band, method, new, store, subregion)
        else:
            eodata.upsample(eodata.data[0], band, method, new, store, subregion)

        if subregion is None:
            level = "raw"
        else:
            level = subregion
        
        if series:
            for im in eodata.data:
                # Check if path exists
                assert getattr(im, band)["10"][level]
                # Open image to check
                src = rasterio.open(getattr(im, band)["10"][level])
                # Check datatype
                if band in ["NDVI", "NDBI", "NDWI"]:
                    dtype = ["float32"]
                else:
                    dtype = ["uint8", "uint16"]

                assert src.meta["dtype"] in dtype
                
                output_format = ["GTiff", "JP2OpenJPEG"]
                assert src.meta["driver"] in output_format
                src.close()