import os
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.exceptions import BandNotFound
import rasterio

for directory, _, _ in os.walk("./data"):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))


if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")

search_params = [(None, "NDWI", "./data/eodata_local", "AOI", None, do_not_raise()),
                 (None, "NDBI", "./data/eodata_local", "AOI", None, do_not_raise()),
                 (None, "NDVI", "./data/eodata_local", "AOI", None, do_not_raise()),
                 (None, None, "./data/eodata_local", "AOI", None, do_not_raise()),
                 (None, None, None, None, None, do_not_raise()),
                 ("./data/eodata/Sentinel-2/L2A/2018/06/27/S2A_MSIL2A_20180627T104021_N0208_R008_T31TEJ_20180627T143337.SAFE/GRANULE/L2A_T31TEJ_A015735_20180627T104837/IMG_DATA/T31TEJ_20180627T104021_B04_10m.jp2", None, None, None, None, pytest.raises(TypeError)),
                 (None, "NDBI", "./data/eodata_local", "AOI", "highest", pytest.raises(BandNotFound)),
                ]

@pytest.mark.parametrize("image, band, store, subregion, resolution, exception", search_params)
def test_apply_SCL(image, band, store, subregion, resolution, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find("./data/eodata")
        eodata.clipbyMask("./data/AOI/AOI.geojson", store = "./data/eodata_local")
        eodata.getVI("NDVI", store = "./data/eodata_local", subregion = "AOI") # Subregion is the name of the mask shapefile
        eodata.getVI("NDWI", store = "./data/eodata_local", subregion = "AOI") # This is in 10m
        eodata.getVI("NDBI", store = "./data/eodata_local", subregion = "AOI") # This is in 20m
        eodata.apply_SCL(image, band, store, subregion, resolution)
        
        if subregion is None:
            level = "raw"
        else:
            level = subregion
        
        if image is None:
            for im in eodata.data:
                if band is None:
                    bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

                    for b in bands:
                        if resolution == "highest":
                            res = "10"
                        else:
                            res = im.setResolution(b)
                        # Check if path exists
                        assert getattr(im, b)[res][level]
                        # Open image to check
                        src = rasterio.open(getattr(im, b)[res][level])
                        # Check datatype
                        if b in ["NDVI", "NDBI", "NDWI"]:
                            dtype = ["float32"]
                        else:
                            dtype = ["uint8", "uint16"]

                        assert src.meta["dtype"] in dtype
                        
                        output_format = ["GTiff", "JP2OpenJPEG"]
                        assert src.meta["driver"] in output_format
                        src.close()
                else:
                    if resolution == "highest":
                        res = "10"
                    else:
                        res = im.setResolution(band)
                    # Check if path exists
                    assert getattr(im, band)[res][level]
                    # Open image to check
                    src = rasterio.open(getattr(im, band)[res][level])
                    # Check datatype
                    if band in ["NDVI", "NDBI", "NDWI"]:
                        dtype = ["float32"]
                    else:
                        dtype = ["uint8", "uint16"]

                    assert src.meta["dtype"] in dtype
                    
                    output_format = ["GTiff", "JP2OpenJPEG"]
                    assert src.meta["driver"] in output_format
                    src.close()
