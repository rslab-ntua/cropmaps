import os
import rasterio
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries

for directory, _, _ in os.walk("./data"):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))


if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")

eodata = sentimeseries("S2-timeseries")
eodata.find("./data/eodata")
image = eodata.data[0]

search_params = [("./data/AOI/AOI.geojson", False, "B08", False, None, None, "./data/eodata_local", True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", False, "B08", False, None, None, None, True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", False, "B8A", True, None, None, None, True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", True, "B8A", True, None, None, None, True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", False, None, True, None, None, None, True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", False, "B8A", False, None, None, None, True, do_not_raise()),
                 ("./data/AOI/AOI.geojson", False, None, True, None, None, "./data/eodata_local", True, do_not_raise()),
                ]

@pytest.mark.parametrize("shapefile, series, band, resize, method, new, store, force_update, exception", search_params)
def test_clip_by_mask(shapefile, series, band, resize, method, new, store, force_update, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find("./data/eodata")
        if series:
            eodata.clipbyMask(shapefile, None, band, resize, method, new, store, force_update)
            for im in eodata.data:
                if band is None:
                    bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', "SCL"]
                    for b in bands:
                        if resize:
                            resolution = "10"
                        else:
                            resolution = im.setResolution(b)
                        
                        assert getattr(im, b)[resolution][os.path.splitext(os.path.basename(shapefile))[0]]
                        # Check the correct datatype
                        dtype = ["uint8", "uint16"]
                        src = rasterio.open(getattr(im, b)[resolution][os.path.splitext(os.path.basename(shapefile))[0]])
                        assert src.meta["dtype"] in dtype
                        
                        # Check the correct datatype                
                        output_format = "GTiff"
                        assert output_format == src.meta["driver"]
                        src.close()
                else:
                    if resize:
                        resolution = "10"
                    else:
                        resolution = im.setResolution(band)
                
                    # Check if attribute exists
                    assert getattr(im, band)[resolution][os.path.splitext(os.path.basename(shapefile))[0]]
                    
                    # Check the correct datatype
                    dtype = ["uint8", "uint16"]
                    src = rasterio.open(getattr(im, band)[resolution][os.path.splitext(os.path.basename(shapefile))[0]])
                    assert src.meta["dtype"] in dtype
                    
                    # Check the correct driver                
                    output_format = "GTiff"
                    assert output_format == src.meta["driver"]
                    src.close()
        else:
            eodata.clipbyMask(shapefile, eodata.data[0], band, resize, method, new, store, force_update)
            image = eodata.data[0]
            if band is None:
                bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', "SCL"]
                for b in bands:
                    if resize:
                        resolution = "10"
                    else:
                        resolution = image.setResolution(b)
                    
                    assert getattr(image, b)[resolution][os.path.splitext(os.path.basename(shapefile))[0]]
                    # Check the correct datatype
                    dtype = ["uint8", "uint16"]
                    src = rasterio.open(getattr(image, b)[resolution][os.path.splitext(os.path.basename(shapefile))[0]])
                    assert src.meta["dtype"] in dtype
                    
                    # Check the correct driver                
                    output_format = "GTiff"
                    assert output_format == src.meta["driver"]
                    src.close()

            else:
                if resize:
                    resolution = "10"
                else:
                    resolution = image.setResolution(band)
                
                # Check if attribute exists
                assert getattr(image, band)[resolution][os.path.splitext(os.path.basename(shapefile))[0]]
                
                # Check the correct datatype
                dtype = ["uint8", "uint16"]
                src = rasterio.open(getattr(image, band)[resolution][os.path.splitext(os.path.basename(shapefile))[0]])
                assert src.meta["dtype"] in dtype
                
                # Check the correct driver                
                output_format = "GTiff"
                assert output_format == src.meta["driver"]
                src.close()
