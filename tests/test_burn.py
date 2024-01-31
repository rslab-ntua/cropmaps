import os
import pytest
import numpy as np
import rasterio
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.prepare_vector import burn

for directory, _, _ in os.walk(os.path.join(os.path.dirname(__file__), "data")):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            os.remove(os.path.join(directory, file))
        if file.endswith(".tif.aux.xml"):
            os.remove(os.path.join(directory, file))

if not os.path.exists(os.path.join(os.path.dirname(__file__), "data/eodata_local")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "data/eodata_local"))

search_params = [(os.path.join(os.path.dirname(__file__), "data/reference_data/france_data_2018.shp"), "EC_hcat_n", None, True, None, "gt.tif", do_not_raise()),
                ]

@pytest.mark.parametrize("shapefile, classes, classes_id, save_nomenclature, save_to, outfname, exception", search_params)
def test_burn(shapefile, classes, classes_id, save_nomenclature, save_to, outfname, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(os.path.join(os.path.dirname(__file__), "data/eodata"))
        # To AOI
        eodata.clipbyMask(os.path.join(os.path.dirname(__file__), "data/AOI/AOI.geojson"), store = os.path.join(os.path.dirname(__file__), "data/eodata_local"))
        base = eodata.data[0].B04["10"]["AOI"]
        metadata = rasterio.open(base).meta.copy()

        gt_data_raster = burn(shapefile, classes, metadata, classes_id, save_nomenclature, save_to, outfname)

        with rasterio.open(gt_data_raster) as src:
            assert src.meta == metadata