import os
import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.exceptions import NoDataError

if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")
    
# Test cropmaps.sts.sentimeseries.find()

search_params = [("./data/eodata", "L2A", do_not_raise()),
                ("./data/eodata", "L1C", pytest.raises(NoDataError))
                ]

@pytest.mark.parametrize("search, level, exception", search_params)
def test_find(search, level, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search, level = level)
        assert isinstance(eodata, sentimeseries)
        assert eodata.total == 10.

# Test cropmaps.sts.sentimeseries.find_DIAS()

def local_DIAS_path_creator(image):
    if (image.satellite == "Sentinel-2A") or (image.satellite == "Sentinel-2B"):
        satellite = "Sentinel-2"
    else:
        raise ValueError("Satellite name unkwown!")

    if image.processing_level == "Level-2A" or image.processing_level == "Level-2Ap":
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

# Get data
eodata = sentimeseries("S2-timeseries")
eodata.find("./data/eodata")
eodata.sort_images(date=True)

creodias_paths = []
for image in eodata.data:
    src = os.path.join(image.path, image.name)
    DIAS_path = local_DIAS_path_creator(image)
    creodias_paths.append(os.path.join("./data/eodata", DIAS_path, image.name))

search_params = [(creodias_paths, do_not_raise()),
                ([], pytest.raises(NoDataError))
                ]

@pytest.mark.parametrize("paths, exception", search_params)
def test_find(paths, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find_DIAS(paths)
        assert isinstance(eodata, sentimeseries)
        assert eodata.total == 10.
