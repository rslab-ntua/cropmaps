import pytest
from contextlib import suppress as do_not_raise
from cropmaps.sts import sentimeseries
from cropmaps.exceptions import MinMaxCloudBoundError, MinMaxDateError
import os

if not os.path.exists("./data/eodata_local"):
    os.makedirs("./data/eodata_local")

# Test cropmaps.ts.remove_cloudy()

search_params = [("./data/eodata", 5, 0, do_not_raise()),
                 ("./data/eodata", "5", 0, pytest.raises(TypeError)),
                 ("./data/eodata", 5, "0", pytest.raises(TypeError)),
                 ("./data/eodata", 0, 5, pytest.raises(MinMaxCloudBoundError)),
                ]

@pytest.mark.parametrize("search, max_cloud, min_cloud, exception", search_params)
def test_remove_cloudy(search, max_cloud, min_cloud, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search)
        # remove image based on cloud coverage
        eodata.remove_cloudy(max_cloud = max_cloud, min_cloud = min_cloud)
        assert eodata.total == 8
        for image, cc in zip(eodata.data, eodata.cloud_cover):
            assert image.cloud_cover == cc

# Test cropmaps.ts.keep_timerange()

search_params = [("./data/eodata", "104100", "104400", do_not_raise()),
                 ("./data/eodata", 104100, "104400", pytest.raises(TypeError)),
                 ("./data/eodata", "104100", 104400, pytest.raises(TypeError)),
                 ("./data/eodata", "104400", "104100", pytest.raises(ValueError))
                ]

@pytest.mark.parametrize("search, start_time, end_time, exception", search_params)
def test_keep_timerange(search, start_time, end_time, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search)
        eodata.keep_timerange(start_time = start_time, end_time = end_time)
        assert eodata.total == 3
        for image, datetime in zip(eodata.data, eodata.dates):
            assert image.datetime.time() == datetime.time()

# Test cropmaps.ts.remove_date()

search_params = [("./data/eodata", "23012017", do_not_raise()),
                 ("./data/eodata", "23012018", do_not_raise()),
                 ("./data/eodata", "230120188", pytest.raises(ValueError)),
                 ("./data/eodata", ["23012018", "28012018"], do_not_raise()),
                 ("./data/eodata", ["23012018", "28012018","28012017"], do_not_raise()),
                 ("./data/eodata", ["23012018", "28012018","280120177"], pytest.raises(ValueError)),
                 ("./data/eodata", 23012018, pytest.raises(TypeError)),
                 ("./data/eodata", [23012018], pytest.raises(TypeError))
                ]

@pytest.mark.parametrize("search, date, exception", search_params)
def test_keep_timerange(search, date, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search)
        eodata.remove_date(date)
        assert (eodata.total == 10) or (eodata.total == 9) or (eodata.total == 8)
        for image, datetime in zip(eodata.data, eodata.dates):
            assert image.datetime == datetime

# Test cropmaps.ts.filter_dates()
            
search_params = [("./data/eodata", 23012017, None, pytest.raises(TypeError)),
                 ("./data/eodata", "231020181", None, pytest.raises(ValueError)),
                 ("./data/eodata", "01102018", None, do_not_raise()),
                 ("./data/eodata", "01102018", 23012017, pytest.raises(TypeError)),
                 ("./data/eodata", "01102018", "230120171", pytest.raises(ValueError)),
                 ("./data/eodata", "01102018", "01042018", do_not_raise()),
                ]

@pytest.mark.parametrize("search, max_date, min_date, exception", search_params)
def test_filter_dates(search, max_date, min_date, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search)
        eodata.filter_dates(max_date = max_date, min_date = min_date)
        assert (eodata.total) == 9 or (eodata.total == 6)
        for image, datetime in zip(eodata.data, eodata.dates):
            assert image.datetime == datetime

# Test cropmaps.ts.sort_images()

search_params = [("./data/eodata", "True", True, pytest.raises(TypeError)),
                 ("./data/eodata", True, "True", pytest.raises(TypeError)),
                 ("./data/eodata", False, False, do_not_raise()),
                 ("./data/eodata", True, False, do_not_raise()),
                 ("./data/eodata", False, True, do_not_raise()),
                ]

@pytest.mark.parametrize("search, cloud_coverage, date, exception", search_params)
def test_sort_images(search, cloud_coverage, date, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find(search)
        eodata.sort_images(cloud_coverage = cloud_coverage, date = date)
        if cloud_coverage:
            assert all(float(eodata.cloud_cover[i]) <= float(eodata.cloud_cover[i+1]) for i in range(len(eodata.cloud_cover)-1))
            for image, cc in zip(eodata.data, eodata.cloud_cover):
                assert image.cloud_cover == cc
            for image, datetime in zip(eodata.data, eodata.dates):
                assert image.datetime == datetime
            for image, name in zip(eodata.data, eodata.names):
                assert image.name == name
        elif date:
            assert all(eodata.dates[i] <= eodata.dates[i+1] for i in range(len(eodata.dates)-1))
            for image, cc in zip(eodata.data, eodata.cloud_cover):
                assert image.cloud_cover == cc
            for image, datetime in zip(eodata.data, eodata.dates):
                assert image.datetime == datetime
            for image, name in zip(eodata.data, eodata.names):
                assert image.name == name
        elif (date == False) and (cloud_coverage == False):
            assert all(eodata.names[i] <= eodata.names[i+1] for i in range(len(eodata.names)-1))
            for image, cc in zip(eodata.data, eodata.cloud_cover):
                assert image.cloud_cover == cc
            for image, datetime in zip(eodata.data, eodata.dates):
                assert image.datetime == datetime
            for image, name in zip(eodata.data, eodata.names):
                assert image.name == name
