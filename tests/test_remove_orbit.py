import os
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

search_params = [("8", 0, do_not_raise()),
                 (None, 10, pytest.raises(TypeError)),
                 ("5", 10, do_not_raise()),
                ]

@pytest.mark.parametrize("orbit, total, exception", search_params)
def test_remove_orbit(orbit, total, exception):
    with exception:
        eodata = sentimeseries("S2-timeseries")
        eodata.find("./data/eodata")
        eodata.remove_orbit(orbit)
        assert eodata.total == total
