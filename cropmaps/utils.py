import os
import requests
import shapely
import geopandas as gpd

from tqdm.auto import tqdm

from cropmaps import logger
logging = logger.setup(name = __name__)

def worldcover(geometry:shapely.geometry.polygon.Polygon, savepath:str)->gpd.GeoDataFrame:
    """Downloads landcover maps from worldcover project
    Args:
        geometry (shapely.geometry.polygon.Polygon): Path to AOI file to download data or a list with geometries
        savepath (str): Path to store data
    Returns:
        gpd.GeoDataFrame: Downloaded tiles 
    """
    logging.info("Getting ESA WorldCover information...")
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    url = f'{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson'
    grid = gpd.read_file(url)

    # get grid tiles intersecting AOI
    tiles = grid[grid.intersects(geometry)]
    
    # works only if AOI covers one tile
    for tile in tqdm(tiles.ll_tile):
        url = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
        r = requests.get(url, allow_redirects = True)
        out_fn = f"ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
        with open(os.path.join(savepath, out_fn), 'wb') as f:
            f.write(r.content)    
    
    logging.info("Done.")

    return tiles