import os
import rasterio
from rasterio import features
import geopandas as gpd

from cropmaps import logger
logging = logger.setup(name = __name__)

def buffer():
    pass

def burn(shapefile:str, classes:str, metadata:dict, classes_id:str = None, save_nomenclature:bool = True, save_to:str = None, outfname:str = "gt.tif")->str:
    """Burns vector ground truth data to raster.

    Args:
        shapefile (str): Path to shapefile ground truth data
        classes (str): Column name with the categories
        metadata (dict): A dictionary with width, height, crs, transform (Can retrieved from a base image with rasterio.open().meta)
        classes_id (str, optional): Column name with ID for each class. If None then this function creates an ID for each unique category. Defaults to None
        save_nomenclature (bool, optional): Writes the nomenclature data. Defaults to True
        save_to (str, optional): Save path of the results. If None then the data will be written at the same path as the shapefile. Defaults to None
        outfname (str, optional): Name of the ground truth image. Defaults to "gt.tif"

    Returns:
        str: Path to ground truth image data
    """
    gt = gpd.read_file(shapefile)

    if classes_id is None:
        classes_id = "class_id"
    
        gt[classes_id] = gt.sort_values(by=classes).groupby(classes).ngroup()
        gt[classes_id] = gt[classes_id] + 1
    
    nomenclature = gt[[classes, classes_id]].drop_duplicates().sort_values(by=classes)

    if save_nomenclature:
        if save_to is None:
            path = os.path.dirname(shapefile)
            nomenclature.to_csv(os.path.join(path, "nomenclature.csv"), index = False)
        else:
            nomenclature.to_csv(os.path.join(save_to, "nomenclature.csv"), index = False)

    gt = gt.to_crs(metadata["crs"]) # Change CRS of vector data to image CRS
    shapes = [[row.geometry, row[classes_id]] for _, row in gt.iterrows()]

    # Burn geometries to raster
    id_gt = features.rasterize(shapes,
            out_shape = (metadata['height'], metadata['width']),
            all_touched = False, # not so fat edges
            transform = metadata['transform'])

    # Save as image
    metadata.update(dtype = id_gt.dtype, nodata = None, count = 1, driver = "GTiff")

    if save_to is None:
        path = os.path.dirname(shapefile)
    else:
        path = save_to

    id_gt_outname = os.path.join(path, outfname) 

    with rasterio.open(id_gt_outname, "w", **metadata) as dest:
        dest.write(id_gt, 1)
    
    return id_gt_outname