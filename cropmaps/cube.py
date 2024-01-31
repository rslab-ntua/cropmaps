import os
from typing import List, Dict, Tuple
from cropmaps.sts import sentimeseries
import numpy as np
import rasterio
from scipy.interpolate import interp1d
import datetime

from cropmaps import logger
logging = logger.setup(name = __name__)

convert = lambda x: datetime.datetime.strptime(x, '%Y%m%dT%H%M%S')

def generate_cube_paths(eodata:sentimeseries, bands:list, mask:str = None)->list:
    """Get all the paths from all the available images in a timeseries.

    Args:
        eodata (sentimeseries): Sentinel 2 data timeseries
        bands (list): List of the bands to extract data
        mask (str, optional): If provided then searches for attribute with this specific name. Defaults to None

    Returns:
        list: List of paths
    """
    paths = []
    if mask is None:
        for image in eodata.data:
            for band in bands:
                paths.append(getattr(image, band)["10"]["raw"])
    else:
        for image in eodata.data:
            for band in bands:
                    paths.append(getattr(image, band)["10"][mask])             
        
    return paths

def make_cube(listOfPaths:List[str], searchPath:str, newFilename:str, dtype:np.dtype, nodata:float = -9999, gap_fill:bool = True, harmonize:bool = True, alpha:float = 0.0001, beta:float = 0., force_new:bool = False, compress = False)->Tuple[List[str], Dict]:
    """Stack satellite images (FROM DIFFERENT FILES) as timeseries cube, without loading them in memory.
    If there is a datetime field in filename, could enable sort=True, to sort cube layers by date, ascending.
    Also, if sort=True, dates are written at .txt file which will be saved with the same output name, as cube.

    Args:
        listOfPaths (List[str]): Paths of images which will participate in cube.
        searchPath (str): Where the result will be saved. Fullpath, ending to dir.
        newFilename (str): Not a full path. Only the filename, without format ending.
        dtype (np.dtype): Destination datatype.
        nodata (float, optional): No data value. Defaults to -9999
        gap_fill(bool, optional): Fill original nodata values where cloud exists and in feasible. Defaults to True
        harmonize(bool, optional): Apply DN to reflectance convertion. User need to provide alpha and beta from alpha * x + beta. Defaults to True
        alpha(float, optional): Alpha component of alpha in alpha * x + beta. Defaults to 0.0001
        beta(float, optional): Beta component of beta in alpha * x + beta. Defaults to 0
        force_new(bool, optional): Update cube with force if exists. Defaults to False
        
    Returns:
        Tuple[List[str], Dict]: Bands description in stacked order, metadata of written cube.
    """
    logging.info("Making hypercube...")

    # Open a random image from images to keep metadata.
    with rasterio.open(listOfPaths[0], 'r') as src:
        # Image metadata.
        metadata = src.meta

    # Keep bands_desc in list & write to file.
    bands_desc = [os.path.basename(os.path.splitext(p)[0]) for p in listOfPaths]

    # Export bands_desc to file.
    with open(os.path.join(searchPath, str(newFilename) + '.txt') , 'w') as myfile:
        for item in bands_desc:
            myfile.write(f"{item}\n")

    # Update third dimension in metadata, as expected for cube.
    metadata.update({
        'dtype': dtype,
        'count': len(listOfPaths),
        'driver':'GTiff',
        'nodata': nodata})
    
    if compress:
        metadata.update({"compress": "lzw"})
    
    # New filename.
    cubeName = os.path.join(searchPath, str(newFilename) + '.tif')
    # Stack products as timeseries cube.
    if not force_new:
        if os.path.exists(cubeName) == True and os.stat(cubeName).st_size != 0:
            logging.info("Done.")
            return cubeName

    with rasterio.open(cubeName, 'w+', **metadata) as dst:
        for id, layer in enumerate(listOfPaths, start=1):
            if gap_fill:
                band = os.path.split(layer)[-1].split(".")[0].split("_")[2]
                previous_layer, next_layer = _find_previous_and_next_strings(listOfPaths, layer, band)
                if (previous_layer is None) or (next_layer is None):
                    with rasterio.open(layer, 'r+') as src:
                        data = src.read(1).astype(dtype)
                        if src.meta["dtype"] == "uint16" or src.meta["dtype"] == "uint8":
                            data[data == 0] = nodata
                            if harmonize:
                                data[data != nodata] = data[data != nodata] * alpha + beta
                        dst.write_band(id, data)
                        band_name = os.path.split(src.name)[-1].split('.')[0]
                        dst.set_band_description(id, band_name)
                else:
                    c = rasterio.open(layer, 'r+')
                    c_array = c.read(1).astype(dtype)
                    if c.meta["dtype"] == "uint16" or c.meta["dtype"] == "uint8":
                        c_array[c_array == 0] = nodata
                        if harmonize:
                            c_array[c_array != nodata] = c_array[c_array != nodata]  * alpha + beta
                    
                    time_c = os.path.split(layer)[-1].split('.')[0].split("_")[1]
                    time_c = convert(time_c)
                    
                    p = rasterio.open(previous_layer, 'r+')
                    p_array = p.read(1).astype(dtype)
                    if p.meta["dtype"] == "uint16" or p.meta["dtype"] == "uint8":
                        p_array[p_array == 0] = nodata
                        if harmonize:
                            p_array[p_array != nodata] = p_array[p_array != nodata]  * alpha + beta

                    time_p = os.path.split(previous_layer)[-1].split('.')[0].split("_")[1]
                    time_p = convert(time_p)
                    
                    n = rasterio.open(next_layer, 'r+')
                    n_array = n.read(1).astype(dtype)
                    if n.meta["dtype"] == "uint16" or n.meta["dtype"] == "uint8":
                        n_array[n_array == 0] = nodata
                        if harmonize:
                            n_array[n_array != nodata] = n_array[n_array != nodata]  * alpha + beta

                    time_n = os.path.split(next_layer)[-1].split('.')[0].split("_")[1]
                    time_n = convert(time_n)
                    
                    c_array = _gap_fill(c_array, p_array, n_array, time_c, time_p, time_n, nodata = nodata)

                    dst.write_band(id, c_array)
                    band_name = os.path.split(layer)[-1].split('.')[0]
                    dst.set_band_description(id, band_name)
            else:
                with rasterio.open(layer, 'r+') as src:
                    data = src.read(1).astype(dtype)
                    if src.meta["dtype"] == "uint16" or src.meta["dtype"] == "uint8":
                        data[data == 0] = nodata
                        if harmonize:
                            data[data != nodata] = data[data != nodata]  * alpha + beta
                    
                    dst.write_band(id, data)
                    band_name = os.path.split(src.name)[-1].split('.')[0]
                    dst.set_band_description(id, band_name)
    
    logging.info("Done.")

    return cubeName

def _gap_fill(current: np.ndarray, p: np.ndarray, n: np.ndarray, 
                time_current: datetime.datetime, time_p: datetime.datetime, time_n: datetime.datetime, nodata = -9999) -> np.ndarray:
    """
    Interpolate values of the `current` array where the values are equal to -9999
    and the values of `p` and `n` are different than -9999 using linear interpolation 
    based on the provided time information. If either `p` or `n` is -9999, no interpolation is performed.

    Args:
    current (np.ndarray): The array of values that need to be interpolated.
    p (np.ndarray): The array of previous values.
    n (np.ndarray): The array of next values.
    time_current (datetime.datetime): The datetime value corresponding to `current`.
    time_p (datetime.datetime): The datetime value corresponding to `p`.
    time_n (datetime.datetime): The datetime value corresponding to `n`.

    Returns:
    - np.ndarray: The interpolated array.

    Note:
    It is assumed that `current`, `p`, and `n` have the same shape.
    """
    
    mask = (current == nodata) & (p != nodata) & (n != nodata)
    
    # Convert datetime to timestamp for interpolation
    time_current_ts = time_current.timestamp()
    time_p_ts = time_p.timestamp()
    time_n_ts = time_n.timestamp()
    
    # Calculate interpolation weights
    weight_p = (time_n_ts - time_current_ts) / (time_n_ts - time_p_ts)
    weight_n = 1 - weight_p
    
    # Interpolate where necessary
    current[mask] = weight_p * p[mask] + weight_n * n[mask]
    
    return current

def _find_previous_and_next_strings(input_list: list, target_element: str, target_letters: str) -> tuple:
    """
    Finds the previous and next strings in a list that contain specific letters, based on a specific element.

    Args:
    input_list (list of str): A list of strings to search through.
    target_element (str): The specific element in the list to use as a reference.
    target_letters (str): The specific letters to search for in the strings.

    Returns:
    tuple of str: A tuple containing two elements - the previous string (or None if there is no previous string)
    and the next string (or None if there is no next string) containing the target letters.
    """

    try:
        # Find the index of the target element in the list
        index = input_list.index(target_element)

        # Find the previous string containing the target letters, if it exists
        previous_string = next((x for x in reversed(input_list[:index]) if target_letters in x), None)

        # Find the next string containing the target letters, if it exists
        next_string = next((x for x in input_list[index + 1:] if target_letters in x), None)
    except ValueError:
        # If the target element is not found in the list, both previous and next strings are None
        previous_string = None
        next_string = None

    return previous_string, next_string
