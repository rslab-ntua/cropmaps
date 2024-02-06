import os
import logging
import xml.etree.ElementTree as Etree
import fnmatch
import datetime
import pyproj
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

from cropmaps.vi import vi
from cropmaps.exceptions import BandNotFound, PathError, VegetationIndexNotInList

from cropmaps import logger
logging = logger.setup(name = __name__)

# Define a lambda function to convert dates
convert = lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')

class sentinel2():
    """A Sentinel 2 image."""
    
    def __init__(self, path, name):
        """ A Sentinel 2 image.
        Args:
            path (str, path-like): Path to image
            name (str): Name of the file
        """
        self.path = path
        self.name = name
        self.md_file = None
        self.tile_md_file = None
        self.satellite = None
        self.datetime = None
        self.date = None
        self.time = None
        self.str_datetime = None
        self.gml_coordinates = None
        self.cloud_cover = None
        self.processing_level = None
        self.tile_id = None
        self.crs = None
        self.orbit = None

    def getmetadata(self):
        """Searching for metadata (XML) files.
        """
        for (dirpath, _, filenames) in os.walk(os.path.join(self.path, self.name)):
            for file in filenames:
                if file.startswith("MTD_MSI"):
                    self.md_file = file
                    XML = self._readXML(dirpath, file)
                    self._parseGeneralMetadata(XML)
                elif file.startswith("MTD_TL"):
                    self.tile_md_file = file
                    XML = self._readXML(dirpath, file)
                    self._parseTileMetadata(XML)

    def _readXML(self, path:str, file:str):
        """Reads XML file.

        Args:
            path (str): Path to file
            file (str): Name of the file plus extention

        Returns:
            Etree.Element: XML opened file
        """
        tree = Etree.parse(os.path.join(path, file))
        root = tree.getroot()

        return root

    def _parseGeneralMetadata(self, root):
        """Parsing general S2 metadata from eTree.Element type object.

        Args:
            root (eTree.Element): S2 metadata from eTree.Element type object
        """
        logging.info("Parsing Image Metadata file...")
        self.satellite = root.findall(".//SPACECRAFT_NAME")[0].text
        self.str_datetime = self.name[11:26]
        self.datetime = convert(root.findall(".//DATATAKE_SENSING_START")[0].text)
        self.date = self.datetime.date()
        self.time = self.datetime.time()
        self.gml_coordinates = root.findall(".//EXT_POS_LIST")[0].text
  
        self.cloud_cover = "{:.3f}".format(float(root.findall(".//Cloud_Coverage_Assessment")[0].text))
        self.processing_level = root.findall(".//PROCESSING_LEVEL")[0].text
        if self.processing_level == "Level-2Ap":
            self.processing_level = "Level-2A"
        self.tile_id = self.name[39:44]
        self.orbit = root.findall(".//SENSING_ORBIT_NUMBER")[0].text       
        logging.info("Done!")

    def _parseTileMetadata(self, root):
        """Parsing general S2 tile metadata from eTree.Element type object.

        Args:
            root (eTree.Element): S2 tile metadata from eTree.Element type object
        """

        logging.info("Parsing Tile Metadata file...")
        epsg = root[1][0][1].text
        self.crs = pyproj.crs.CRS(epsg)
        logging.info("Done!")

    @staticmethod
    def setResolution(band):
        """ Getting band resolution for Sentinel 2.
        Args:
            band (str): Band short name as string
        Returns:
            str: Band resolution
        """
        resolutions = {
            "B01": "60",
            "B02": "10",
            "B03": "10",
            "B04": "10",
            "B05": "20",
            "B06": "20",
            "B07": "20",
            "B08": "10",
            "B8A": "20",
            "B09": "60",
            "B10": "60",
            "B11": "20",
            "B12": "20",
            "SCL": "20",
            "NDVI": "10",
            "NDBI": "20",
            "NDWI": "10",
        }
        return resolutions.get(band)

    def getBands(self):
        """Finds all the available bands of an image and sets new attributes for each band.
        """

        bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', "SCL"]

        for band in bands:
            resolution = self.setResolution(band)

            for (dirpath, _, filenames) in os.walk(os.path.join(self.path, self.name)):
                for file in filenames:
                    if self.processing_level == 'Level-2A':
                        if fnmatch.fnmatch(file, "*{}*{}m*.jp2".format(band, resolution)):
                            setattr(self, 'datapath_{}'.format(resolution), os.path.join(dirpath))
                            break
                    else:
                        if fnmatch.fnmatch(file, "*_{}_*.jp2".format(band)):
                            logging.debug(os.path.join(dirpath, file))
                            setattr(self, 'datapath', os.path.join(dirpath))
                            break

            for (dirpath, _, filenames) in os.walk(os.path.join(self.path, self.name)):
                for file in filenames:
                    if self.processing_level == 'Level-2A':
                        if fnmatch.fnmatch(file, "*{}*{}m*.jp2".format(band, resolution)):
                            setattr(self, '{}'.format(band), {resolution: {"raw" : os.path.join(dirpath, file)}})
                    else:    
                        if fnmatch.fnmatch(file, "*_{}_*.jp2".format(band)):
                            setattr(self, '{}'.format(band), {resolution: {"raw" : os.path.join(dirpath, file)}})

    @property
    def show_metadata(self):
        """Prints metadata using __dict__
        """
        print (self.__dict__)
    
    @staticmethod
    def writeResults(path:str, name:str, array:np.array, metadata:dict):
        """Writing a new image with the use of rasterio module.
        Args:
            path (str): Path to image
            name (str): Image name
            array (np.ndarray): Image numpy array
            metadata (dict): Metadata dictionary
        """
        logging.info("Saving {}...".format(name))
        with rasterio.open(os.path.join(path, name), "w", **metadata) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            else:
                dst.write(array)

    def upsample(self, band = None, store = None, new = None, subregion = None, method = None, ext = 'tif'):
        """Upsample 20 meters spatial resolution bands to 10 meters.

        Args:
            store (str, optional): Path to store data. If None then stores the results to default image path. Defaults to None
            band (str, optional): Band to apply upsample. If None then raises error. Defaults to None
            new (str, optional): Name extension. If None then adds new resolution and _Upsampled. Defaults to None
            subregion (str, optional): Perform upsample to subregion. Defaults to None.
            method (Resampling, optional): Resampling method. If None then applies Resampling.nearest. Defaults to None
            ext (str, optional): Image extension. Defaults to 'tif'.
        """
        if subregion is None:
            region = "raw"
        else:
            region = subregion

        if band != None:
            if hasattr(self, band):
                resolution = self.setResolution(band)
                if int(resolution) == 20:
                    if hasattr(self, 'datapath'):
                        if store is None:
                            path = self.datapath
                            res = 10
                        else:
                            path = store
                            res = 10
                    else:
                        if store is None:
                            if hasattr(self, 'datapath_10'):
                                res = 10
                                path = self.datapath_10
                            else:
                                raise PathError("Could not find a path to store the image.")
                        else:
                            res = 10
                            path = store
                else:
                    logging.warning("File {} already in highest resolution...".format(getattr(self, band)[self.setResolution(band)][region]))
                    return
                
                if new is None:
                    new = str(res) + "_Upsampled"

                if subregion is None:    
                    # New name for output image
                    out_tif = os.path.join(path, "T{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, band, new, ext))
                else:
                    out_tif = os.path.join(path, "T{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, band, region, new, ext))

                if os.path.exists(out_tif) == True and os.stat(out_tif).st_size != 0:
                    # Pass if file already exists & it's size is not zero
                    logging.warning("File {} already exists...".format(out_tif))
                    try:
                        getattr(self, band)[str(res)][region] = out_tif
                    except KeyError:
                        getattr(self, band).update({str(res): {region: out_tif}})    
                    return
                
                if method is None:
                    resampling = Resampling.nearest
                else:
                    resampling = method

                # Use as high resolution bands only 4 and 8 that are trustworthy
                hr_bands = ['B04', 'B08']
                hr_band = None
                for hrb in hr_bands:
                    if hasattr(self, hrb):
                        hr_band = getattr(self, hrb)["10"][region]
                        break
                if hr_bands is None:
                    raise BandNotFound("No high resolution band found!")

                fpath = getattr(self, band)[str(resolution)][region]
                self.reproj_match(fpath, hr_band, to_file = True, outfile = out_tif, resampling = resampling)

                try:
                    getattr(self, band)[str(res)][region] = out_tif
                except KeyError:
                    getattr(self, band).update({str(res): {region: out_tif}})
            
            else:
                raise BandNotFound("Object {} has no attribute {} (band).".format(self, band))
        else:
            raise ValueError("Argument 'band' must be provided!")
        
    @staticmethod
    def reproj_match(image:str, base:str, to_file:bool = False, outfile:str = "output.tif", resampling:rasterio.warp.Resampling = Resampling.nearest, compress = False) -> None:
        """Reprojects/Resamples an image to a base image.
        Args:
            image (str): Path to input file to reproject/resample
            base (str): Path to raster with desired shape and projection 
            outfile (str): Path to saving Geotiff
        """
        # open input
        with rasterio.open(image) as src:
            # open input to match
            with rasterio.open(base) as match:
                dst_crs = match.crs
                dst_transform = match.meta["transform"]
                dst_width = match.width
                dst_height = match.height
            # set properties for output
            metadata = src.meta.copy()
            metadata.update({"crs": dst_crs,
                            "transform": dst_transform,
                            "width": dst_width,
                            "height": dst_height,
                            })
            if compress:
                metadata.update({"compress": "lzw"})          
            if to_file:
                with rasterio.open(outfile, "w", **metadata) as dst:
                    # iterate through bands and write using reproject function
                    for i in range(1, src.count + 1):
                        reproject(
                            source = rasterio.band(src, i),
                            destination = rasterio.band(dst, i),
                            src_transform = src.transform,
                            src_crs = src.crs,
                            dst_transform = dst_transform,
                            dst_crs = dst_crs,
                            resampling = resampling)
                return None
            else:
                array, transform = reproject(
                            source = rasterio.band(src, 1),
                            destination = np.ndarray((1, dst_height, dst_width)),
                            src_transform = src.transform,
                            src_crs = src.crs,
                            dst_transform = dst_transform,
                            dst_crs = dst_crs,
                            resampling = resampling)
                
                return(array, transform)
    
    @staticmethod
    def _cloud_mask(mask:rasterio.io.DatasetReader)->np.array:
        """Generate a binary mask with 0 in bad and 1 in good pixels respectively.

        Args:
            mask (rasterio.io.DatasetReader): SCL mask

        Returns:
            np.array: Binary mask
        """
        scl_unreliable = {
                1:'SATURATED_OR_DEFECTIVE',
                2:'DARK_AREA_PIXELS',
                3:'CLOUD_SHADOWS',
                8:'CLOUD_MEDIUM_PROBABILITY',
                9:'CLOUD_HIGH_PROBABILITY',
                10:'THIN_CIRRUS'}
        
        mask_array = mask.read(1)
        # 0 for bad scl classes
        for c in list(scl_unreliable.keys()):
            mask_array[mask_array == c] = 0
        # 1 for not bad scl classes (set 1 as nodata value)
        mask_array[mask_array != 0] = 1

        return mask_array
    
    def apply_cloud_mask(self, band:str = None, store:str = None, subregion:str = None, resolution:str = None, new:str = "CLOUDMASK", compress = False)->None:
        """Apply default SCL mask to S2 images.

        Args:
            band (str, optional): Band to apply SCL mask. If None then applies mask to all default bands. Defaults to None.
            store (str, optional): Path to store the new images. If None then the results are saved in the default S2 path. Defaults to None.
            subregion (str, optional): Apply mask to subregion. If subregion is None then applies the mask to the default raw images. Defaults to None.
            resolution (str, optional): Resolution to apply cloud mask. If None the applies to default resolution else tries to apply to the highest 10m resolution. Defaults to None.
            new (str, optional): New name extension. Defaults to "CLOUDMASK".
        """

        if band is None:
            bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
            logging.info(f"Applying SCL cloud mask to all bands of image {self.name}")
            for b in tqdm(bands):
                
                if resolution == "highest":
                    res = "10"
                else:
                    res = self.setResolution(b)
                
                if store is None:
                    if hasattr(self, 'datapath'):
                        path = self.datapath
                    else:
                        if int(res) == 10:
                            path = getattr(self, 'datapath_10')
                        elif int(res) == 20:
                            path = getattr(self, 'datapath_20')
                        else:
                            raise PathError("Can't find storing path!")
                else:
                    path = store                
                
                if subregion is None:
                    subregion = "raw"

                try:
                    image = getattr(self, b)[res][subregion]
                except KeyError:
                    raise BandNotFound(f"Band {b} in resolution {res} is missing!")
                
                with rasterio.open(image, "r") as src:
                    meta = src.meta.copy()
                    
                    if meta['dtype'] == "uint16" or meta['dtype'] == "uint8":
                        nodata = 0
                    else:
                        nodata = -9999
                        
                    meta.update({'driver': 'Gtiff'})
                    ext = "tif"
                        
                    if subregion == "raw":
                        new_name = "T{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, b, res, new, ext)
                    else:
                        new_name = "T{}_{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, b, res, subregion, new, ext)

                        
                    if os.path.exists(os.path.join(path, new_name)) == True and os.stat(os.path.join(path, new_name)).st_size != 0:
                        # Pass if file already exists & it's size is not zero
                        logging.warning("File {} already exists...".format(os.path.join(path, new_name)))
                        try:
                            getattr(self, b)[str(res)][subregion] = os.path.join(path, new_name)
                            continue
                        except KeyError:
                            getattr(self, b).update({str(res): {subregion: os.path.join(path, new_name)}})    
                            continue

                    if hasattr(self, 'SCL'):
                        try:
                            SCL = self.SCL[res][subregion]
                        except KeyError:
                            if subregion == "raw":
                                self.upsample(store = store, band = "SCL")
                            else:    
                                self.upsample(store = store, band = "SCL", subregion = subregion)
                            SCL = self.SCL[res][subregion]
                    else:
                        raise BandNotFound("SCL band is missing!")
                    SCL_data = rasterio.open(SCL)
                    mask = self._cloud_mask(SCL_data)                 
                    
                    array = src.read(1)
                    array[mask==0] = nodata
                    meta.update({"nodata": nodata})
                    if compress:
                        meta.update({"compress": "lzw"})                    
                    with rasterio.open(os.path.join(path, new_name), "w", **meta) as dest:
                        dest.write(array, 1)
            
                getattr(self, b)[str(res)][subregion] = os.path.join(path, new_name)

        else:
            
            logging.info(f"Applying SCL cloud mask for band {band} of image {self.name}.")

            if resolution == "highest":
                res = "10"
            else:
                res = self.setResolution(band)

            if store is None:
                if hasattr(self, 'datapath'):
                    path = self.datapath
                else:
                    if int(res) == 10:
                        path = getattr(self, 'datapath_10')
                    elif int(res) == 20:
                        path = getattr(self, 'datapath_20')
                    else:
                        raise PathError("Can't find storing path!")
            else:
                path = store                
            
            if subregion is None:
                subregion = "raw"
            
            try:
                image = getattr(self, band)[res][subregion]
            except KeyError:
                raise BandNotFound(f"Band {band} in resolution {res} is missing!")

            with rasterio.open(image, "r") as src:
                meta = src.meta.copy()
            
                if meta['dtype'] == "uint16" or meta['dtype'] == "uint8":
                    nodata = 0
                else:
                    nodata = -9999
                
                meta.update({'driver': 'Gtiff'})
                ext = "tif"
                
                if subregion == "raw":
                    new_name = "T{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, band, res, new, ext)
                else:
                    new_name = "T{}_{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, band, res, subregion, new, ext)

                if os.path.exists(os.path.join(path, new_name)) == True and os.stat(os.path.join(path, new_name)).st_size != 0:
                    # Pass if file already exists & it's size is not zero
                    logging.warning("File {} already exists...".format(os.path.join(path, new_name)))
                    try:
                        getattr(self, band)[str(res)][subregion] = os.path.join(path, new_name)
                    except KeyError:
                        getattr(self, band).update({str(res): {subregion: os.path.join(path, new_name)}})    
                    return

                if hasattr(self, 'SCL'):
                    if res in self.SCL:
                        SCL = self.SCL[res][subregion]
                    else:
                        if subregion == "raw":
                            self.upsample(store = store, band = "SCL")
                            SCL = self.SCL[res][subregion]
                        else:    
                            self.upsample(store = store, band = "SCL", subregion = subregion)
                            SCL = self.SCL[res][subregion]
                else:
                    raise BandNotFound("SCL band is missing!")
                
                SCL_data = rasterio.open(SCL)
                mask = self._cloud_mask(SCL_data)                 
                
                array = src.read(1)
                array[mask==0] = nodata
                meta.update({"nodata": nodata})
                if compress:
                    meta.update({"compress": "lzw"})                    
                  
                with rasterio.open(os.path.join(path, new_name), "w", **meta) as dest:
                    dest.write(array, 1)
            
            getattr(self, band)[str(res)][subregion] = os.path.join(path, new_name)

    def calcVI(self, index, store = None, subregion = None, verbose:bool = False, compress = False):
        """Calculates a selected vegetation index (NDVI, NDBI, NDWI).
        Args:
            index (str): Vegetation index to be calculated and saved. Currently only NDVI, NDBI, NDWI are supported
        """
        driver = "Gtiff"
        ext = "tif"

        if subregion is None:
            region = "raw"
            new_name = "T{}_{}_{}.{}".format(self.tile_id, self.str_datetime, index, ext)

        else:
            region = subregion
            new_name = "T{}_{}_{}_{}_{}.{}".format(self.tile_id, self.str_datetime, index, self.setResolution(index), subregion, ext)

        if index == 'NDVI':
            if store == None:
                if os.path.isfile(os.path.join(self.datapath_10, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(self.datapath_10, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_10, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
                    
                    if hasattr(self, "B04"):
                        if self.B04.get("10").get(region) != None:
                            red = rasterio.open(self.B04["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B04 (Image: {})".format(self, self.name))
                    
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))
                    
                    nir_array = nir.read().astype(rasterio.float32)
                    red_array = red.read().astype(rasterio.float32)
                    ndvi_array = vi.ndvi(red_array, nir_array)
                    ndvi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndvi_array[red_array == red.meta["nodata"]] = -9999.
                    path = self.datapath_10
                    metadata = red.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndvi_array.dtype, "nodata": -9999.})
                    if compress:
                        metadata.update({"compress": "lzw"})                    

                    self.writeResults(path, new_name, ndvi_array, metadata)
                    # Setting NDVI attribute to S2 image
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_10, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})   
            else:
                if os.path.isfile(os.path.join(store, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(store, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
                    
                    if hasattr(self, "B04"):
                        if self.B08.get("10").get(region) != None:
                            red = rasterio.open(self.B04["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B04 (Image: {})".format(self, self.name))
                    
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))
                    nir_array = nir.read().astype(rasterio.float32)
                    red_array = red.read().astype(rasterio.float32)
                    ndvi_array = vi.ndvi(red_array, nir_array)
                    ndvi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndvi_array[red_array == red.meta["nodata"]] = -9999.
                    path = store
                    metadata = red.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndvi_array.dtype, "nodata": -9999.})
                    if compress:
                        metadata.update({"compress": "lzw"})                    

                    self.writeResults(path, new_name, ndvi_array, metadata)
                    # Setting NDVI attribute to S2 image
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})
        elif index == 'NDWI':
            if store == None:
                if os.path.isfile(os.path.join(self.datapath_10, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(self.datapath_10, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_10, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
                    
                    if hasattr(self, "B03"):
                        if self.B03.get("10").get(region) != None:
                            green = rasterio.open(self.B03["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B03 (Image: {})".format(self, self.name))
                    
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))
                    
                    nir_array = nir.read().astype(rasterio.float32)
                    green_array = green.read().astype(rasterio.float32)
                    ndwi_array = vi.ndwi(green_array, nir_array)
                    ndwi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndwi_array[green_array == green.meta["nodata"]] = -9999.
                    path = self.datapath_10
                    metadata = green.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndwi_array.dtype, "nodata": -9999.})
                    if compress:            
                        metadata.update({"compress": "lzw"})                    
                    self.writeResults(path, new_name, ndwi_array, metadata)
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_10, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_10, new_name)}})
            else:
                if os.path.isfile(os.path.join(store, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(store, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
                    
                    if hasattr(self, "B03"):
                        if self.B03.get("10").get(region) != None:
                            green = rasterio.open(self.B03["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B03 (Image: {})".format(self, self.name))
                    
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))
                    
                    nir_array = nir.read().astype(rasterio.float32)
                    green_array = green.read().astype(rasterio.float32)
                    ndwi_array = vi.ndwi(green_array, nir_array)
                    ndwi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndwi_array[green_array == green.meta["nodata"]] = -9999.
                    metadata = green.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndwi_array.dtype, "nodata": -9999.})
                    if compress:
                        metadata.update({"compress": "lzw"})                    
                    self.writeResults(store, new_name, ndwi_array, metadata)
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})                    
        elif index == 'NDBI':
            if store == None:
                if os.path.isfile(os.path.join(self.datapath_20, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(self.datapath_20, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_20, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_20, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_20, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B11"):
                        if self.B11.get("20").get(region) != None:
                            swir = rasterio.open(self.B11["20"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B11 (Image: {})".format(self, self.name))

                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
            
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))
                    
                    nir_array, _ = self.reproj_match(self.B08["10"][region], self.B11["20"][region])
                    nir_array = nir_array.astype(rasterio.float32)
                    swir_array = swir.read().astype(rasterio.float32)
                    ndbi_array = vi.ndbi(swir_array, nir_array)
                    ndbi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndbi_array[swir_array == swir.meta["nodata"]] = -9999.
                    path = self.datapath_20
                    metadata = swir.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndbi_array.dtype, "nodata": -9999})
                    self.writeResults(path, new_name, ndbi_array, metadata)
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(self.datapath_20, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(self.datapath_20, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(self.datapath_20, new_name)}})  
            else:
                if os.path.isfile(os.path.join(store, new_name)):
                    logging.warning("File {} already exists...".format(os.path.join(store, new_name)))
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})                    
                    return
                else:
                    if hasattr(self, "B11"):
                        if self.B11.get("20").get(region) != None:
                            swir = rasterio.open(self.B11["20"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B11 (Image: {})".format(self, self.name))

                    if hasattr(self, "B08"):
                        if self.B08.get("10").get(region) != None:
                            nir = rasterio.open(self.B08["10"][region])
                        else:
                            raise BandNotFound("{} object has no stored path for resolution {} and raw data.".format(self, self.setResolution(index)))
                    else:
                        raise BandNotFound("{} object has no attribute B08 (Image: {})".format(self, self.name))
            
                    if verbose:
                        logging.info("Calculating {} for image {}...".format(index, self.name))

                    nir_array, _ = self.reproj_match(self.B08["10"][region], self.B11["20"][region])
                    nir_array = nir_array.astype(rasterio.float32)
                    swir_array = swir.read().astype(rasterio.float32)
                    ndbi_array = vi.ndbi(swir_array, nir_array)
                    ndbi_array[nir_array == nir.meta["nodata"]] = -9999.
                    ndbi_array[swir_array == swir.meta["nodata"]] = -9999.
                    metadata = swir.meta.copy()
                    metadata.update({"driver": driver, "dtype": ndbi_array.dtype, "nodata": -9999})
                    if compress:
                        metadata.update({"compress": "lzw"})                    
                    self.writeResults(store, new_name, ndbi_array, metadata)
                    if not hasattr(self, index):
                        setattr(self, index, {self.setResolution(index): {region: os.path.join(store, new_name)}})
                    else:
                        try:
                            getattr(self, index)[self.setResolution(index)][region] = os.path.join(store, new_name)
                        except KeyError:
                            getattr(self, index).update({self.setResolution(index): {region: os.path.join(store, new_name)}})  

        else:
            VegetationIndexNotInList(f"Index {index} not in list of available indexes.")            