#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import fnmatch
import geopandas as gpd
from tqdm import tqdm
import pkg_resources
import rasterio
from rasterio.merge import merge

from cropmaps.sentinels import sentinel2
from cropmaps.exceptions import NoDataError
from cropmaps.clipper import Clipper
from cropmaps.ts import timeseries
from cropmaps.utils import worldcover

from cropmaps import logger
logging = logger.setup(name = __name__)

class sentimeseries(timeseries):
    """Sentinel 2 time series."""
    
    def __init__(self, name:str):
        timeseries.__init__(self, name)
        self.tiles = []
        self.orbits = []

    def find_DIAS(self, DIAS_data:list, level:str = "L2A"):
        
        for d in DIAS_data:
            path, name = os.path.split(d)
            logging.info(f"Reading image {name}...")
            image = sentinel2(path, name)
            image.getmetadata()
            image.getBands()
            self.data.append(image)
            self.names.append(image.name)
            self.dates.append(image.datetime)
            self.cloud_cover.append(image.cloud_cover)
            self.tiles.append(image.tile_id)
            self.orbits.append(image.orbit)
        
        if len(self.data) == 0:
            raise NoDataError("0 Sentinel 2 raw data found in the selected path.")
        else:
            self.total = len(self.data)
        
        if len(list(set(self.tiles))) > 1:
            logging.warning("Available data are in more than one tiles!")

    def find(self, path:str, level:str = "L2A"):
        """Finds automatically all the available data in a provided path based on the S2 level
        product (L1C or L2A) that the user provides.
        
        Args:
            path (str, path-like): Search path
            level (str, optional): Level of the S2 time series (L1C or L2A). Defaults to 'L2A'.
        
        Raises:
            NoDataError: Raises when no data were found in the provided path
        """
        image = None
        for (dirpath, _, _) in os.walk(path):
            for file in os.listdir(dirpath):
                # Find data
                if fnmatch.fnmatch(str(file), '*{}*.SAFE'.format(level)):
                    
                    image = sentinel2(dirpath, file)
                    image.getmetadata()
                    image.getBands()
                    self.data.append(image)
                    self.names.append(image.name)
                    self.dates.append(image.datetime)
                    self.cloud_cover.append(image.cloud_cover)
                    self.tiles.append(image.tile_id)

        if len(self.data) == 0:
            raise NoDataError("0 Sentinel-2 raw data found in the selected path.")
        else:
            self.total = len(self.data)
        
        if len(list(set(self.tiles))) > 1:
            logging.warning("Available data are in more than one tiles!")

    def _path_generator(self, image:sentinel2)->str:
        """Generate path to store processed products.

        Args:
            image (sentinel2): Image as sentinel2 object

        Returns:
            str: Path to store data (satellite/level/year/month/day)
        """
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

    def getVI(self, index:str, store:str = None, image:sentinel2 = None, subregion = None, verbose = False):
        """Calculates a vegetation index for an image if the user provides an image or
        for all the time series.
        
        Args:
            index (str): Vegetation index. Currently works only for NDVI, NDWI, NDBI
            image (sentinel2): If an sentinel2 object is provided calculates VI for this image only
        """
        if store is None:
            # User can provide either the image name or the object.
            if image is None:
                logging.info("Calculating {} for all time series...".format(index))
                
                for im in self.data:
                    im.calcVI(index, subregion = subregion, verbose = verbose)
            else:
                if isinstance(image, sentinel2):
                    image.calcVI(index, subregion = subregion, verbose = verbose)
                else:
                    raise TypeError("Only sentinel2 objects are supported as image!")
        else:     
            if image is None:
                logging.info("Calculating {} for all time series...".format(index))
                for im in self.data:
                    generated_path = self._path_generator(im)
                    savepath = os.path.join(store, generated_path, im.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    im.calcVI(index, store = savepath, subregion = subregion, verbose = verbose)
            else:
                generated_path = self._path_generator(image)
                savepath = os.path.join(store, generated_path, image.name)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                if isinstance(image, sentinel2):
                    image.calcVI(index, store = savepath, subregion = subregion, verbose = verbose)
                else:
                    raise TypeError("Only sentinel2 objects are supported as image!")

    def apply_SCL(self, image:sentinel2 = None, band:str = None, store:str = None, subregion:str = None, resolution:str = None):
        """Apply default SCL mask to S2 images.

        Args:
            image(sentinel2, optional): Apply SCL mask to one image. If None then applies the mask to all the timeseries. Defaults to None.
            band (str, optional): Band to apply SCL mask. If None then applies mask to all default bands. Defaults to None.
            store (str, optional): Path to store the new images. If None then the results are saved in the default S2 path. Defaults to None.
            subregion (str, optional): Apply mask to subregion. If subregion is None then applies the mask to the default raw images. Defaults to None.
            resolution (str, optional): Resolution to apply cloud mask. If None the applies to default resolution else tries to apply to the highest 10m resolution. Defaults to None.
        """

        if image is None:
            logging.info(f"Applying SCL mask for all time series...")
            if store is None:
                for im in self.data:
                    im.apply_cloud_mask(subregion = subregion, band = band, resolution = resolution)
            else:
                for im in self.data:
                    generated_path = self._path_generator(im)
                    savepath = os.path.join(store, generated_path, im.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    im.apply_cloud_mask(subregion = subregion, band = band, store = savepath, resolution = resolution)
        else:
            if store is None:
                image.apply_cloud_mask(subregion=subregion, band=band, resolution = resolution)
            else:
                generated_path = self._path_generator(im)
                savepath = os.path.join(store, generated_path, im.name)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                image.apply_cloud_mask(subregion = subregion, band = band, store = savepath, resolution = resolution)
                    

    def clipbyMask(self, shapefile, image = None, band = None, resize = False, method = None, new = None, store = None):
        """Masks an image or the complete time series with a shapefile.
        
        Args:
            shapefile (path-like, str): Path to shapefile mask
            image (senimage, optional): Masks a specific image. Defaults to None
            band (str, optional): Masks a specific band. Defaults to None
            resize (bool, optional): Resize band. Defaults to False
            method (rasterio.enums.Resampling): Available resampling methods. If None the Nearest is used. Defaults to None
        """
        bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', "SCL"]

        if image is None:
            if band is None:
                logging.info("Masking all time series with {}...".format(shapefile))
                if store is None:
                    for im in tqdm(self.data):
                        for b in bands:
                            Clipper.clipByMask(im, shapefile, band = b, resize = resize, method = method, new = new)
                else:
                    for im in tqdm(self.data):
                        generated_path = self._path_generator(im)
                        savepath = os.path.join(store, generated_path, im.name)
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        for b in bands:
                            Clipper.clipByMask(im, shapefile, store = savepath, band = b, resize = resize, method = method, new = new)
            else:
                logging.info("Masking band {} for all time series with {}...".format(band, shapefile))
                if store is None:
                    for im in tqdm(self.data):
                        Clipper.clipByMask(im, shapefile, band = band, resize = resize, method = method, new = new)
                else:
                    for im in tqdm(self.data):
                        generated_path = self._path_generator(im)
                        savepath = os.path.join(store, generated_path, im.name)
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        Clipper.clipByMask(im, shapefile, store = savepath, band = band, resize = resize, method = method, new = new)
        else:
            if band is None:
                logging.info("Masking {} with {}...".format(image, shapefile))
                if store is None:
                    for b in bands:
                        Clipper.clipByMask(image, shapefile, band = b, resize = resize, method = method, new = new)
                else:
                    generated_path = self._path_generator(image)
                    savepath = os.path.join(store, generated_path, image.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    for b in bands:
                        Clipper.clipByMask(image, shapefile, store = savepath, band = b, resize = resize, method = method, new = new)
            else:
                logging.info("Masking band {} of image {} with {}...".format(band, image, shapefile))
                if store is None:
                    Clipper.clipByMask(image, shapefile, band = band, resize = resize, method = method, new = new)
                else:
                    generated_path = self._path_generator(image)
                    savepath = os.path.join(store, generated_path, image.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath) 
                    Clipper.clipByMask(image, shapefile, store = savepath, band = band, resize = resize, method = method, new = new)

    def remove_orbit(self, orbit):
        """Remove images with specific orbit.
        
        Args:
            orbit (str): Number of orbit
        """
        if not isinstance(orbit, str):
            raise TypeError("Provide orbit as a string!")

        new = []
        for image in self.data:
            if image.orbit == None:
                logging.warning("Image {} has no date information stored!".format(image.name))
            elif image.orbit == orbit:
                logging.info("Removing {} with orbit {}...".format(image.name, image.orbit))
                index = self.names.index(image.name)
                self.names.remove(image.name)
                self.dates.remove(image.datetime)
                self.cloud_cover.pop(index)
                self.tiles.pop(index)
                self.orbits.pop(index)
            else:
                new.append(image)
                logging.info("Keeping {} with orbit {}...".format(image.name, image.orbit))
        self.data = new
        self.total = len(self.data)
        del new
        logging.info("New number of data after removing orbit {} is: {}".format(orbit, len(self.data)))

    def upsample(self, image = None, band = None, method = None, new = None, store = None, subregion = None):
        """Upsample lower resolution bands to the highest available resolution.

        Args:
            image (sentinel2, optional): Image to perform upsampling. If None, then performs upsample to all images in the timeseries. Defaults to None
            band (str, optional): Name of the band to upsample. If None, then performs upsample to all 20m bands. Defaults to None
            method (Resampling, optional): Upsampling method. Defaults to None
            new (str, optional): Name extension. If None the by default adds _Upsampled. Defaults to None
            store (str, optional): Path to store data. If None then stores the data to default image path. Defaults to None
            subregion (str, optional): Perform upsample to subregion. Defaults to None
        """
        bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', "SCL"]

        if image is None:
            if band is None:
                logging.info("Upsampling all time series...")
                if store is None:
                    for im in tqdm(self.data):
                        for b in bands:
                            im.upsample(band = b, store = store, method = method, new = new, subregion = subregion)
                else:
                    for im in tqdm(self.data):
                        generated_path = self._path_generator(im)
                        savepath = os.path.join(store, generated_path, im.name)
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        for b in bands:
                            im.upsample(band = b, store = savepath, method = method, new = new, subregion = subregion)
            else:
                logging.info("Upsampling band {} for all time series...".format(band))
                if store is None:
                    for im in tqdm(self.data):
                            im.upsample(band = band, store = store, method = method, new = new, subregion = subregion)
                else:
                    for im in tqdm(self.data):
                        generated_path = self._path_generator(im)
                        savepath = os.path.join(store, generated_path, im.name)
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        im.upsample(band = band, store = savepath, method = method, new = new, subregion = subregion)
        else:
            if band is None:
                logging.info("Upsampling image {}...".format(image))
                if store is None:
                    for b in bands:
                            image.upsample(band = b, store = store, method = method, new = new, subregion = subregion)
                else:
                    generated_path = self._path_generator(image)
                    savepath = os.path.join(store, generated_path, image.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    for b in bands:
                        image.upsample(band = b, store = savepath, method = method, new = new, subregion = subregion)
            else:
                logging.info("Upsampling band {} of image {}...".format(band, image))
                if store is None:
                    image.upsample(band = band, store = store, method = method, new = new, subregion = subregion)
                else:
                    generated_path = self._path_generator(image)
                    savepath = os.path.join(store, generated_path, image.name)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath) 
                    image.upsample(band = band, store = savepath, method = method, new = new, subregion = subregion)

    def LandCover(self, store, aoi:str = None, outname = "LC_mosaic.tif")->str:
        """Download and add WorldCover masking data to the object.

        Args:
            aoi (str): Path to AOI
            write (bool, optional): Write result to disk. Defaults to False
        
        Returns:
            str: Path with LC data
        """
        if not os.path.exists(store):
            os.makedirs(store)
        regions = []
        if aoi is None:
            # In this case the service will be download data for all the available tiles
            path = pkg_resources.resource_filename(__name__, os.path.join('aux', 'sentinel-2_tiling_grid.geojson'))
            sentinel_tiles = gpd.read_file(path)
            unique_tiles = list(set(self.tiles))
            for tile in unique_tiles:
                regions.append(sentinel_tiles[sentinel_tiles["Name"] == tile].iloc[0].explode().geometry)
        else:
            data = gpd.open(aoi)
            for row in data.iterrows():
                regions.append(row.iloc[0].explode().geometry)

        for region in regions:
            tiles = worldcover(region, store)

        if len(tiles) > 1:
            lc_data = os.listdir(store)
            
            if outname in lc_data:
                lc_data.remove(outname)

            raster_data = []
            for r in lc_data:
                raster = rasterio.open(os.path.join(store, r))
                raster_data.append(raster)

            mosaic, output = merge(raster_data)
            output_meta = raster.meta.copy()
            output_meta.update(
                {"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output,})
            with rasterio.open(os.path.join(store, outname), "w", **output_meta) as m:
                m.write(mosaic)
            
            return os.path.join(store, outname)
        else:
            tile = tiles.ll_tile.iloc[0]
            lc_data = f"ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"

            return os.path.join(store, lc_data)
        
