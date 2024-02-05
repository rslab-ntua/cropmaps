
import numpy as np
# Ignoring all runtime, divided by zero numpy warnings
np.seterr(all='ignore')

class vi():
    """
    A collection of functions for Vegetation Incices in Python. Created for use with Sentinel 2.
    Sources: http://www.sentinel-hub.com/eotaxonomy/indices
    """

    @staticmethod
    def ndvi(red, nir):
        r"""Normalized Difference Vegetation Index
        General formula:
            .. math:: (NIR — VIS)/(NIR + VIS)
        Sentinel 2:
            .. math:: (B08 - B04) / (B08 + B04)
        
        Args:
            red (ndarray): Red numpy array
            nir (ndarray): NIR numpy array
        Returns:
            ndarray: NDVI numpy array
            
        """
    
        return (nir - red) / (nir + red)

    @staticmethod
    def ndwi(green, nir):
        r"""Normalized Difference Water Index
        
        General formula:
            .. math:: (GREEN — NIR)/(GREEN + NIR)
        Sentinel 2:
            .. math:: (B03 - B08) / (B03 + B08)
        
        Args:
            green (ndarray): Green numpy array
            nir (ndarray): NIR numpy array
        Returns:
            ndarray: NDWI numpy array
        """

        return (green - nir) / (green + nir)
    
    @staticmethod
    def ndbi(swir, nir):
        r"""Normalized Difference Buildings Index
        
        General formula:
            .. math:: (SWIR — NIR)/(SWIR + NIR)
        Sentinel 2:
            .. math:: (B11 - B08) / (B11 + B08)
        
        Args:
            swir (ndarray): SWIR numpy array
            nir (ndarray): NIR numpy array
        Returns:
            ndarray: NDBI numpy array
        """

        return (swir - nir) / (swir + nir)