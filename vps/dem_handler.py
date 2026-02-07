
import rasterio
import numpy as np
import os

class DEMHandler:
    """Handles lookups for Digital Elevation Model (DEM) data."""
    
    def __init__(self, dem_path):
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
            
        self.dem_path = dem_path
        self.src = rasterio.open(dem_path)
        print(f"[DEM] Loaded: {os.path.basename(dem_path)}")
        
    def get_elevation(self, lat, lon):
        """Get elevation in meters at specific lat/lon."""
        try:
            # Note: rasterio sample expects (x, y) which is (lon, lat)
            vals = self.src.sample([(lon, lat)])
            val = next(vals)[0]
            
            # Check for nodata
            if val == self.src.nodata:
                return None
                
            return float(val)
        except Exception as e:
            print(f"[DEM] Error getting elevation for ({lat}, {lon}): {e}")
            return None
            
    def close(self):
        self.src.close()
