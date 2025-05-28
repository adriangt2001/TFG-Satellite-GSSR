import re
import os
import copy
import numpy as np
from collections import OrderedDict

import rasterio
from rasterio.windows import Window

import torch
import torch.utils.data

class ResolutionGroup:
    """Handles bands of the same resolution"""
    def __init__(self, paths: str | list, res: int, bands: str | list) -> None:
        self.bands = []
        self.paths = []
        self.srcs = []
        self.res = res
        self._bounds = None
        self._crs = None
        self.add_paths(paths, bands)
        
    def open(self, mode: str) -> None:
        if len(self.srcs) != 0: return
        for path in self.paths:
            self.srcs.append(rasterio.open(path, mode))
            
    def close(self) -> None:
        if len(self.srcs) == 0: return
        for src in self.srcs:
            src.close()
        self.srcs.clear()

    def add_paths(self, paths: str | list, bands: str | list) -> None:
        if isinstance(paths, str):
            self.bands.append(bands)
            self.paths.append(paths)
        else:
            self.bands += bands
            self.paths += paths
            
    def sort(self) -> None:
        self.bands, self.paths = [list(i) for i in zip(*sorted(zip(self.bands, self.paths),
                                            key=lambda dual: dual[0]))]

    def get_patch(self, window: Window) -> np.ndarray:
        """Read a patch from all bands at given window"""
        patches = []
        if len(self.srcs) == 0:
            for path in self.paths:
                with rasterio.open(path, 'r') as src:
                    patch = src.read(1, window=window)
                    patches.append(patch)
        else:
            for src in self.srcs:
                patch = src.read(1, window=window)
                patches.append(patch)
        return np.stack(patches)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self._bounds:
            if len(self.srcs) == 0:
                with rasterio.open(self.paths[0], 'r') as src:
                    self._bounds = src.bounds
            else:
                self._bounds = self.srcs[0].bounds
        return self._bounds

    @property 
    def crs(self) -> str:
        if not self._crs:
            if len(self.srcs) == 0:
                with rasterio.open(self.paths[0], 'r') as src:
                    self._crs = src.crs
            else:
                self._crs = self.srcs[0].crs
        return self._crs

    @property
    def shape(self) -> tuple[int, int]:
        """Get height and width of raster"""
        if len(self.srcs) == 0:
            with rasterio.open(self.paths[0], 'r') as src:
                return src.height, src.width
        return self.srcs[0].height, self.srcs[0].width

    @property
    def transform(self) -> rasterio.Affine:
        if len(self.srcs) == 0:
            with rasterio.open(self.paths[0], 'r') as src:
                return src.transform
        return self.srcs[0].transform

class Sen2Dataset(torch.utils.data.Dataset):
    """Sentinel-2 Dataset that divides bands into patches based on resolution and creates downsampled versions for super-resolution training"""
    
    filename_regex = r"""
        ^T(?P<tile>\d{2}[A-Z]{3})
        _(?P<date>\d{8}T\d{6})
        _(?P<band>B[018][\dA])
        (?:_(?P<resolution>.*m))?
        \..*$
    """

    PATCH_SIZES = {
        60: 96,     # 60m bands -> 96x96 patches (16×6)
        20: 288,    # 20m bands -> 288x288 patches (48×6)
        10: 576     # 10m bands -> 576x576 patches (96×6)
    }

    # Define which bands belong to each resolution
    BANDS = {
        10: ['B02', 'B03', 'B04', 'B08'],           # 10m bands
        20: ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],  # 20m bands
        60: ['B01', 'B09']                   # 60m bands
    }

    def __init__(self, root_dir: str, input_res: int, target_res: int, transform=None, downsampling=None, preprocessing=None):
        """
        Args:
            root_dir (str): Directory containing Sentinel-2 bands
            input_res (list): List of input resolutions
            target_res (int): Target resolution
        Kwargs:
            transform (callable, optional): Transform to apply to patches
            downsampling (callable, optional): Downsampling transform to apply to input patches
        """
        self.input_res = copy.deepcopy(input_res)
        self.target_res = target_res
        self.transform = transform
        self.downsampling = downsampling
        self.preprocessing = preprocessing
        self.raster_groups = self._find_rasters(root_dir)
        self.x_patches, self.y_patches = self._generate_patches()
        
    def _find_rasters(self, root_dir: str) -> OrderedDict:
        """Find and group raster files by resolution"""
        raster_groups = {}
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        for root, _, files in sorted(os.walk(root_dir)):
            if not files:
                continue

            for name in files:
                matched = re.match(filename_regex, name)
                if matched:
                    path = os.path.join(root, matched.string)
                    res = int(matched.group('resolution')[:-1])
                    band = matched.group('band')
                    
                    # Only add the band if it belongs to this resolution
                    if band in self.BANDS[res]:
                        if res in raster_groups:
                            raster_groups[res].add_paths(path, band)
                        else:
                            raster_groups[res] = ResolutionGroup(path, res, band)

        # Verify we have all required bands for each resolution
        for res, required_bands in self.BANDS.items():
            if res not in raster_groups:
                raise ValueError(f"Missing all bands for {res}m resolution")
            
            found_bands = set(raster_groups[res].bands)
            missing_bands = set(required_bands) - found_bands
            if missing_bands:
                raise ValueError(f"Missing required bands for {res}m resolution: {missing_bands}")

        for raster in raster_groups.values():
            raster.sort()
            
        return OrderedDict(sorted(raster_groups.items()))

    def _generate_patches(self) -> list:
        """Generate and store all patches for each resolution in memory"""
        patches = {10: [], 20: [], 60: []}

        # Get dimensions of highest resolution
        base_height, base_width = self.raster_groups[min(self.raster_groups.keys())].shape
        base_patch_size = self.PATCH_SIZES[min(self.raster_groups.keys())]
        
        # Calculate number of complete patches that fit within the image
        n_patches_h = base_height // base_patch_size
        n_patches_w = base_width // base_patch_size
        
        # Open all raster files
        for group in self.raster_groups.values():
            group.open('r')

        # Generate windows for each patch
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                
                for res, group in self.raster_groups.items():
                    patch_size = self.PATCH_SIZES[res]
                    row_off = int(i * patch_size)
                    col_off = int(j * patch_size)
                    
                    window = Window(col_off, row_off, patch_size, patch_size)
                    patch = group.get_patch(window)
                    patch = torch.from_numpy(patch).float()
                    
                    patches[res].append(patch)
        
        # Close all raster files
        for group in self.raster_groups.values():
            group.close()
        
        x_patches = [torch.stack(patches[res]) for res in self.input_res]
        y_patches = torch.stack(patches[self.target_res])

        if self.transform:
            x_patches = [self.transform(patch) for patch in x_patches]
            y_patches = self.transform(y_patches)
        
        if self.preprocessing:
            x_patches, y_patches = self.preprocessing(x_patches, y_patches)
        
        if self.downsampling:
            x_patches = [self.downsampling(patch) for patch in x_patches]
        
        return x_patches, y_patches

    def __len__(self):
        return len(self.x_patches[0])

    def __getitem__(self, idx):
        """Get pre-generated patches for all resolutions at index"""
        x = [patch[idx] for patch in self.x_patches]
        y = self.y_patches[idx]
        return x, y
