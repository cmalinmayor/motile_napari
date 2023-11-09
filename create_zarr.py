import numpy as np
import zarr
import tifffile
from glob import glob

name = "mskcc_confocal_s1"

# create zarr container
zarr_data = zarr.open(name + ".zarr", "a")

filenames = sorted(glob(name + "/images/*.tif"))

im_data = []
for filename in filenames:
    im = tifffile.imread(filename)  # z y x
    im_data.append(im)

zarr_data["images"] = np.asarray(im_data)  # s z y x
zarr_data["images"].attrs["resolution"] = (5.0, 1, 1)
zarr_data["images"].attrs["axis_names"] = ("s", "c", "z", "y", "x")
