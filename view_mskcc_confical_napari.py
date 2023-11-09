import time
from pathlib import Path

import napari
import numpy as np
from skimage.io import imread
from tqdm import tqdm

from napari_utils import load_mskcc_confocal_tracks, to_napari_tracks_layer

DATA_PATH = Path("~/data/mskcc-confocal/mskcc_confocal_s1").expanduser()
IMAGE_PATH = DATA_PATH / "images"
IMAGE_FILENAME = "mskcc_confocal_s1_t{time:03}.tif"
MASK_PATH = DATA_PATH / "tracks" / "mskcc_confocal_s1_mask.hdf"
TRACKS_PATH = DATA_PATH / "tracks" / "tracks.txt"


def load_images(frames=None):
    image_files = sorted(IMAGE_PATH.glob("*.tif"))
    if frames:
        filtered = []
        for t in range(frames[0], frames[1]):
            image_file = IMAGE_FILENAME.format({"time": t})
            if image_file in image_files:
                filtered.append(image_file)
        image_files = filtered
    start_time = time.time()
    images = np.array([imread(imfile) for imfile in tqdm(image_files)])
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to load data at {IMAGE_PATH}")
    print(images.shape)
    print(images.dtype)
    return images


if __name__ == "__main__":
    raw = load_images()
    # seg = load_segmentation()
    track_graph = load_mskcc_confocal_tracks(TRACKS_PATH)
    track_data, track_props, track_edges = to_napari_tracks_layer(
        track_graph, location_keys=('z', 'y', 'x'), properties=('radius'))
    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw", scale=([5, 1, 1]))
    viewer.add_tracks(track_data, properties=track_props, graph=track_edges)
    napari.run()
