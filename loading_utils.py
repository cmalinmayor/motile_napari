import csv

import networkx as nx
import numpy as np
import time
from skimage.io import imread
import zarr


def load_mskcc_confocal_tracks(tracks_path, frames=None):
    """Load tracks from a csv to a networkx graph.
    Args:
        tracks_path (str): path to tracks file
        frames (tuple): Tuple of start frame, end frame to limit the tracks to
        these time points. Includes start frame, excludes end frame.
    """
    graph = nx.DiGraph()
    with open(tracks_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # t	z	y	x	cell_id	parent_id	track_id	radius	name	div_state
        for cell in reader:
            cell = _convert_types(cell)
            if frames:
                time = cell["t"]
                if time < frames[0] or time >= frames[1]:
                    continue
            cell_id = cell["cell_id"]
            graph.add_node(cell["cell_id"], **cell)
            parent_id = cell["parent_id"]
            if parent_id != -1:
                if (not frames) or time > frames[0]:
                    graph.add_edge(parent_id, cell_id)
    return graph


def _convert_types(row):
    """Helper function for loading the tracks csv with correct types
    Designed for mskcc_confocal dataset.

    Args:
        row (dict): Row from csv.DictReader

    Returns:
        dict: Same row with the types converted from strings to ints/floats
        for the appropriate keys.
    """
    int_vals = ["t", "cell_id", "parent_id", "track_id", "div_state"]
    float_vals = ["z", "y", "x", "radius"]
    for k in int_vals:
        row[k] = int(row[k])
    for k in float_vals:
        row[k] = float(row[k])
    return row


def load_mskcc_confocal_images(image_path, image_filename, frames=None):
    image_files = sorted(image_path.glob("*.tif"))
    if frames:
        filtered = []
        for t in range(frames[0], frames[1]):
            image_file = image_filename.format({"time": t})
            if image_file in image_files:
                filtered.append(image_file)
        image_files = filtered
    print("starting to load images")
    start_time = time.time()
    images = np.array([imread(imfile) for imfile in image_files])
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to load data at {image_path}")
    print(images.shape)
    print(images.dtype)
    return images


def load_cellulus_results(
    path_to_zarr,
    image_group="test/raw",
    seg_group="post-processed-segmentation-run_2",
    seg_channel=0,
):
    base = zarr.open(path_to_zarr, "r")
    images = base[image_group]
    segmentation = base[seg_group][
        :, seg_channel
    ]  # orginally t, c, z, y, x. want to select channel

    # should return (t, z, y, x) for both
    return np.squeeze(images), np.squeeze(segmentation)
