# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: napari_motile
#     language: python
#     name: python3
# ---

# %%
import csv
import time
from pathlib import Path

import napari
import networkx as nx
import numpy as np
from napari_graph import UndirectedGraph
from skimage.io import imread

import motile

# %%
from napari_utils import to_napari_tracks_layer, load_mskcc_confocal_tracks

# %%
DATA_PATH = Path("~/data/mskcc-confocal/mskcc_confocal_s1").expanduser()
IMAGE_PATH = DATA_PATH / "images"
IMAGE_FILENAME = "mskcc_confocal_s1_t{time:03}.tif"
MASK_PATH = DATA_PATH / "tracks" / "mskcc_confocal_s1_mask.hdf"
TRACKS_PATH = DATA_PATH / "tracks" / "tracks.txt"


# %%
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
    images = np.array([imread(imfile) for imfile in image_files])
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to load data at {IMAGE_PATH}")
    print(images.shape)
    print(images.dtype)
    return images


# %%
raw_data = load_images()

# %%
gt_track_graph = load_mskcc_confocal_tracks(TRACKS_PATH)

# %%
gt_track_data, track_props, track_edges = to_napari_tracks_layer(
        gt_track_graph, location_keys=('z', 'y', 'x'), properties=('radius'))


# %%
viewer = napari.Viewer()
viewer.add_image(raw_data, name="raw", scale=([5, 1, 1]))
viewer.add_tracks(gt_track_data, properties=track_props, graph=track_edges, name='gt_tracks')
napari.run()


# %% [markdown]
# ## Delete GT edges
#  - Also determine max length of gt edges to use as distance threshold (plus 10%)

# %%

# %% [markdown]
# ## Create candidate graph by adding edges from t to t+1 within a distance threshold

# %%

# %% [markdown]
# # Create solver
# - add constraints (max children=2, max_parents=2)
# - add costs - edge distance, cost appear, (cost disappear or divide maybe)
# - solve

# %%

# %%
# visualize results in Napari

# %%

# %%
# iterate to improve results

# %% [markdown]
# # Optional stuff
# - evaluate with traccuracy
# - learn weights with ssvm (with small portion of GT)
# - add fake node score (random from .5 to 1, or something)

# %% [markdown]
#
