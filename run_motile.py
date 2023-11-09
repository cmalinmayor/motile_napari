#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import time
from pathlib import Path

import napari
import networkx as nx
import numpy as np
from napari_graph import UndirectedGraph
from skimage.io import imread
import toml
import zarr
import motile


# In[ ]:


from napari_utils import to_napari_tracks_layer, load_mskcc_confocal_tracks


# In[ ]:


config_file = "configs/cmm_config.toml"
config = toml.load(config_file)
DATA_PATH = Path(config['base']).expanduser()
ZARR_PATH = DATA_PATH / config['zarr_dir']
IMAGE_PATH = DATA_PATH / config['image_dir']
IMAGE_FILENAME = config['image_filename']
TRACKS_PATH = DATA_PATH / config['tracks']


# In[ ]:


def load_images(frames=None):
    image_files = sorted(IMAGE_PATH.glob("*.tif"))
    if frames:
        filtered = []
        for t in range(frames[0], frames[1]):
            image_file = IMAGE_FILENAME.format({"time": t})
            if image_file in image_files:
                filtered.append(image_file)
        image_files = filtered
    print("starting to load images")
    start_time = time.time()
    images = np.array([imread(imfile) for imfile in image_files])
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to load data at {IMAGE_PATH}")
    print(images.shape)
    print(images.dtype)
    return images


# In[ ]:


def load_zarr():
    f = zarr.open(ZARR_PATH)
    return f['images']


# In[ ]:


#raw_data = load_images()
raw_data = load_zarr()


# In[ ]:


gt_track_graph = load_mskcc_confocal_tracks(TRACKS_PATH)


# In[ ]:


gt_track_data, track_props, track_edges = to_napari_tracks_layer(
        gt_track_graph, location_keys=('z', 'y', 'x'), properties=('radius'))


# In[ ]:


viewer = napari.Viewer()
viewer.add_image(raw_data, name="raw", scale=([5, 1, 1]))
viewer.add_tracks(gt_track_data, properties=track_props, graph=track_edges, name='gt_tracks')


# In[ ]:


napari.run()


# ## Delete GT edges
#  - Also determine max length of gt edges to use as distance threshold (plus 10%)

# In[ ]:


nodes_only = nx.create_empty_copy(gt_track_graph, with_data=True)
nodes_only


# In[ ]:


def get_location(node_data, loc_keys=('z', 'y', 'x')):
    return [node_data[k] for k in loc_keys]


# In[ ]:


import math
def get_max_distance(graph):
    max_dist = 0
    for source, target in graph.edges:
        source_loc = get_location(graph.nodes[source])
        target_loc = get_location(graph.nodes[target])
        dist = math.dist(source_loc , target_loc)
        if dist > max_dist:
            max_dist = dist

    return max_dist
get_max_distance(gt_track_graph)


# In[ ]:


max_edge_distance = get_max_distance(gt_track_graph)
dist_threshold = max_edge_distance * 1.1
dist_threshold


# ## Create candidate graph by adding edges from t to t+1 within a distance threshold

# In[ ]:





# # Solve with motile!
# - Create solver
# - add constraints (max children=2, max_parents=2)
# - add costs - edge distance, cost appear, (cost disappear or divide maybe)
# - solve

# In[ ]:





# In[ ]:


# visualize results in Napari


# In[ ]:





# In[ ]:


# iterate to improve results


# # Optional stuff
# - evaluate with traccuracy
# - learn weights with ssvm (with small portion of GT)
# - add fake node score (random from .5 to 1, or something)

# 
