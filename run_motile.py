# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import math
from pathlib import Path

import napari
import networkx as nx
import toml
import zarr

# %%

# %%
from napari_utils import to_napari_graph_layer, to_napari_tracks_layer
from loading_utils import load_mskcc_confocal_tracks

# %%
config_file = "configs/cmm_mskcc_confocal_local.toml"
config = toml.load(config_file)
DATA_PATH = Path(config["base"]).expanduser()
TRACKS_PATH = DATA_PATH / config["tracks"]
ZARR_PATH = DATA_PATH / config["zarr_dir"] if "zarr_dir" in config else None


# %%
def load_zarr():
    f = zarr.open(ZARR_PATH)
    return f["images"]


# %%
raw_data = load_zarr()

# %%
gt_track_graph = load_mskcc_confocal_tracks(TRACKS_PATH)

# %%
gt_track_data, track_props, track_edges = to_napari_tracks_layer(
    gt_track_graph, location_keys=("z", "y", "x"), properties=("radius")
)

gt_graph_layer = to_napari_graph_layer(gt_track_graph, "gt_graph")

gt_graph_layer = to_napari_graph_layer(gt_track_graph, "gt_graph")


# %%
viewer = napari.Viewer()
viewer.add_image(raw_data, name="raw", scale=([5, 1, 1]))
viewer.add_tracks(
    gt_track_data, properties=track_props, graph=track_edges, name="gt_tracks"
)
viewer.add_layer(gt_graph_layer)


# %%
napari.run()

# %% [markdown]
# ## Delete GT edges
#  - Also determine max length of gt edges to use as distance threshold (plus 10%)

# %%
nodes_only = nx.create_empty_copy(gt_track_graph, with_data=True)
nodes_only


# %%
def get_location(node_data, loc_keys=("z", "y", "x")):
    return [node_data[k] for k in loc_keys]


# %%
def get_max_distance(graph):
    max_dist = 0
    for source, target in graph.edges:
        source_loc = get_location(graph.nodes[source])
        target_loc = get_location(graph.nodes[target])
        dist = math.dist(source_loc, target_loc)
        if dist > max_dist:
            max_dist = dist

    return max_dist


# %%
max_edge_distance = get_max_distance(gt_track_graph)
dist_threshold = max_edge_distance * 1.1
dist_threshold

# %% [markdown]
# ## Create candidate graph by adding edges from t to t+1 within a distance threshold

# %%
cand_graph = nodes_only.copy()
node_frame_dict = {}
for node, data in cand_graph.nodes(data=True):
    frame = data["t"]
    if frame not in node_frame_dict:
        node_frame_dict[frame] = []
    node_frame_dict[frame].append(node)

# %%
frames = sorted(node_frame_dict.keys())
for frame in tqdm(frames):
    if frame + 1 not in node_frame_dict:
        continue
    next_nodes = node_frame_dict[frame + 1]
    next_locs = [get_location(cand_graph.nodes[n]) for n in next_nodes]
    for node in node_frame_dict[frame]:
        loc = get_location(cand_graph.nodes[node])
        for next_id, next_loc in zip(next_nodes, next_locs):
            dist = math.dist(next_loc, loc)
            if dist < dist_threshold:
                cand_graph.add_edge(node, next_id, dist=dist)


# %%
cand_graph.number_of_edges()

# %%
cand_graph.number_of_nodes()

# %% [markdown]
# # Optional: Visualize Candidate Graph with Napari Graph Layer


# %%
cand_graph_layer = to_napari_graph_layer(cand_graph, "candidate_graph")

# %%
viewer = napari.Viewer()
viewer.add_image(raw_data, name="raw", scale=([5, 1, 1]))
viewer.add_layer(cand_graph_layer)

# %%
napari.run()

# %% [markdown]
# # Solve with motile!
# - Create solver
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
