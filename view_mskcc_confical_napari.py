import csv
import time
from pathlib import Path

import h5py
import napari
import networkx as nx
import numpy as np
from napari_graph import UndirectedGraph
from skimage.io import imread

import btrack

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
    images = np.array([imread(imfile) for imfile in image_files])
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to load data at {IMAGE_PATH}")
    print(images.shape)
    print(images.dtype)
    return images


def load_segmentation():
    seg_data = h5py.File(MASK_PATH, 'r')['volumes']['mask']
    print(seg_data.shape)


def _convert_types(row):
    int_vals = ['t', 'cell_id', 'parent_id', 'track_id', 'div_state']
    float_vals = ['z', 'y', 'x', 'radius']
    for k in int_vals:
        row[k] = int(row[k])
    for k in float_vals:
        row[k] = float(row[k])
    return row


def load_tracks(frames=None):
    graph = nx.DiGraph()
    with open(TRACKS_PATH, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # t	z	y	x	cell_id	parent_id	track_id	radius	name	div_state
        for cell in reader:
            cell = _convert_types(cell)
            if frames:
                time = cell["t"]
                if time < frames[0] or time >= frames[1]:
                    continue
            cell_id = cell['cell_id']
            graph.add_node(cell['cell_id'], **cell)
            parent_id = cell['parent_id']
            if parent_id != -1:
                graph.add_edge(parent_id, cell_id)
    return graph


def assign_track_ids(graph):
    graph_copy = graph.copy()

    parents = [node for node, degree in graph.out_degree() if degree >= 2]
    intertrack_edges = []

    # Remove all intertrack edges from a copy of the original graph
    for parent in parents:
        daughters = [child for p, child in graph.out_edges(parent)]
        for daughter in daughters:
            graph_copy.remove_edge(parent, daughter)
            intertrack_edges.append((parent, daughter))

    track_id = 0
    for tracklet in nx.weakly_connected_components(graph_copy):
        nx.set_node_attributes(graph, {node: {"track_id": track_id} for node in tracklet})
        track_id += 1
    return graph, intertrack_edges


def tracks_to_napari(graph, frame_key="t", location_keys=("y", "x"), properties=()):
    """
    locations keys should be in order: (Z), Y, X

    Returns
    -------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
    properties : dict {str: array (N,)}
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    graph : dict {int: list}
        Graph representing associations between tracks. Dictionary defines the
        mapping between a track ID and the parents of the track. This can be
        one (the track has one parent, and the parent has >=1 child) in the
        case of track splitting, or more than one (the track has multiple
        parents, but only one child) in the case of track merging."""
    napari_data = np.zeros((graph.number_of_nodes(), len(location_keys) + 2))
    napari_properties = {prop: np.zeros(graph.number_of_nodes()) for prop in properties}
    napari_edges = {}
    graph, intertrack_edges = assign_track_ids(graph)
    for index, node in enumerate(graph.nodes(data=True)):
        node_id, data = node
        location = [data[loc_key] for loc_key in location_keys] 
        napari_data[index] = [data['track_id'], data[frame_key]] + location
        for prop in properties:
            if prop in data:
                napari_properties[prop][index] = data[prop]
    napari_edges = {}
    for parent, child in intertrack_edges:
        parent_track_id = graph.nodes[parent]['track_id']
        child_track_id = graph.nodes[child]['track_id']
        if child_track_id in napari_edges:
            napari_edges[child_track_id].append(parent_track_id)
        else:
            napari_edges[child_track_id] = [parent_track_id]
    return napari_data, napari_properties, napari_edges


if __name__ == "__main__":
    raw = load_images()
    #seg = load_segmentation()
    track_graph = load_tracks()
    track_data, track_props, track_edges = tracks_to_napari(track_graph, location_keys=('z', 'y', 'x'), properties=('radius'))
    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw", scale=([5, 1, 1]))
    viewer.add_tracks(track_data, properties=track_props, graph=track_edges)
    napari.run()