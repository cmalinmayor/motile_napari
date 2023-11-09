import csv

import networkx as nx
import numpy as np


def _convert_types(row):
    """Helper function for loading the tracks csv with correct types
    Designed for mskcc_confocal dataset.

    Args:
        row (dict): Row from csv.DictReader

    Returns:
        dict: Same row with the types converted from strings to ints/floats
        for the appropriate keys.
    """
    int_vals = ['t', 'cell_id', 'parent_id', 'track_id', 'div_state']
    float_vals = ['z', 'y', 'x', 'radius']
    for k in int_vals:
        row[k] = int(row[k])
    for k in float_vals:
        row[k] = float(row[k])
    return row


def load_mskcc_confocal_tracks(tracks_path, frames=None):
    """Load tracks from a csv to a networkx graph.
    """
    graph = nx.DiGraph()
    with open(tracks_path, 'r') as f:
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


def assign_tracklet_ids(graph):
    """Add a tracklet_id attribute to a graph by removing division edges,
    assigning one id to each connected component.
    Designed as a helper for visualizing the graph in the napari Tracks layer.

    Args:
        graph (nx.DiGraph): A networkx graph with a tracking solution

    Returns:
        nx.DiGraph: The same graph with the tracklet_id assigned. Probably 
        occurrs in place but returned just to be clear.
    """
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
        nx.set_node_attributes(
            graph, {node: {"tracklet_id": track_id} for node in tracklet})
        track_id += 1
    return graph, intertrack_edges


def to_napari_tracks_layer(graph, frame_key="t", location_keys=("y", "x"), properties=()):
    """Function to take a networkx graph and return the data needed to add to 
    a napari tracks layer.

    Args:
        graph (nx.DiGraph): _description_
        frame_key (str, optional): Key in graph attributes containing time frame.
            Defaults to "t".
        location_keys (tuple, optional): Keys in graph node attributes containing
            location. Should be in order: (Z), Y, X. Defaults to ("y", "x").
        properties (tuple, optional): Keys in graph node attributes to add
            to the visualization layer. Defaults to (). NOTE: not working now :(

    Returns:
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
            parents, but only one child) in the case of track merging.
    """
    napari_data = np.zeros((graph.number_of_nodes(), len(location_keys) + 2))
    napari_properties = {prop: np.zeros(
        graph.number_of_nodes()) for prop in properties}
    napari_edges = {}
    graph, intertrack_edges = assign_tracklet_ids(graph)
    for index, node in enumerate(graph.nodes(data=True)):
        node_id, data = node
        location = [data[loc_key] for loc_key in location_keys]
        napari_data[index] = [data['tracklet_id'], data[frame_key]] + location
        for prop in properties:
            if prop in data:
                napari_properties[prop][index] = data[prop]
    napari_edges = {}
    for parent, child in intertrack_edges:
        parent_track_id = graph.nodes[parent]['tracklet_id']
        child_track_id = graph.nodes[child]['tracklet_id']
        if child_track_id in napari_edges:
            napari_edges[child_track_id].append(parent_track_id)
        else:
            napari_edges[child_track_id] = [parent_track_id]
    return napari_data, napari_properties, napari_edges
