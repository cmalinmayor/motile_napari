
import csv
import math
import time
from pathlib import Path

from motile import Solver, TrackGraph
from motile.constraints import MaxChildren, MaxParents
from motile.costs import EdgeSelection, Appear
from motile.variables import NodeSelected, EdgeSelected
import networkx as nx
import numpy as np
import toml
import zarr
from skimage.io import imread
from napari_utils import load_mskcc_confocal_tracks, to_napari_tracks_layer
from tqdm import tqdm
import pprint

from traccuracy import TrackingGraph
from traccuracy.matchers import Matched
from traccuracy.metrics import CTCMetrics, DivisionMetrics


def get_location(node_data, loc_keys=('z', 'y', 'x')):
        return [node_data[k] for k in loc_keys]


def get_max_distance(graph):
    max_dist = 0
    for source, target in graph.edges:
        source_loc = get_location(graph.nodes[source])
        target_loc = get_location(graph.nodes[target])
        dist = math.dist(source_loc, target_loc)
        if dist > max_dist:
            max_dist = dist

    return max_dist

if __name__ == "__main__":
    config_file = "configs/cmm_mskcc_confocal.toml"
    config = toml.load(config_file)
    DATA_PATH = Path(config['base']).expanduser()
    TRACKS_PATH = DATA_PATH / config['tracks']


    gt_track_graph = load_mskcc_confocal_tracks(TRACKS_PATH, frames=(50, 60))
    print(f"GT nodes: {gt_track_graph.number_of_nodes()}")
    print(f"GT edges: {gt_track_graph.number_of_edges()}")

    # Delete GT edges
    #  - Also determine max length of gt edges to use as distance threshold (plus 10%)

    nodes_only = nx.create_empty_copy(gt_track_graph, with_data=True)
    nodes_only

    max_edge_distance = get_max_distance(gt_track_graph)
    dist_threshold = max_edge_distance * 2
    print(f"Dist_threshold: {dist_threshold}")

    # Create candidate graph by adding edges from t to t+1 within a distance threshold

    cand_graph = nodes_only.copy()
    node_frame_dict = {}
    for node, data in cand_graph.nodes(data=True):
        frame = data['t']
        if frame not in node_frame_dict:
            node_frame_dict[frame] = []
        node_frame_dict[frame].append(node)

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

    print(f"Candidate nodes: {cand_graph.number_of_nodes()}")
    print(f"Candidate edges: {cand_graph.number_of_edges()}")

    motile_cand_graph = TrackGraph(cand_graph)
    solver = Solver(motile_cand_graph)

    solver.add_constraints(MaxChildren(2))
    solver.add_constraints(MaxParents(1))

    solver.add_costs(
        EdgeSelection(1, attribute='dist', constant=-20)
    )
    solver.add_costs(
        Appear(30)
    )

    solution = solver.solve()

    node_selected = solver.get_variables(NodeSelected)
    edge_selected = solver.get_variables(EdgeSelected)

    selected_nodes = [node for node in cand_graph.nodes if solution[node_selected[node]] > 0.5]
    selected_edges = [edge for edge in cand_graph.edges if solution[edge_selected[edge]] > 0.5]

    print(f"Selected nodes: {len(selected_nodes)}")
    print(f"Selected edges: {len(selected_edges)}")
    solution_graph = nx.edge_subgraph(cand_graph, selected_edges)

# %% [markdown]
# # Optional stuff
# - evaluate with traccuracy
# - learn weights with ssvm (with small portion of GT)
# - add fake node score (random from .5 to 1, or something)

    traccuracy_gt_graph = TrackingGraph(gt_track_graph)
    node_mapping = [(n, n) for n in solution_graph.nodes()]
    matched = Matched(traccuracy_gt_graph, TrackingGraph(solution_graph), node_mapping)
    
    ctc_metrics = CTCMetrics().compute(matched)
    pprint.pprint(ctc_metrics)

    div_metrics = DivisionMetrics().compute(matched)
    pprint.pprint(div_metrics)

