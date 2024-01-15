import math
from pathlib import Path
import numpy as np

from motile import Solver, TrackGraph
from motile.constraints import MaxChildren, MaxParents
from motile.costs import EdgeSelection, Appear
from motile.variables import NodeSelected, EdgeSelected
import networkx as nx
import toml
from loading_utils import load_cellulus_results
from tqdm import tqdm
import pprint
import time
from skimage.measure import regionprops
import tifffile
from traccuracy import TrackingGraph
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.loaders import load_ctc_data
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
# logging.getLogger('traccuracy.matchers._ctc').setLevel(logging.DEBUG)


def get_cand_graph_from_segmentation(
    segmentation, max_edge_distance, pos_labels=["y", "x"]
):
    """_summary_

    Args:
        segmentation (np.array): A numpy array with shape (t, [z,], y, x)
    """
    # add nodes
    node_frame_dict = (
        {}
    )  # construct a dictionary from time frame to node_id for efficiency
    cand_graph = nx.DiGraph()

    for t in range(len(segmentation)):
        nodes_in_frame = []
        props = regionprops(segmentation[t])
        for i, regionprop in enumerate(props):
            node_id = f"{t}_{regionprop.label}"  # TODO: previously node_id= f"{t}_{i}"
            attrs = {
                "t": t,
                "segmentation_id": regionprop.label,
                "area": regionprop.area,
            }
            centroid = regionprop.centroid  # [z,] y, x
            for label, value in zip(pos_labels, centroid):
                attrs[label] = value
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        node_frame_dict[t] = nodes_in_frame

    print(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    frames = sorted(node_frame_dict.keys())
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_nodes = node_frame_dict[frame + 1]
        next_locs = [
            get_location(cand_graph.nodes[n], loc_keys=pos_labels) for n in next_nodes
        ]
        for node in node_frame_dict[frame]:
            loc = get_location(cand_graph.nodes[node], loc_keys=pos_labels)
            for next_id, next_loc in zip(next_nodes, next_locs):
                dist = math.dist(next_loc, loc)
                attrs = {
                    "dist": dist,
                }
                if dist < max_edge_distance:
                    cand_graph.add_edge(node, next_id, **attrs)

    print(f"Candidate edges: {cand_graph.number_of_edges()}")
    return cand_graph


def get_location(node_data, loc_keys=("z", "y", "x")):
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


def solve_with_motile(cand_graph):
    motile_cand_graph = TrackGraph(cand_graph)
    solver = Solver(motile_cand_graph)

    solver.add_constraints(MaxChildren(2))
    solver.add_constraints(MaxParents(1))

    solver.add_costs(EdgeSelection(1, attribute="dist", constant=-20))
    solver.add_costs(Appear(30))

    start_time = time.time()
    solution = solver.solve()
    print(f"Solution took {time.time() - start_time} seconds")
    return solution, solver


def get_solution_nx_graph(solution, solver):
    node_selected = solver.get_variables(NodeSelected)
    edge_selected = solver.get_variables(EdgeSelected)

    selected_nodes = [
        node for node in cand_graph.nodes if solution[node_selected[node]] > 0.5
    ]
    selected_edges = [
        edge for edge in cand_graph.edges if solution[edge_selected[edge]] > 0.5
    ]

    print(f"Selected nodes: {len(selected_nodes)}")
    print(f"Selected edges: {len(selected_edges)}")
    solution_graph = nx.edge_subgraph(cand_graph, selected_edges)
    return solution_graph


def evaluate_with_traccuracy(ds_name, ctc_data_path, solution_graph, solution_seg):
    gt_tracking_graph = load_ctc_data(
        ctc_data_path, track_path=ctc_data_path / "man_track.txt"
    )
    pred_tracking_graph = TrackingGraph(solution_graph, segmentation=solution_seg)
    for node in pred_tracking_graph.nodes:
        assert pred_tracking_graph.nodes[node][pred_tracking_graph.label_key]
    matcher = CTCMatcher()
    matched = matcher.compute_mapping(gt_tracking_graph, pred_tracking_graph)

    ctc_metrics = CTCMetrics().compute(matched)
    pprint.pprint(ctc_metrics)

    div_metrics = DivisionMetrics().compute(matched)
    pprint.pprint(div_metrics)


def save_result_tifs_res_track(solution_nx_graph, segmentation, output_tif_dir):
    tracked_masks = np.zeros_like(segmentation)
    new_mapping = {}  # <t_id> in segmentation mask: id in tracking mask
    res_track = {}  # id in tracking mask: t_start, t_end, parent_id in tracking mask
    id_counter = 1
    for in_node, out_node in tqdm(solution_nx_graph.edges()):
        t_in, id_in = in_node.split("_")
        t_out, id_out = out_node.split("_")
        t_in, id_in = int(t_in), int(id_in)
        t_out, id_out = int(t_out), int(id_out)
        num_out_edges = len(solution_nx_graph.out_edges(in_node))
        if num_out_edges == 1:
            if in_node in new_mapping.keys():
                # i.e. continuation of an existing edge
                res_track[new_mapping[in_node]][
                    1
                ] = t_out  # update the end time for this tracklet
                tracked_masks[t_in][segmentation[t_in] == id_in] = new_mapping[in_node]
                new_mapping[out_node] = new_mapping[in_node]
                tracked_masks[t_out][segmentation[t_out] == id_out] = new_mapping[
                    out_node
                ]
            else:
                # i.e. start of a new edge
                res_track[id_counter] = [t_in, t_out, 0]
                new_mapping[in_node] = id_counter
                new_mapping[out_node] = id_counter
                tracked_masks[t_in][segmentation[t_in] == id_in] = id_counter
                tracked_masks[t_out][segmentation[t_out] == id_out] = id_counter
                id_counter += 1
        elif num_out_edges == 2:
            out_edge1, out_edge2 = solution_nx_graph.out_edges(in_node)
            _, out_node1 = out_edge1
            _, out_node2 = out_edge2
            t_out1, id_out1 = out_node1.split("_")
            t_out1, id_out1 = int(t_out1), int(id_out1)
            t_out2, id_out2 = out_node2.split("_")
            t_out2, id_out2 = int(t_out2), int(id_out2)
            if in_node in new_mapping.keys():
                # i.e. in node was connected by one outgoing edge previously
                res_track[new_mapping[in_node]][1] = t_in
                tracked_masks[t_in][segmentation[t_in] == id_in] = new_mapping[in_node]
                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    tracked_masks[t_out1][segmentation[t_out1] == id_out1] = id_counter
                    res_track[id_counter] = [t_out1, t_out1, new_mapping[in_node]]
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    tracked_masks[t_out2][segmentation[t_out2] == id_out2] = id_counter
                    res_track[id_counter] = [t_out2, t_out2, new_mapping[in_node]]
                    id_counter += 1
            else:
                res_track[id_counter] = [
                    t_in,
                    t_in,
                    0,
                ]  # since it divides immediately after
                new_mapping[in_node] = id_counter
                tracked_masks[t_in][segmentation[t_in] == id_in] = id_counter
                id_counter += 1
                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    tracked_masks[t_out1][segmentation[t_out1] == id_out1] = id_counter
                    res_track[id_counter] = [t_out1, t_out1, new_mapping[in_node]]
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    tracked_masks[t_out2][segmentation[t_out2] == id_out2] = id_counter
                    res_track[id_counter] = [t_out2, t_out2, new_mapping[in_node]]
                    id_counter += 1
    # ensure that path where tifs will be saved, exists.
    if Path(output_tif_dir).exists():
        pass
    else:
        Path(output_tif_dir).mkdir()
    # write tifs
    for i in range(tracked_masks.shape[0]):
        tifffile.imwrite(
            Path(output_tif_dir) / ("mask" + str(i).zfill(3) + ".tif"),
            tracked_masks[i].astype(np.uint16),
        )
    # write res_track.txt
    res_track_list = []
    for key in res_track.keys():
        res_track_list.append(
            [key, res_track[key][0], res_track[key][1], res_track[key][2]]
        )
    np.savetxt(
        Path(output_tif_dir) / ("res_track.txt"), np.asarray(res_track_list), fmt="%i"
    )
    return new_mapping, res_track, tracked_masks


if __name__ == "__main__":
    config_file = "configs/cellulus_hela.toml"
    config = toml.load(config_file)
    cellulus_data_path = Path(config["zarr_dataset"])
    cellulus_dataset_name = config["dataset_name"]
    ctc_data_path = Path(config["ctc_format"])
    edge_dist_threshold = config["edge_distance_threshold"]
    ds_name = config["ds_name"]
    output_tifs_directory = config["output_tifs_directory"]

    print(f"Data path: {cellulus_data_path}")
    print(f"Path exists?: {cellulus_data_path.exists()}")
    images, segmentation = load_cellulus_results(
        cellulus_data_path, seg_group=cellulus_dataset_name
    )
    print(f"Image shape: {images.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    cand_graph = get_cand_graph_from_segmentation(segmentation, edge_dist_threshold)
    print(f"Cand graph has {cand_graph.number_of_nodes()} nodes")

    solution, solver = solve_with_motile(cand_graph)
    solution_nx_graph = get_solution_nx_graph(solution, solver)
    # evaluate_with_traccuracy(ds_name, ctc_data_path, solution_nx_graph, segmentation)
    new_mapping, res_track, new_segmentations = save_result_tifs_res_track(
        solution_nx_graph, segmentation, output_tifs_directory
    )


print(f"Default solver weights are:\n{solver.weights}")
