import math
from pathlib import Path

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
from traccuracy import TrackingGraph
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.loaders import load_ctc_data
import logging
from saving_utils import save_result_tifs_res_track

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
# logging.getLogger('traccuracy.matchers._ctc').setLevel(logging.DEBUG)


def get_cand_graph_from_segmentation(
    segmentation, max_edge_distance, pos_labels=["y", "x"], f_a=30
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
                "cost_appear": 0 if t == 0 else f_a,
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


def solve_with_motile(cand_graph, w_e=1, b_e=-20, w_a=1, b_a=0):
    motile_cand_graph = TrackGraph(cand_graph)
    solver = Solver(motile_cand_graph)

    solver.add_constraints(MaxChildren(2))
    solver.add_constraints(MaxParents(1))

    solver.add_costs(EdgeSelection(w_e, attribute="dist", constant=b_e))
    solver.add_costs(Appear(weight=w_a, attribute="cost_appear", constant=b_a))

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
    
    # specify weights
    w_e = 1
    b_e = -20
    w_a = 1
    b_a = 0
    
    # specify attribute for appearance
    f_a = 30

    # note now there would be 4 weights which would be needed (w_e, b_e, w_a=1, b_a=0)
    cand_graph = get_cand_graph_from_segmentation(
        segmentation, edge_dist_threshold, f_a=f_a
    )
    print(f"Cand graph has {cand_graph.number_of_nodes()} nodes")

    solution, solver = solve_with_motile(cand_graph, w_e=w_e, b_e=b_e, w_a = w_a, b_a = b_a)
    solution_nx_graph = get_solution_nx_graph(solution, solver)
    # evaluate_with_traccuracy(ds_name, ctc_data_path, solution_nx_graph, segmentation)
    # new_mapping, res_track, new_segmentations = save_result_tifs_res_track(
    #     solution_nx_graph, segmentation, output_tifs_directory
    # )
    print(f"Value of objective function after optimisation is {solver.solution.get_value()}")


print(f"Default solver weights are:\n{solver.weights}")

# ## SSVM

# Select a lineage tree randomly from the ground truth. <br>
# Here, we select the zeroth weakly connected graph!

gt_track_graph = load_ctc_data(
    ctc_data_path, track_path=ctc_data_path / "man_track.txt"
)
print(f"Number of GT nodes: {len(gt_track_graph.nodes())}")
print(f"Number of GT edges: {len(gt_track_graph.edges())}")

connected_nodes = list(nx.weakly_connected_components(gt_track_graph.graph))[0]
track = gt_track_graph.graph.subgraph(connected_nodes)

# Next, let's go over the nodes of this track and find the corresponding segmentation id. Set that to `True`

import tifffile
import numpy as np
gt_mask_names = list((ctc_data_path).glob("*.tif"))
for gt_node_in, gt_node_out in track.edges():
    id_in, t_in = gt_node_in.split("_")
    id_out, t_out = gt_node_out.split("_")
    
    t_in, id_in = int(t_in), int(id_in)
    t_out, id_out = int(t_out), int(id_out)
    
    ma_gt_t_in = tifffile.imread(gt_mask_names[t_in])
    ma_gt_t_out = tifffile.imread(gt_mask_names[t_out])

    y_t_in, x_t_in = np.where(ma_gt_t_in == id_in)
    y_t_out, x_t_out = np.where(ma_gt_t_out == id_out)

    ids_t_in = np.unique(segmentation[t_in][y_t_in, x_t_in])
    ids_t_in = ids_t_in[ids_t_in != 0]

    ids_t_out = np.unique(segmentation[t_out][y_t_out, x_t_out])
    ids_t_out = ids_t_out[ids_t_out != 0]

    # Set the corresponding nodes and edges in candidate graph to be True
    # Also set the other outgoing edges from these nodes to be False
    if len(ids_t_in) == 1 and len(ids_t_out) == 1:
        node_id_in = str(t_in) + "_" + str(ids_t_in[0])
        node_id_out = str(t_out) + "_" + str(ids_t_out[0])
        
        cand_graph.nodes[node_id_in]["gt"] = True
        cand_graph.nodes[node_id_out]["gt"] = True

        edges = cand_graph.out_edges(node_id_in)
        for edge in edges:
            _, out_node = edge
            if len(gt_track_graph.graph.out_edges(gt_node_in)) == 1:
                if out_node == node_id_out:
                    cand_graph.edges[(node_id_in, out_node)]["gt"] = True

                else:
                    cand_graph.edges[(node_id_in, out_node)]["gt"] = False
            elif len(gt_track_graph.graph.out_edges(gt_node_in)) == 2:
                if out_node == node_id_out:
                    cand_graph.edges[(node_id_in, out_node)]["gt"] = True
                elif (
                    "gt" in cand_graph.edges[(node_id_in, out_node)].keys()
                    and cand_graph.edges[(node_id_in, out_node)]["gt"]
                ):
                    pass  # must be the other daughter which is already assigned True
                else:
                    cand_graph.edges[(node_id_in, out_node)]["gt"] = False


def fit_weights(solver, regularizer_weight=0.01, max_iterations=5):
    start_time = time.time()
    solver.fit_weights(
        gt_attribute="gt",
        regularizer_weight=regularizer_weight,
        max_iterations=max_iterations,
    )
    optimal_weights = solver.weights
    print(f"Optimal weights are {optimal_weights}")
    solution = solver.solve()
    print(f"Solution took {time.time() - start_time} seconds")
    return solution, solver


regularizer_weight = 0.01
max_iterations = 5
solution, solver = fit_weights(solver, regularizer_weight, max_iterations)
solution_nx_graph = get_solution_nx_graph(solution, solver)

print(
    f"Solver weights after SSVM and using regularization {regularizer_weight} is \n{solver.weights}"
)
