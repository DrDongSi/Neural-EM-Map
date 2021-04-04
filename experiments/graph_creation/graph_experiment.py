from bisect import bisect_left
from datetime import datetime
import heapq
import json
from math import sqrt
import os
import time
import traceback
from typing import List, Tuple

from deeptracer.tracer import DeepTracer
from deeptracer.common.logging import Logger, LoggingType
from deeptracer.common.protein_structure import AminoAcid, Atom, Model, Chain, ProteinStructure
from deeptracer.common.util import InputFiles
from deeptracer.postprocessing.evaluation import find_matching_residues, matching_percentage, rmsd
from density_graph import DensityGraph
from neural_dataset import select_data
from neural_density_map import NeuralDensityMap


def create_neural_map(map_file: str, contour: float, neural_filename: str) -> None:
    if not os.path.exists(neural_filename):
        print(f"{datetime.now()} - Training neural representation for {map_file}...")
        neural_map = NeuralDensityMap.from_voxel_map(
            map_file, contour, devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"])
        print(f"{datetime.now()} - Completed training. Saving to {neural_filename}")
        neural_map.save(neural_filename)
        recreate_filename = f"{os.path.dirname(neural_filename)}/{neural_map.map_id}_neural.mrc"
        print(f"{datetime.now()} - Re-creating voxels using neural rep. Saving to {recreate_filename}")
        neural_map.save_as_voxel_map(
            recreate_filename, devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"])


def create_graph(neural_filename: str, resolution: float, graph_filename: str,
                 seed_threshold: float = None) -> None:
    if not os.path.exists(graph_filename):
        neural_map = NeuralDensityMap.load(neural_filename)
        print(f"{datetime.now()} - Creating graph for {neural_map.map_id}...")
        adjacency_threshold = resolution * 2
        min_nodes = 0
        start_ns = time.time_ns()
        graph = DensityGraph.from_neural_density_map(
            neural_map,
            seed_threshold=seed_threshold,
            adjacency_threshold=adjacency_threshold,
            min_number_nodes=min_nodes,
            devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"])
        end_ns = time.time_ns()
        print(f"{datetime.now()} - Completed graph. Saving to {graph_filename}")
        map_id = graph.map_id
        graph.save(graph_filename)
        creation_params = {
            "map_id": graph.map_id,
            "adjacency": adjacency_threshold,
            "time": (end_ns - start_ns) / 1000000000,
            "resolution": resolution,
        }
        with open(f"{os.path.dirname(graph_filename)}/{map_id}_graph_params.json", 'w') as params:
            json.dump(creation_params, params, indent=2)
        graph_viz_filename = f"{os.path.dirname(graph_filename)}/{map_id}_graph.cif"
        print(f"{datetime.now()} - Creating graph visualization. Saving to {graph_viz_filename}")
        graph.save_as_mmcif(graph_viz_filename)


def perform_deeptracer_prediction(map_dir: str, deeptracer_pred_filename: str) -> None:
    if not os.path.exists(deeptracer_pred_filename):
        print(f"{datetime.now()} - Running DeepTracer prediction on {map_dir}.")
        logger = Logger('DeepTracer', log_level=LoggingType.ERROR)
        tracer = DeepTracer(logger)
        density_map = InputFiles.read_density_map(map_dir)
        seq = InputFiles.read_amino_acid_seq(map_dir)
        protein_structure = InputFiles.read_solved_structure(map_dir)
        predicted_structure, pred_eval = tracer.trace(density_map, seq, protein_structure)
        print(f"{datetime.now()} - Saving predicted structure to {deeptracer_pred_filename}.")
        predicted_structure.save(deeptracer_pred_filename)
        with open(f"{os.path.dirname(deeptracer_pred_filename)}/deeptracer_eval.json", 'w') as deeptracer_eval:
            json.dump(pred_eval, deeptracer_eval, indent=2)


def find_close_residues(predicted_structure: ProteinStructure,
                        solved_structure: ProteinStructure) -> List[Tuple[AminoAcid, AminoAcid]]:
    # Sort predicted residues by their x coordinates to filter number of residues
    # that can be within 3A of a native residue
    pred_residues = sorted(predicted_structure.amino_acids, key=lambda r: r.carbon_alpha.x)
    native_residues = solved_structure.amino_acids

    # Sorted x coordinates of predicted residues used for filtering below
    pred_x_coordinates = [r.carbon_alpha.x for r in pred_residues]

    # Maps native residues to predicted residues within 3A distance
    native_to_pred = [[] for _ in range(len(native_residues))]
    # Maps predicted residues to native residues within 3A distance
    pred_to_native = [[] for _ in range(len(pred_residues))]

    for i, native_residue in enumerate(native_residues):
        # Find starting index where the x coordinates is at most 3A off. If it is
        # more off this residue cannot be within 3A distance
        j = bisect_left(pred_x_coordinates, native_residue.carbon_alpha.x - 3)

        # Loop until x coordinate of residue is further than 3A off
        while j < len(pred_residues) and pred_residues[j].carbon_alpha.x < native_residue.carbon_alpha.x + 3:
            distance = native_residue.distance_to(pred_residues[j])

            if distance <= 3:
                heapq.heappush(native_to_pred[i], (distance, j))
                heapq.heappush(pred_to_native[j], (distance, i))

            j += 1
    # Make sure that each predicted residue is mapped to only one native residue
    for pred_residue, close_native_residues in enumerate(pred_to_native):
        for _, native_residue in close_native_residues[1:]:
            native_to_pred[native_residue] = [e for e in native_to_pred[native_residue] if e[1] != pred_residue]
    return native_to_pred


def match_atoms(predicted_atoms: List[Atom], native_atoms: List[Atom]) -> List[Tuple[Atom, Atom]]:
    # Sort predicted atoms by their x coordinates to filter number of atoms
    # that can be within 3A of a native atom
    pred_atoms = sorted(predicted_atoms, key=lambda a: a.x)

    # Sorted x coordinates of predicted atoms used for filtering below
    pred_x_coordinates = [a.x for a in pred_atoms]

    # Maps native residues to predicted residues within 3A distance
    native_to_pred = [[] for _ in range(len(native_atoms))]
    # Maps predicted residues to native residues within 3A distance
    pred_to_native = [[] for _ in range(len(pred_atoms))]

    for i, native_atom in enumerate(native_atoms):
        # Find starting index where the x coordinates is at most 3A off. If it is
        # more off this residue cannot be within 3A distance
        j = bisect_left(pred_x_coordinates, native_atom.x - 1)

        # Loop until x coordinate of residue is further than 3A off
        while j < len(pred_atoms) and pred_atoms[j].x < native_atom.x + 1:
            distance = native_atom.distance_to(pred_atoms[j])

            if distance <= 1:
                heapq.heappush(native_to_pred[i], (distance, j))
                heapq.heappush(pred_to_native[j], (distance, i))

            j += 1

    # Make sure that each predicted residue is mapped to only one native residue
    for pred_atom, close_native_atoms in enumerate(pred_to_native):
        for _, native_atom in close_native_atoms[1:]:
            native_to_pred[native_atom] = [e for e in native_to_pred[native_atom] if e[1] != pred_atom]

    # Make sure that each native residue is mapped to only one predicted residue
    return [(native_atoms[r], pred_atoms[m[0][1]]) for r, m in enumerate(native_to_pred) if len(m) > 0]


def atomic_rmsd(matching_atoms: List[Tuple[Atom, Atom]]) -> float:
    if len(matching_atoms) == 0:
        return -1
    else:
        return sqrt(sum(a1.squared_distance_to(a2) for a1, a2 in matching_atoms) / len(matching_atoms))


def evaluate(neural_filename: str,
             graph_filename: str,
             deeptracer_pred_filename: str,
             pdb_file: str,
             resolution: float,
             eval_filename: str):
    # Get neural map metrics
    neural_map = NeuralDensityMap.load(neural_filename)
    neural_stats = dict()
    neural_stats["shape"] = neural_map.voxel_shape
    neural_stats["regions"] = list()
    for neural_region in neural_map._sub_regions.keys():
        neural_stats["regions"].append({
            "loss": neural_region.loss,
            "training_time": neural_region.training_time,
            "save_size": os.path.getsize(f"{os.path.dirname(neural_filename)}/{neural_region.name}.pt"),
        })
    num_regions = len(neural_stats["regions"])

    # Get graph metrics
    graph = DensityGraph.load(graph_filename)
    pdb_structure = ProteinStructure.open(pdb_file)
    aas = [AminoAcid(Atom(n.x, n.y, n.z)) for n in graph.nodes]
    graph_structure = ProteinStructure(graph.map_id, [Model([Chain(aas)])])
    close = find_close_residues(graph_structure, pdb_structure)
    num_close = len([i for c in close for i in c])
    matching = find_matching_residues(graph_structure, pdb_structure)

    if resolution < 2.0:
        graph_atoms = [Atom(n.x, n.y, n.z) for n in graph.nodes]
        atomic_matches = match_atoms(graph_atoms, pdb_structure.atoms)
        percent_atom_w_node = len(atomic_matches) / len(pdb_structure.atoms) * 100
        percent_node_w_atom = len(atomic_matches) / len(graph_atoms) * 100
        atom_rmsd = atomic_rmsd(atomic_matches)
    else:
        percent_atom_w_node = -1
        percent_node_w_atom = -1
        atom_rmsd = -1

    # Get eval metrics
    with open(f"{os.path.dirname(deeptracer_pred_filename)}/deeptracer_eval.json", 'r') as f:
        deeptracer_eval = json.load(f)
    with open(f"{os.path.dirname(graph_filename)}/{graph.map_id}_graph_params.json", 'r') as f:
        graph_params = json.load(f)
    evaluation = {
        "map_id": graph.map_id,
        "pdb_id": deeptracer_eval["PDB"],
        "resolution": resolution,
        "Voxel count": neural_stats["shape"][0] * neural_stats["shape"][1] * neural_stats["shape"][2],
        "Neural region count": num_regions,
        "Total training time": sum([r["training_time"] for r in neural_stats["regions"]]),
        "Average region training time": sum([r["training_time"] for r in neural_stats["regions"]]) / num_regions,
        "Average loss": sum([r["loss"] for r in neural_stats["regions"]]) / num_regions,
        "Neural save size": sum([r["save_size"] for r in neural_stats["regions"]]) + os.path.getsize(neural_filename),
        "Native save size": os.path.getsize(f"{os.path.dirname(neural_filename)}/{neural_map.map_id}_neural.mrc"),
        "Graph creation time": graph_params["time"],
        "Graph save size": os.path.getsize(graph_filename),
        "Num nodes": len(graph.nodes),
        "Num C-a": len(pdb_structure.amino_acids),
        "% C-a with Node": matching_percentage(matching, pdb_structure),
        "% Node with C-a": num_close / len(graph_structure.amino_acids) * 100,
        "Graph RMSD": rmsd(matching),
        "DeepTracer % matching": deeptracer_eval["% Matching"],
        "DeepTracer RMSD": deeptracer_eval["RMSD"],
        "Num atoms": len(pdb_structure.atoms),
        "% Atom with node": percent_atom_w_node,
        "% Node with atom": percent_node_w_atom,
        "Graph Atomic RMSD": atom_rmsd,
    }

    # Add evaluation to cumulative evaluation for entire experiment
    if os.path.exists(eval_filename):
        with open(eval_filename, 'r') as f:
            cumulative_eval = json.load(f)
    else:
        cumulative_eval = list()
    cumulative_eval.append(evaluation)
    with open(eval_filename, 'w') as f:
        json.dump(cumulative_eval, f, indent=2)


def main():
    # Select the maps to train
    selected_data = select_data("/data/nranno/Neural-Training", 5, 512 ** 3)

    # Create neural maps, graphs, and DeepTracer predictions for each selected sample
    for data_dict in selected_data:
        try:
            # Get input info
            map_file = data_dict["map_file"]
            resolution = data_dict["resolution"]
            contour = data_dict["contour"]
            pdb_file = data_dict["pdb_file"]

            # Create save location
            map_id = os.path.splitext(os.path.split(map_file)[1])[0]
            save_dir = f"./experiments/graph_creation/{map_id}"
            os.makedirs(save_dir, exist_ok=True)

            # Create or get the neural density map
            neural_filename = f"{save_dir}/{map_id}.neural.gz"
            create_neural_map(map_file, contour, neural_filename)

            # Create or get the density map graph
            graph_filename = f"{save_dir}/{map_id}.graph.gz"
            create_graph(neural_filename, resolution, graph_filename)

            # Perform or get the DeepTracer prediction
            deeptracer_pred_filename = f"{save_dir}/{map_id}.pdb"
            perform_deeptracer_prediction(os.path.dirname(map_file), deeptracer_pred_filename)

            # Evaluate the results and publish to file
            evaluation_file = "./experiments/graph_creation/evaluation.json"
            evaluate(neural_filename, graph_filename, deeptracer_pred_filename, pdb_file, resolution, evaluation_file)
        except Exception:
            print(f"{datetime.now()} - Exception: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
