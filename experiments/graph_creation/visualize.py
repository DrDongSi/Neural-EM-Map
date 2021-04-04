import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def neural_training_times(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, (g1, g2) = plt.subplots(1, 2)

    num_regions = [r["Neural region count"] for r in results]
    print("Number of neural regions:")
    print(f"  Avg: {np.average(num_regions)}")
    print(f"  Std: {np.std([num_regions])}")
    voxel_count = [r["Voxel count"] for r in results]
    total_time = [r["Total training time"] for r in results]
    average_time = [r["Average region training time"] for r in results]

    g1.set_title("Total Time")
    g1.plot(voxel_count, total_time, 'bo', fillstyle='none')
    g1.set_xlabel("Total voxel count")
    g1.set_ylabel("Time (seconds)")
    print("Total training time:")
    print(f"  Avg: {np.average(total_time)}")
    print(f"  Std: {np.std(total_time)}")

    g2.set_title("Average Time Per Region")
    g2.plot(voxel_count, average_time, 'bo', fillstyle='none')
    g2.set_xlabel("Total voxel count")
    g2.set_ylabel("Time (seconds)")
    print("Training time per region:")
    print(f"  Avg: {np.average(average_time)}")
    print(f"  Std: {np.std(average_time)}")

    fig.tight_layout()
    plt.show()


def neural_sizes(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, g1 = plt.subplots(1, 1)

    native_size = [r["Native save size"] / (1024 ** 2) for r in results]
    neural_size = [r["Neural save size"] / (1024 ** 2) for r in results]

    print("% Size Difference:")
    print(f"  Min: {np.min(abs(np.true_divide(neural_size, native_size) - 1))}")
    print(f"  Max: {np.max(abs(np.true_divide(neural_size, native_size) - 1))}")
    print(f"  Avg: {np.average(abs(np.true_divide(neural_size, native_size) - 1))}")

    size_fit_x = np.unique(native_size)
    size_fit_y = np.poly1d(np.polyfit(native_size, neural_size, 1))(np.unique(native_size))
    g1.plot(native_size, neural_size, 'go', fillstyle='none')
    g1.plot(size_fit_x, size_fit_y, color='0.3', linestyle='--', linewidth=1.25)
    g1_max = max(max(native_size), max(neural_size))
    g1_range = [0, g1_max + (0.04 * g1_max)]
    g1.set_xlim(g1_range)
    g1.set_ylim(g1_range)
    g1.plot(g1.get_xlim(), g1.get_ylim(), color='0.3', linestyle='-', linewidth=1.0)
    g1.set_xlabel("Native Size (MB)")
    g1.set_ylabel("Neural Save State Size (MB)")

    fig.tight_layout()
    plt.show()


def scatter_graph(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, (g1, g2, g3) = plt.subplots(1, 3)

    x = [float(r["resolution"]) for r in results]
    graph_rmsd = [r["Graph RMSD"] for r in results]
    graph_ca_match = [r["% C-a with Node"] for r in results]
    graph_node_match = [r["% Node with C-a"] for r in results]
    unique_res = sorted(list(set(x)))
    avg_graph_rmsd = list()
    avg_graph_ca_match = list()
    avg_graph_node_match = list()
    for res in unique_res:
        avg_graph_rmsd.append(np.average([r["Graph RMSD"] for r in results if float(r["resolution"]) == res]))
        avg_graph_ca_match.append(np.average([r["% C-a with Node"] for r in results if float(r["resolution"]) == res]))
        avg_graph_node_match.append(np.average([r["% Node with C-a"] for r in results if float(r["resolution"]) == res]))

    print("Average % Node Matching:")
    print(f"  Graph: {np.average(graph_node_match)}")

    g1.plot(x, graph_rmsd, 'bo', alpha=0.7, fillstyle='none')
    g1.plot(unique_res, avg_graph_rmsd, color='k', marker='.', linestyle='-', linewidth='1.0')
    g1.set_ylim(0)
    g1.set_xlabel("Resolution (Å)")
    g1.set_ylabel("RMSD")

    g2.plot(x, graph_ca_match, 'bo', alpha=0.7, fillstyle='none')
    g2.plot(unique_res, avg_graph_ca_match, color='k', marker='.', linestyle='-', linewidth='1.0')
    g2.set_ylim(0, 102)
    g2.set_xlabel("Resolution (Å)")
    g2.set_ylabel("% C-α Matching")

    g3.plot(x, graph_node_match, 'bo', alpha=0.7, fillstyle='none')
    g3.plot(unique_res, avg_graph_node_match, color='k', marker='.', linestyle='-', linewidth='1.0')
    g3.set_ylim(0, 102)
    g3.set_xlabel("Resolution (Å)")
    g3.set_ylabel("% Nodes Matching")

    # fig.tight_layout()
    plt.show()


def compare_graph_and_deeptracer(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, ((g1, g2), (g3, g4)) = plt.subplots(2, 2)

    deeptracer_rmsd = [r["DeepTracer RMSD"] for r in results]
    graph_rmsd = [r["Graph RMSD"] for r in results]
    g1_max = max(max(deeptracer_rmsd), max(graph_rmsd))
    g1_range = [0, g1_max + (0.05 * g1_max)]
    g1.plot(deeptracer_rmsd, graph_rmsd, 'go', fillstyle='none')
    g1.set_xlim(g1_range)
    g1.set_ylim(g1_range)
    g1.plot(g1.get_xlim(), g1.get_ylim(), color='0.3', linestyle='-', linewidth=1.0)
    g1.set_xlabel("DeepTracer RMSD")
    g1.set_ylabel("Graph RMSD")

    deeptracer_match = [r["DeepTracer % matching"] for r in results]
    graph_match = [r["% C-a with Node"] for r in results]
    g2.plot(deeptracer_match, graph_match, 'go', fillstyle='none')
    g2.set_xlim(0, 102)
    g2.set_ylim(0, 102)
    g2.plot(g2.get_xlim(), g2.get_ylim(), color='0.3', linestyle='-', linewidth=1.0)
    g2.set_xlabel("DeepTracer % Matching")
    g2.set_ylabel("Graph % Matching")

    print("Average RMSD:")
    print(f"  DeepTracer: {np.average(deeptracer_rmsd)}")
    print(f"  Graph: {np.average(graph_rmsd)}")
    print("Average % Matching:")
    print(f"  DeepTracer: {np.average(deeptracer_match)}")
    print(f"  Graph: {np.average(graph_match)}")

    res = [float(r["resolution"]) for r in results]
    unique_res = sorted(list(set(res)))
    avg_graph_rmsd = list()
    avg_deeptracer_rmsd = list()
    avg_graph_match = list()
    avg_deeptracer_match = list()
    for u in unique_res:
        avg_graph_rmsd.append(np.average([r["Graph RMSD"] for r in results if float(r["resolution"]) == u]))
        avg_deeptracer_rmsd.append(np.average([r["DeepTracer RMSD"] for r in results if float(r["resolution"]) == u]))
        avg_graph_match.append(np.average([r["% C-a with Node"] for r in results if float(r["resolution"]) == u]))
        avg_deeptracer_match.append(np.average([r["DeepTracer % matching"] for r in results if float(r["resolution"]) == u]))
    g3.plot(res, graph_rmsd, 'bo', alpha=0.55, fillstyle='none', label="Graph Nodes")
    g3.plot(unique_res, avg_graph_rmsd, color='b', marker='.', linestyle='-', linewidth=1.0)
    g3.plot(res, deeptracer_rmsd, 'ro', alpha=0.55, fillstyle='none', label="DeepTracer C-α")
    g3.plot(unique_res, avg_deeptracer_rmsd, color='r', marker='.', linestyle='-', linewidth=1.0)
    g3.set_ylim(0)
    g3.set_xlabel("Resolution (Å)")
    g3.set_ylabel("RMSD")
    g3.legend(loc='upper left')

    g4.plot(res, graph_match, 'bo', alpha=0.55, fillstyle='none', label="Graph Nodes")
    g4.plot(unique_res, avg_graph_match, color='b', marker='.', linestyle='-', linewidth=1.0)
    g4.plot(res, deeptracer_match, 'ro', alpha=0.55, fillstyle='none', label="DeepTracer C-α")
    g4.plot(unique_res, avg_deeptracer_match, color='r', marker='.', linestyle='-', linewidth=1.0)
    g4.set_xlabel("Resolution (Å)")
    g4.set_ylim(0, 102)
    g4.set_ylabel("% C-α Matching")
    g4.legend(loc='lower left')

    # fig.tight_layout()
    plt.show()


def compare_ultra_high_res(results_file: str):
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, ((g1, g2), (g3, g4)) = plt.subplots(2, 2)

    resolution_threshold = 1.6
    x = [float(r["resolution"]) for r in results if float(r["resolution"]) <= resolution_threshold]
    graph_rmsd = [r["Graph RMSD"] for r in results if float(r["resolution"]) <= resolution_threshold]
    deeptracer_rmsd = [r["DeepTracer RMSD"] for r in results if float(r["resolution"]) <= resolution_threshold]
    graph_ca_match = [r["% C-a with Node"] for r in results if float(r["resolution"]) <= resolution_threshold]
    deeptracer_ca_match = [r["DeepTracer % matching"] for r in results if float(r["resolution"]) <= resolution_threshold]
    graph_node_match = [r["% Node with C-a"] for r in results if float(r["resolution"]) <= resolution_threshold]
    graph_atomic_match = [r["% Atom with node"] for r in results if float(r["resolution"]) <= resolution_threshold]
    graph_node_match_atomic = [r["% Node with atom"] for r in results if float(r["resolution"]) <= resolution_threshold]
    atomic_rmsd = [r["Graph Atomic RMSD"] for r in results if float(r["resolution"]) <= resolution_threshold]
    emdb_ids = [r["map_id"] for r in results if float(r["resolution"]) <= resolution_threshold]
    pdb_ids = [r["pdb_id"] for r in results if float(r["resolution"]) <= resolution_threshold]

    print("Ultra-High Resolution Average RMSD:")
    print(f"  DeepTracer: {np.average(deeptracer_rmsd)}")
    print(f"  Graph: {np.average(graph_rmsd)}")
    print("Ultra-High Resolution Average % Matching:")
    print(f"  DeepTracer: {np.average(deeptracer_ca_match)}")
    print(f"  Graph: {np.average(graph_ca_match)}")
    print("Ultra-High Resolution Average % Node Matching:")
    print(f"  Graph: {np.average(graph_node_match)}")
    print("Ultra-High Resolution Average Atomic RMSD:")
    print(f"  Graph: {np.average(atomic_rmsd)}")
    print("Ultra-High Resolution Average % Atom Matching:")
    print(f"  Graph: {np.average(graph_atomic_match)}")
    print("Ultra-High Resolution Average % Node Matching Atoms:")
    print(f"  Graph: {np.average(graph_node_match_atomic)}")
    print("|  EMDB ID  | PDB ID |      RMSD      | % Atoms Matching | % Nodes Matching |")
    print("|-----------|--------|----------------|------------------|------------------|")
    for i in np.argsort(x):
        print(f"|{emdb_ids[i]:^11}|{pdb_ids[i]:^8}|{atomic_rmsd[i]:^16.10f}|{graph_atomic_match[i]:^18.10f}|{graph_node_match_atomic[i]:^18.10f}|")

    g1.plot(x, graph_rmsd, 'bo', fillstyle='none', label="Graph method")
    g1.plot(x, deeptracer_rmsd, 'ro', fillstyle='none', label="DeepTracer")
    g1.set_ylim(0)
    g1.set_xlabel("Resolution (Å)")
    g1.set_ylabel("RMSD")
    g1.legend(loc='lower right')

    g2.plot(x, graph_ca_match, 'bo', fillstyle='none', label="Graph method")
    g2.plot(x, deeptracer_ca_match, 'ro', fillstyle='none', label="DeepTracer")
    g2.set_ylim(0, 102)
    g2.set_xlabel("Resolution (Å)")
    g2.set_ylabel("% C-α matching")
    g2.legend(loc='lower right')

    g3.plot(x, atomic_rmsd, 'bo', fillstyle='none')
    g3.set_ylim(0)
    g3.set_xlabel("Resolution (Å)")
    g3.set_ylabel("RMSD")

    g4.plot(x, graph_atomic_match, 'bo', fillstyle='none', label="% Atoms Matched")
    g4.plot(x, graph_node_match_atomic, 'ro', fillstyle='none', label="% Nodes Matched")
    g4.set_ylim(0, 102)
    g4.set_xlabel("Resolution (Å)")
    g4.set_ylabel("% matching")
    g4.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


def compare_high_res(results_file: str):
    with open(results_file, 'r') as f:
        results = json.load(f)

    resolution_threshold = 1.6
    graph_rmsd = [r["Graph RMSD"] for r in results if float(r["resolution"]) > resolution_threshold]
    deeptracer_rmsd = [r["DeepTracer RMSD"] for r in results if float(r["resolution"]) > resolution_threshold]
    graph_ca_match = [r["% C-a with Node"] for r in results if float(r["resolution"]) > resolution_threshold]
    deeptracer_ca_match = [r["DeepTracer % matching"] for r in results if float(r["resolution"]) > resolution_threshold]
    graph_node_match = [r["% Node with C-a"] for r in results if float(r["resolution"]) > resolution_threshold]

    print("Only High Resolution Average RMSD:")
    print(f"  DeepTracer: {np.average(deeptracer_rmsd)}")
    print(f"  Graph: {np.average(graph_rmsd)}")
    print("Only High Resolution Average % Matching:")
    print(f"  DeepTracer: {np.average(deeptracer_ca_match)}")
    print(f"  Graph: {np.average(graph_ca_match)}")
    print("Only High Resolution Average % Node Matching:")
    print(f"  Graph: {np.average(graph_node_match)}")


def to_csv(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Residue level eval and DeepTracer comparison
    csv_file = os.path.join(os.path.dirname(results_file), "graph_residue_comparison.csv")
    csv_fields = ["EMDB ID", "PDB ID", "Resolution",
                  "Num. C-a", "Num. nodes",
                  "% C-a with Node", "% Node with C-a", "Graph RMSD",
                  "DeepTracer % matching", "DeepTracer RMSD"]
    csv_rows = [
        {
            "EMDB ID": r["map_id"],
            "PDB ID": r["pdb_id"],
            "Resolution": r["resolution"],
            "Num. C-a": r["Num C-a"],
            "Num. nodes": r["Num nodes"],
            "% C-a with Node": r["% C-a with Node"],
            "% Node with C-a": r["% Node with C-a"],
            "Graph RMSD": r["Graph RMSD"],
            "DeepTracer % matching": r["DeepTracer % matching"],
            "DeepTracer RMSD": r["DeepTracer RMSD"],
        } for r in results]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Atomic level evaluation
    csv_file = os.path.join(os.path.dirname(results_file), "graph_atomic_comparison.csv")
    csv_fields = ["EMDB ID", "PDB ID", "Resolution",
                  "Num. atoms", "% Atom with node", "% Node with atom", "Graph Atomic RMSD"]
    csv_rows = [
        {
            "EMDB ID": r["map_id"],
            "PDB ID": r["pdb_id"],
            "Resolution": r["resolution"],
            "Num. atoms": r["Num atoms"],
            "% Atom with node": r["% Atom with node"],
            "% Node with atom": r["% Node with atom"],
            "Graph Atomic RMSD": r["Graph Atomic RMSD"],
        } for r in results if float(r["resolution"]) <= 1.6]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


def main():
    results_file = "./experiments/graph_creation/results/evaluation.json"

    neural_training_times(results_file)
    neural_sizes(results_file)

    scatter_graph(results_file)
    compare_graph_and_deeptracer(results_file)
    compare_ultra_high_res(results_file)
    compare_high_res(results_file)

    to_csv(results_file)


if __name__ == "__main__":
    main()
