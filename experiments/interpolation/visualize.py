import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def dataset_evaluation(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, g = plt.subplots(1, 1)

    bars = dict()
    for res in np.arange(10, 41, 1):
        bars[str(res / 10)] = len([r for r in results if float(r["resolution"]) == res / 10])

    g.bar(bars.keys(), bars.values(), tick_label=[b if (".0" in b) or (".5" in b) else "" for b in bars.keys()])
    g.set_xlabel("Resolution (Å)")
    g.set_ylabel("Num. Entries")

    fig.tight_layout()
    plt.show()


def error_vs_resolution(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, g1 = plt.subplots(1, 1)

    x = [float(r["resolution"]) for r in results]
    linear_mae = [r["linear"]["mae"] for r in results]
    neural_mae = [r["neural"]["mae"] for r in results]
    linear_avg = [r["linear"]["avg"] for r in results]
    neural_avg = [r["neural"]["avg"] for r in results]
    unique_res = sorted(list(set(x)))
    avg_linear_mae = list()
    avg_neural_mae = list()
    avg_linear_avg = list()
    avg_neural_avg = list()
    for res in unique_res:
        avg_linear_mae.append(np.average([r["linear"]["mae"] for r in results if float(r["resolution"]) == res]))
        avg_neural_mae.append(np.average([r["neural"]["mae"] for r in results if float(r["resolution"]) == res]))
        avg_linear_avg.append(np.average([r["linear"]["avg"] for r in results if float(r["resolution"]) == res]))
        avg_neural_avg.append(np.average([r["neural"]["avg"] for r in results if float(r["resolution"]) == res]))

    print("Average MAE:")
    print(f"  Linear: {np.average(linear_mae)}")
    print(f"  Neural: {np.average(neural_mae)}")
    print("Std. Deviation MAE:")
    print(f"  Linear: {np.std(linear_mae)}")
    print(f"  Neural: {np.std(neural_mae)}")

    g1.plot(x, linear_mae, 'ro', fillstyle='none', label="Linear Interpolation")
    linear_mae_fit_x = np.unique(x)
    linear_mae_fit_y = np.poly1d(np.polyfit(x, linear_mae, 1))(np.unique(x))
    g1.plot(linear_mae_fit_x, linear_mae_fit_y, color='0.3', linestyle='--', linewidth=1.25)
    g1.plot(x, neural_mae, 'bo', fillstyle='none', label="Neural Interpolation")
    neural_mae_fit_x = np.unique(x)
    neural_mae_fit_y = np.poly1d(np.polyfit(x, neural_mae, 1))(np.unique(x))
    g1.plot(neural_mae_fit_x, neural_mae_fit_y, color='0.3', linestyle='--', linewidth=1.25)
    g1.set_xlabel("Resolution (Å)")
    g1.set_ylabel("Mean Absolute Error")
    g1.legend()

    fig.tight_layout()
    plt.show()


def linear_vs_neural_error(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)
    fig, (g1, g2) = plt.subplots(1, 2)
    fig.suptitle("Linear Error vs Neural Error")

    linear_mae = [r["linear"]["mae"] for r in results]
    neural_mae = [r["neural"]["mae"] for r in results]
    linear_avg = [r["linear"]["avg"] for r in results]
    neural_avg = [r["neural"]["avg"] for r in results]

    g1.plot(linear_mae, neural_mae, 'go', fillstyle='none')
    g1_max = max(max(linear_mae), max(neural_mae))
    g1_range = [0, g1_max + (0.05 * g1_max)]
    g1.plot(g1_range, g1_range, color='0.3', linestyle='-', linewidth=1.0)
    g1.set_xlim(g1_range)
    g1.set_ylim(g1_range)
    g1.set_xlabel("Linear MAE")
    g1.set_ylabel("Neural MAE")

    g2.plot(linear_avg, neural_avg, 'go', fillstyle='none')
    g2_min = min(min(linear_avg), min(neural_avg))
    g2_max = max(max(linear_avg), max(neural_avg))
    g2_range = [g2_min - (0.05 * g2_min), g2_max + (0.05 * g2_max)]
    g2.plot(g2_range, g2_range, color='0.3', linestyle='-', linewidth=1.0)
    g2.set_xlim(g2_range)
    g2.set_ylim(g2_range)
    g2.set_xlabel("Linear Mean Error")
    g2.set_ylabel("Neural Mean Error")

    fig.tight_layout()
    plt.show()


def to_csv(results_file: str) -> None:
    with open(results_file, 'r') as f:
        results = json.load(f)

    csv_file = os.path.splitext(results_file)[0] + ".csv"
    csv_fields = ["PDB ID", "Resolution",
                  "Linear Min. Error", "Linear Max. Error", "Linear MAE",
                  "Neural Min. Error", "Neural Max. Error", "Neural MAE"]
    csv_rows = [
        {
            "PDB ID": r["pdb_id"],
            "Resolution": r["resolution"],
            "Linear Min. Error": r["linear"]["min"],
            "Linear Max. Error": r["linear"]["max"],
            "Linear MAE": r["linear"]["mae"],
            "Neural Min. Error": r["neural"]["min"],
            "Neural Max. Error": r["neural"]["max"],
            "Neural MAE": r["neural"]["mae"],
        } for r in results]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


def main():
    results_file = "./experiments/interpolation/interpolation_evaluation.json"

    # Dataset distribution
    dataset_evaluation(results_file)

    # Error of linear and neural vs resolution
    error_vs_resolution(results_file)

    # Error of linear (x-axis) vs error of neural (y-axis)
    linear_vs_neural_error(results_file)

    to_csv(results_file)


if __name__ == "__main__":
    main()
