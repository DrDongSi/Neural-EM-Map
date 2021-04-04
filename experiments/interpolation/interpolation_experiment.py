from copy import deepcopy
from datetime import datetime
import json
import mrcfile
import numpy as np
import os
import shutil

from deeptracer.common.chimera import Chimera
from deeptracer.common.logging import Logger, LoggingType
from neural_dataset import select_data
from neural_density_map import NeuralDensityMap


def create_maps(pdb_file: str,
                resolution: float) -> None:
    pdb_id = os.path.basename(pdb_file)[0:4]
    output_dir = os.path.join("./experiments/interpolation", pdb_id)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(pdb_file, os.path.join(output_dir, f"{pdb_id}.pdb"))
    simulated_output = os.path.join(output_dir, f"{pdb_id}.mrc")
    base_resample = os.path.join(output_dir, f"{pdb_id}_resample.mrc")
    linear_resample = os.path.join(output_dir, f"{pdb_id}_linear_resample.mrc")
    neural_resample = os.path.join(output_dir, f"{pdb_id}_neural_resample.mrc")
    logger = Logger("Chimera-operations", LoggingType.WARNING)
    grid_spacing = 0.2

    # Create simulated map
    if not os.path.exists(simulated_output):
        print(f"{datetime.now()} - Creating simulated map for {pdb_id}")
        Chimera.run(logger, ".", [
            "volume showPlane false",
            f"open {pdb_file}",
            f"molmap #0@C,N,O {resolution}",
            f"volume # save {simulated_output}"])
        with mrcfile.open(simulated_output, 'r+') as mrc:
            ox = float(mrc.header.origin.x)
            oy = float(mrc.header.origin.x)
            oz = float(mrc.header.origin.x)
            max_val = mrc.data.max()
            min_val = mrc.data.min()
            mrc.data[:] = (mrc.data - min_val) / (max_val - min_val)
            mrc.update_header_stats()

    # Base resampling
    if not os.path.exists(base_resample):
        print(f"{datetime.now()} - Creating base interpolation map for {pdb_id}")
        Chimera.run(logger, ".", [
            "volume showPlane false",
            f"open {pdb_file}",
            f"molmap #0@C,N,O {resolution} gridSpacing {grid_spacing}",
            f"volume # save {base_resample}"])
        with mrcfile.open(base_resample, 'r+') as mrc:
            nx = int(mrc.header.nx)
            ny = int(mrc.header.ny)
            nz = int(mrc.header.nz)
            mrc.data[:] = (mrc.data - min_val) / (max_val - min_val)
            mrc.update_header_stats()

    # Linear resampling
    if not os.path.exists(linear_resample):
        print(f"{datetime.now()} - Creating linear interpolated map for {pdb_id}")
        Chimera.run(logger, ".", [
            "volume showPlane false",
            f"open {simulated_output}",
            f"vop new grid gridSpacing {grid_spacing} size {nx},{ny},{nz} origin {ox},{oy},{oz}",
            "vop resample #0 onGrid #1",
            f"volume #2 save {linear_resample}"])

    # Neural resampling
    neural_map_filename = os.path.join(output_dir, f"{pdb_id}_neural", f"{pdb_id}.neural")
    if not os.path.exists(neural_map_filename):
        device_list = ["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"]
        print(f"{datetime.now()} - Training neural density map for {pdb_id}")
        neural_map = NeuralDensityMap.from_voxel_map(simulated_output, devices=device_list)
        neural_map.save(neural_map_filename)
    if not os.path.exists(neural_resample):
        print(f"{datetime.now()} - Creating neural interpolation map for {pdb_id}")
        neural_map.save_as_voxel_map(
            neural_resample,
            (grid_spacing, grid_spacing, grid_spacing),
            device_list)


def evaluate_interpolation(pdb_id: str,
                           resolution: float,
                           eval_file: str) -> None:
    results_dir = f"./experiments/interpolation/{pdb_id}"
    with mrcfile.open(os.path.join(results_dir, f"{pdb_id}_resample.mrc")) as mrc:
        base_mrc = deepcopy(mrc.data)
    with mrcfile.open(os.path.join(results_dir, f"{pdb_id}_linear_resample.mrc")) as mrc:
        linear_mrc = deepcopy(mrc.data)
    with mrcfile.open(os.path.join(results_dir, f"{pdb_id}_neural_resample.mrc")) as mrc:
        neural_mrc = deepcopy(mrc.data)

    # # Pad neural map
    # lol = np.zeros(base_mrc.shape)
    # lol[0:neural_mrc.shape[0], 0:neural_mrc.shape[1], 0:neural_mrc.shape[2]] = neural_mrc
    # neural_mrc = lol

    comparison_indices = np.nonzero(base_mrc)
    linear_diffs = base_mrc[comparison_indices] - linear_mrc[comparison_indices]
    neural_diffs = base_mrc[comparison_indices] - neural_mrc[comparison_indices]

    min_linear = linear_diffs.min()
    max_linear = linear_diffs.max()
    avg_linear = np.mean(linear_diffs)
    mae_linear = np.mean(np.abs(linear_diffs))

    min_neural = neural_diffs.min()
    max_neural = neural_diffs.max()
    avg_neural = np.mean(neural_diffs)
    mae_neural = np.mean(np.abs(neural_diffs))

    eval_filename = os.path.join("./experiments/interpolation/", eval_file)
    if os.path.exists(eval_filename):
        with open(eval_filename, 'r') as f:
            evals = json.load(f)
    else:
        evals = list()

    evals.append({
        "pdb_id": pdb_id,
        "resolution": resolution,
        "linear": {
            "min": float(min_linear),
            "max": float(max_linear),
            "avg": float(avg_linear),
            "mae": float(mae_linear),
        },
        "neural": {
            "min": float(min_neural),
            "max": float(max_neural),
            "avg": float(avg_neural),
            "mae": float(mae_neural),
        }
    })

    with open(eval_filename, 'w') as f:
        json.dump(evals, f, indent=2)


def main():
    # Select the maps to interpolate
    selected_data = select_data("/data/nranno/Neural-Training", 5, 512 ** 3)

    # For each map in dataset, get the resolution and deposited structure
    for data_dict in selected_data:
        pdb_file = data_dict["pdb_file"]
        resolution = data_dict["resolution"]
        pdb_id = os.path.basename(pdb_file)[0:4]
        create_maps(pdb_file, resolution)
        evaluate_interpolation(pdb_id, resolution, "evaluation.json")


if __name__ == "__main__":
    main()
