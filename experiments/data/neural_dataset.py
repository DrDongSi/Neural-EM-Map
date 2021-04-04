from collections import defaultdict
from copy import deepcopy
import csv
import glob
import json
import os
import random
import requests
from typing import Dict, List

import mrcfile

from deeptracer.common.logging import Logger
from deeptracer.ml.dataset import download_experimental_data


def _rest_call(url: str) -> Dict:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed for {url} (Status code: {response.status_code})")
    return response.json()


def download_dataset(emdr_search_results_csv_file: str, download_location: str) -> None:
    # Open EMDB search results
    with open(emdr_search_results_csv_file, 'r',  encoding='utf-8-sig') as f:
        search_results = [{k: v for k, v in row.items()} for row in csv.DictReader(f)]

    # Filter for maps with single fitted PDB entry
    fitted = [s for s in search_results if s["Associated PDB"] != "" and " " not in s["Associated PDB"]]

    # Filter for unique titles, selecting the highest resolution of the duplicates
    filtered = list()
    titles = defaultdict(list)
    for f in fitted:
        titles[f["Citation Title"]].append(f)
    for t in titles.values():
        if len(t) > 1:
            sorted_by_res = sorted(t, key=lambda k: float(k["Resolution (Å)"]))
            filtered.append(sorted_by_res[0])
        else:
            filtered.append(t[0])

    # Download the dataset
    logger = Logger("Dataset_Downloader")
    download_experimental_data(logger, filtered, download_location)

    # Populate metadata file for downloaded maps
    map_files = glob.glob(f"{download_location}/*/*.map")
    for map_file in map_files:
        map_id = os.path.splitext(os.path.split(map_file)[1])[0][4:]
        meta_file = os.path.join(os.path.split(map_file)[0], "metadata.json")
        if not os.path.exists(meta_file):
            emdb_id = f"EMD-{map_id}"
            res_url = f"https://www.ebi.ac.uk/pdbe/api/emdb/entry/processing/{emdb_id}"
            contour_url = f"https://www.ebi.ac.uk/pdbe/api/emdb/entry/map/{emdb_id}"
            resolution = str(_rest_call(res_url)[emdb_id][0]["processing"]["reconstruction"]["resolution_by_author"])
            contour = str(_rest_call(contour_url)[emdb_id][0]["map"]["contour_level"]["value"])
            metadata = {
                "resolution": resolution,
                "contour": contour,
            }
            with open(meta_file, 'w') as meta:
                json.dump(metadata, meta)


def select_data(dataset_location: str, max_num_per_resolution: int, max_map_size: int) -> List[Dict]:
    # Organize maps and corresponding pdb files by resolution, rounded to the nearest tenth Angstrom
    map_files = glob.glob(f"{dataset_location}/*/*.map")
    maps_by_resolution = defaultdict(list)
    for map_file in map_files:
        meta_file = os.path.join(os.path.split(map_file)[0], "metadata.json")
        with open(meta_file, 'r') as meta:
            metadata = json.load(meta)
        # Check for existence of pdb file, do not select if missing
        if len(glob.glob(os.path.join(os.path.split(map_file)[0], "*.pdb"))) != 1:
            continue
        pdb_file = glob.glob(os.path.join(os.path.split(map_file)[0], "*.pdb"))[0]
        resolution = round(float(metadata["resolution"]), 1)
        contour = float(metadata["contour"])
        data_dict = {
            "map_file": map_file,
            "pdb_file": pdb_file,
            "resolution": resolution,
            "contour": contour
        }
        maps_by_resolution[str(resolution)].append(data_dict)

    # Randomly select up to given amount per resolution
    random.seed(45067)
    selected_data = list()
    for _, v in maps_by_resolution.items():
        samples = deepcopy(v)
        num_selected = 0
        while num_selected < max_num_per_resolution:
            if len(samples) == 0:
                break
            selected = random.choice(samples)
            with mrcfile.open(selected["map_file"], header_only=True) as mrc:
                map_size = mrc.header.nx * mrc.header.ny * mrc.header.nz
            # Do not consider maps above the maximum size
            if map_size > max_map_size:
                samples.remove(selected)
            else:
                selected_data.append(selected)
                samples.remove(selected)
                num_selected += 1
    return selected_data


if __name__ == "__main__":
    download_dataset("./EMDRSearch_HighRes.csv", "/data/nranno/Neural-Training")
