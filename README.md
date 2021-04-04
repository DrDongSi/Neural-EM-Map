# Neural-EM-Map

Neural Representations of Cryo-EM Maps and a Graph-Based Interpretation
Preprint: 

## Repository Contents

The two main modules for creating and using the neural cryo-EM maps and density
graphs are the files `neural_density_map.py` and `density_graph.py`
respectively. These modules, as well as the classes and functions within them,
are fully documented with docstrings that describe the purpose and arguments
for each. The `siren.py` module contains the PyTorch implementation of the
SIREN neural network that is used by the neural cryo-EM map.

The `experiments` folder contains the scripts used to perform the experiments
and produce the visualizations described in the published work. See the
Experiments section below for instructions on how to run the code.

## Executing Code

### Prerequisites

Python version >=3.7 is required for the code in this repository.

This document is assuming that CUDA 10.1 is used to enable GPU usage for
PyTorch and Tensorflow.

The neural cryo-EM maps and density graphs may be created without depending
on DeepTracer, but the experiment code leverages existing code in the
DeepTracer backend project for dataset creation and evaluation. Find the
DeepTracer project [here](https://github.com/DrDongSi/Deep-Tracer-1.0). Follow
the instructions there to install the project, preferably in the same Conda
environment provided with this repository.

The file `environment.yaml` is the exported Conda environment used in the
execution and development of the code in this repository, and it may be used
to replicate that environment by running

```bash
conda env create -f environment.yml
```

It contains packages for running DeepTracer as well as the neural cryo-EM map
and density graph. For just the dependencies of the neural map and density
graph, see the `requirements.txt` file.

### Neural Cryo-EM Maps

To use the NeuralDensityMap class, import the `neural_density_map.py` module
into your code.

The NeuralDensityMap class encapsulates both the creation of the neural
representation for a voxel density map as a whole and the access of the
density values and gradient vectors at any point in the map. During the
construction of the NeuralDensityMap, the composite NeuralDensityRegions
are patched together such that each point in the map region is assigned
only one NeuralDensityRegion. The NeuralDensityMap can be constructed fully
from the input NeuralDensityRegions, but the metadata of the original
voxel data may be used in construction in order to allow for the data
mining and reproduction of a 3DEM map. The normal creation method involves
the static `from_voxel_map()` or `from_voxel_data()` functions. E.g.

```python
neural_map = NeuralDensityMap.from_voxel_map("./emd_6272.map")
```

The neural representation can be saved to prevent the need for re-training
of the neural regions by using the `save()` method:

```python
neural_map.save("./emd_6272.neural")
```

To visualize the NeuralDensityMap, a voxel density map can be created
using the `save_as_voxel_map()` function. This function is also useful
for resamping a density map with a different voxel size. For example,

```python
neural_map.save_as_voxel_map("./emd_6272_resampled.map",
                             (0.5, 0.5, 0.5))
```

For querying values in the map, the methods `get_xxx()` are for arbitrary
points within the map. For uniform sampling of density values and
gradients, use the `sample_xxx()` methods as they are optimized for whole
map querying.

### Density Graphs

To use the DensityGraph class, import the `density_graph.py` module into your
code.

The graph places nodes at locally dense points, detected by the following
method. Points are sampled across the entire map, iteratively walked along
their gradient to density peaks, and then clustered. Candidate nodes are
placed at the centroid locations for each cluster. Nodes are filtered based
on adjacency and sub-graph constraints and then represented by a undirected
Graph from the NetworkX library.

To create the DensityGraph, load a NeuralDensityMap and use the static
construction method `from_neural_density_map()`:

```python
neural_map = NeuralDensityMap.load("emd_6272.neural")
density_map_graph = DensityGraph.from_neural_density_map(neural_map)
```

Though the graph must be created from a NeuralDensityMap, the method
`from_voxel_density_map()` can create a DensityGraph from the voxel
representation and handles the intermediate NeuralDensityMap creation. The
downside is that the NeuralDensityMap will not be saved.

The graph may be saved to disk to avoid re-creating the graph:

```python
density_map_graph.save("emd_6272.graph")
```

### Experiments

Experimental code was run on DAIS7.uwb.edu, which contains at least 7 GPUs. To
run scripts on other systems, the scripts may need to be modified to
accommodate the GPU setup of that system, and the runtime will be impacted.

#### Dataset Creation and Selection

The folder `experiments/data/` contains the resources to generate and
downselect maps for use in the experiments in the paper. EMDB map files, FASTA
sequence files, and PDB files are are downloaded for the items in the file
`EMDRSearch_HighRes.csv` leveraging existing code in the DeepTracer project
for the download.

The parameters for dataset downselection are given in the `neural_dataset.py`
module.

#### Interpolation Experiment

Chimera is a prerequisite for this experiment, and the path to the executable
for Chimera must be specified in the environment variable `CHIMERA_PATH`.

This experiment uses the pre-downloaded and selected dataset to generate four
cryo-EM maps for each dataset item:

1. Simulated cryo-EM map from PDB file at the resolution of the corresponding
EMDB map created using Chimera's molmap tool with default arguments.

2. Simulated cryo-EM map from PDB file at the resolution of the corresponding
EMDB map created using Chimera's molmap tool with 0.2 Angstrom voxel size,
which serves as the control in the experiment.

3. Cryo-EM map interpolated to voxels of size 0.2 Angstroms using Chimera's
resampling tool.

4. Neural cryo-EM map interpolated to voxels of size 0.2 Angstroms.

The maps are evaluated for their error against the control.

The `results` subfolder contains the raw data from the experiments run for the
paper. The `visualize.py` module uses the raw data to generate figures and CSV
versions of the raw data.

#### Density Graph Experiment

This experiment creates density graphs from the pre-downloaded and selected
dataset of EMDB maps and evaluates the graphs' nodes against the corresponding
PDB structures. For each item in the dataset, we:

1. Create and save a neural cryo-EM map, also generating a re-creation of the
original voxel map.

2. Create and save a density graph using the neural cryo-EM map, also
generating a mmCIF visualization of the graph.

3. Perform a DeepTracer prediction on the original voxel map.

4. Gather metrics on all the various outputs and save them as a `dict()` item
in a JSON file.

The `results` subfolder contains the raw data from the experiments run for the
paper. The `visualize.py` module uses the raw data to generate figures and CSV
versions of the raw data.
