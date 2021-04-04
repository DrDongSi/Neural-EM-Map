"""The module provides a graph data format created directly from 3DEM maps.

The DensityGraph class utilizes the NeuralDensityMap format to create a
graph from local dense points. This method is agnostic to original map
resolution as it simply finds locally dense points for node placement.
Consumers of this graph may need to contextualize the underlying resolution.
Graphs are represented using the undirected Graph type from the NetworkX
library.
"""
from __future__ import annotations
from copy import deepcopy
import os
import multiprocessing as mp
from typing import Dict, List, Tuple

from networkx import connected_components, Graph, read_gpickle, write_gpickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_chunked

from neural_density_map import NeuralDensityMap


class DensityGraph:
    """Graph format of 3DEM maps, created from the NeuralDensityMap format.

    The graph places nodes at locally dense points, detected by the following
    method. Points are sampled across the entire map, iteratively walked along
    their gradient to density peaks, and then clustered. Candidate nodes are
    placed at the centroid locations for each cluster. Nodes are filtered based
    on adjacency and sub-graph constraints and then represented by a undirected
    Graph from the NetworkX library.

    To create the DensityGraph, load a NeuralDensityMap and use the static
    construction method `from_neural_density_map()`:

        neural_map = NeuralDensityMap.load("emd_0420.neural")
        density_map_graph = DensityGraph.from_neural_density_map(neural_map)

    Though the graph must be created from a NeuralDensityMap, the method
    `from_voxel_density_map()` can create a DensityGraph from the voxel
    representation and handles the intermediate NeuralDensityMap creation. The
    downside is that the NeuralDensityMap will not be saved.

    The graph may be saved to disk to avoid re-creating the graph:

        density_map_graph.save("emd_0420.graph")

    Attributes:
        graph: Graph represented as a NetworkX undirected Graph. See
            https://networkx.org/documentation/stable/reference/classes/graph.html
            for information on the type and how to use it. It will contain an
            attribute called "map_id" which is used to identify the cryo-EM
            map whose density the graph represents. Each node has attributes:
                "x": The x-coordinate of the node in Angstroms
                "y": The y-coordinate of the node in Angstroms
                "z": The z-coordinate of the node in Angstroms
                "density": The cryo-EM density of the node
            and each edge as an attribute:
                "length": Length in Angstroms of the edge connecting two nodes
    """
    def __init__(self,
                 map_id: str,
                 nodes: List[Tuple[int, Dict[str, float]]],
                 edges: List[Tuple[int, int, Dict[str, float]]]) -> None:
        """Constructs a DensityGraph.

        Args:
            map_id: ID of the cryo-EM map this graph represents, used as an
                identifier for the graph.
            nodes: List of nodes of the graph in the node 2-tuple form:
                [(
                  unique node index,
                  {
                      "x": x-coordinate in Angstroms,
                      "y": y-coordinate in Angstroms,
                      "z": z-coordinate in Angstroms,
                      "density": cryo-EM density at coordinate,
                  })]
            edges: List of edges of the graph in the edge 3-tuple form:
                [(
                  node index of head,
                  node index of tail,
                  {
                      "length": length of edge in Angstroms
                  })]
        """
        graph = Graph(map_id=map_id)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        self.graph = graph

    def save_as_mmcif(self,
                      filename: str = None,
                      include_bonds: bool = True) -> None:
        """Saves the graph as a mmCIF file for visualization purposes.

        For very large graphs, set the include_bonds argument to false to
        limit the file size and visualization performance.

        Args:
            filename: Optional; Name of the save file.
            include_bonds: Optional; Flag to include the bonds in the CIF file
                which will reduce the file size and time to load the file in
                the visualization software.
        """
        # Use the map ID in the default filename
        if filename is None:
            filename = f"./{self.graph.graph['map_id']}_graph.cif"

        # Format the mmCIF dictionaries using a custom GRA chemical component
        data_block = f"data_{self.graph.graph['map_id']}_graph"
        entity = "\n".join([
            "_entity.id       1",
            "_entity.details  'Graph Visualization'"])
        author = "_audit_author.name DeepTracer"
        chem_comp = "\n".join([
            "_chem_comp.id            GRA",
            "_chem_comp.type          non-polymer",
            "_chem_comp.model_source  'Graph Visualization'",
            "_chem_comp.name          Graph"])
        chem_comp_atoms_header = "\n".join([
            "loop_",
            "_chem_comp_atom.comp_id",
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol"])
        atom_types = "\n".join([
            "loop_",
            "_atom_type.symbol",
            "C"])
        atom_sites_header = "\n".join([
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num"])

        # Chemical component names are limited to 4 characters, so create a set
        # of names
        char_set = ("0123456789"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz")
        chain_ids = [f"{l1}"
                     for l1 in char_set]
        chain_ids += [f"{l1}{l2}"
                      for l1 in char_set[1:]
                      for l2 in char_set]
        chain_ids += [f"{l1}{l2}{l3}"
                      for l1 in char_set[1:]
                      for l2 in char_set
                      for l3 in char_set]
        chain_ids += [f"{l1}{l2}{l3}{l4}"
                      for l1 in char_set[1:]
                      for l2 in char_set
                      for l3 in char_set
                      for l4 in char_set]

        # Use the graph nodes to format the chemical component and
        # atomic locations
        chem_comp_atom_lines = [f"GRA {chain_ids[node]} C"
                                for node in self.graph.nodes]
        chem_comp_atoms = "\n".join(
            [chem_comp_atoms_header, "\n".join(chem_comp_atom_lines)])
        atom_lines = [(f"HETATM {i} C {chain_ids[n]} . GRA A 1 . ? "
                       f"{ndata['x']} {ndata['y']} {ndata['z']} "
                       f"1.00 0.00 ? . GRA A {chain_ids[n]} 1")
                      for i, (n, ndata) in enumerate(self.graph.nodes.data())]
        atom_sites = "\n".join([atom_sites_header, "\n".join(atom_lines)])

        # Include the bonds (graph edges) in the file
        if include_bonds:
            chem_comp_bonds_header = "\n".join([
                "loop_",
                "_chem_comp_bond.comp_id",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2"])
            chem_comp_bonds_lines = [
                f"GRA {chain_ids[u]} {chain_ids[v]}"
                for u, v in self.graph.edges]
            chem_comp_bonds = "\n".join(
                [chem_comp_bonds_header, "\n".join(chem_comp_bonds_lines)])
        else:
            chem_comp_bonds = ""

        # Write the CIF file
        with open(filename, 'w') as mmcif:
            mmcif.write("\n#\n".join([
                data_block, entity, author,
                chem_comp, chem_comp_atoms, chem_comp_bonds,
                atom_types, atom_sites]))
            mmcif.write("\n#\n")

    def save(self, filename: str = None) -> None:
        """Saves the DensityGraph to a file for persistent storage.

        This method saves the graph using pickling, and it can be retrieved
        reconstructed by the class's load() method. The default save name is
        "./{self.map_id}.graph", but a filename ending in ".gz" will result in
        the saved file being compressed.

        Args:
            filename: Optional; Name of the save file.
        """
        if filename is None:
            filename = f"./{self.map_id}.graph"
        dir_name = os.path.dirname(filename)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        # Write out the file
        write_gpickle(self.graph, filename)

    @staticmethod
    def _calculate_centroid(points: np.ndarray,
                            cluster_labels: np.ndarray,
                            cluster: int) -> np.ndarray:
        """Calculates the centroid of the given cluster.

        This function is intended to be used in a multiprocessing context, and
        is therefore not a memory-efficient implementation. By including the
        point-cluster masking in this function we can offload that to multiple
        processes at the cost of copying all the points and cluster labels to
        each process.

        Args:
            points: Array of points with shape (n, 3), where n is the
                number of points.
            cluster_labels: Array of cluster labels with shape (n,) where each
                index n corresponds to a point in the points argument.
            cluster: The cluster this function will calculate the centroid of.

        Returns:
            The centroid point.
        """
        # Get the points within the cluster
        cluster_points = points[cluster_labels == cluster]

        # Perform an average of the points
        length = cluster_points.shape[0]
        sum_x = np.sum(cluster_points[:, 0])
        sum_y = np.sum(cluster_points[:, 1])
        sum_z = np.sum(cluster_points[:, 2])
        return np.array([sum_x / length,
                         sum_y / length,
                         sum_z / length])

    @staticmethod
    def _cluster_points(neural_map: NeuralDensityMap,
                        points: np.ndarray,
                        min_samples: int,
                        eps: float,
                        devices: List[str] = []) -> np.ndarray:
        """Clusters the given points and returns the centroid locations.

        This is a static helper method for creating the DensityGraph.
        It encapsulates the clustering portion of the creation process,
        where groups of locally dense points are identified to be used for
        node placement.

        Args:
            neural_map: The NeuralDensityMap whose density values are used for
                weighting the centroid calculation.
            points: Points to be clustered.
            min_samples: Minimum number of grouped points to be considered
                a cluster.
            eps: Maximum distance between points for them to be considered
                part of the same group.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            Set of centroid locations representing the locally dense points
            of the NeuralDensityMap.
        """
        # Perform clustering
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        num_clusters = clusters.max() + 1

        # Calculate the centroids of the clusters using multiprocessing for
        # a performance boost
        with mp.Pool(mp.cpu_count()) as pool:
            centroids = pool.starmap(
                DensityGraph._calculate_centroid,
                [(points, clusters, c) for c in range(num_clusters)])

        return np.stack(centroids)

    @staticmethod
    def _walk_to_dense_points(neural_map: NeuralDensityMap,
                              seed_points: np.ndarray,
                              step_size: float,
                              devices: List[str] = []) -> np.ndarray:
        """Walks the input points along their gradients to density peaks.

        This is a static helper method for creating the DensityGraph.
        It encapsulates the logic to find a density peak location for each
        given point. Using the densit and gradient values of the
        NeuralDensityMap, each seed point is iteratively translated along
        the gradient vector until the step results in a lower density value
        than the previous step.

        Args:
            neural_map: The NeuralDensityMap for sampling density and gradient.
            seed_points: The points to walk to their density peaks.
            step_size: The distance to adjust each point along their
                respective gradient vector each iteration.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            Array of points the same size and shape of the input seed points
            whose locations are each seed point's density peak.
        """
        dense_points = deepcopy(seed_points)
        walk_points = deepcopy(dense_points)
        num_points = dense_points.shape[0]
        walk_mask = np.full(num_points, True)
        highest_densities = np.full(num_points, 0.0)
        while True:
            # Get the gradient at all points that have yet to reach a peak
            values = neural_map.get_values(walk_points[walk_mask], devices)
            densities = values[:, 0]
            gradients = values[:, 1:4]

            # Update and check exit criteria
            not_nan = np.logical_not(np.isnan(densities))
            keep_walking = not_nan
            threshold = highest_densities[walk_mask]
            keep_walking[not_nan] = (densities[not_nan] > threshold[not_nan])
            walk_mask[walk_mask] = keep_walking
            if not np.any(walk_mask):
                return dense_points
            highest_densities[walk_mask] = densities[keep_walking]
            dense_points[walk_mask] = walk_points[walk_mask]

            # Take a step toward (hopefully) denser location
            gradients = gradients[keep_walking]
            # Manually compute norms since it is faster than np.linalg.norm(),
            # then convert to column vector
            magnitudes = np.sqrt((gradients * gradients).sum(axis=1))
            magnitudes = magnitudes[:, np.newaxis]
            walk_step = step_size * (gradients / magnitudes)
            walk_points[walk_mask] += walk_step

    @staticmethod
    def from_neural_density_map(
        neural_map: NeuralDensityMap,
        seed_threshold: float = None,
        adjacency_threshold: float = 4.0,
        min_num_nodes: int = 10,
        devices: List[str] = []
    ) -> DensityGraph:
        """Creates a DensityGraph from a NeuralDensityMap.

        This method creates a graph form of a NeuralDensityMap. The method
        places nodes at the map's locally dense points and connects them
        according to the given adjacency distance. There are four main steps
        to this algorithm:
            1. Uniformly sample the map to find non-sparse points
            2. Use the gradient vectors the points to move them to peak
               density locations.
            3. Cluster the density peaks and use the cluster centroids as
               node locations.
            4. Filter the nodes based on the adjacency threshold and allowed
               size of disconnected sub-graphs.

        Args:
            neural_map: The NeuralDensityMap from which to create the
                DensityGraph.
            seed_threshold: Optional; This value, if given, is the threshold of
                density values used when sampling the NeuralDensityMap for
                initial points. If None, the seed threshold is given as the
                (mean + (std. deviation * 3)) of the original 3DEM map data.
            adjacency_threshold: Optional; This distance is the maximum
                Euclidean distance nodes can be from one another to be
                considered adjacent. Adjacent nodes are connected by an edge.
            min_num_nodes: Optional; Used as a filter mechanism in the graph
                creation, it ensures that no disconnected sub-graphs exist that
                are too small.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A constructed DensityGraph.
        """
        # Sample the map with half Angstrom steps at most, filtered at the
        # mean + (standard deviation * 3) or 0.1 if those values are not
        # available
        sample_step = min(0.5, min(neural_map.voxel_size))
        if seed_threshold is None:
            if (neural_map.original_mean is not None and
                    neural_map.original_std is not None):
                seed_threshold = max(
                    0.1, neural_map.original_mean + (3 * neural_map.original_std))
            else:
                seed_threshold = 0.1
        seed_points = neural_map.sample_points(
            step_x=sample_step,
            step_y=sample_step,
            step_z=sample_step,
            density_threshold=seed_threshold,
            devices=devices)

        # With the seed points, walk gradients to find local dense points
        dense_points = DensityGraph._walk_to_dense_points(
            neural_map, seed_points, step_size=0.05, devices=devices)

        # Cluster the points and calculate centroids for node placement
        node_locations = DensityGraph._cluster_points(
            neural_map, dense_points, min_samples=1, eps=0.2, devices=devices)

        # To avoid the memory explosion from calculating the contact map for a
        # large number of potential node locations, break up the computation
        # into chunks of max. size of 2 GB
        contact_indices = list()

        def extract_contact_indices(distance_chunk: np.ndarray,
                                    start_row: int) -> None:
            """Extract contact indices from pairwise distance calculations.

            Args:
                distance_chunk: Slice of pairwise distance matrix.
                start_row: Offset row of slice in relation to a full pairwise
                    square matrix.
            """
            # Pairwise distance matrix is symmetric, so we only care about
            # upper (or lower) triangle. Also exclude diagonal to prevent
            # self-contacts
            chunk_contact_indices = np.stack(
                np.nonzero(
                    np.triu(
                        distance_chunk <= adjacency_threshold,
                        k=(start_row + 1))),
                axis=-1)
            chunk_contact_indices[:, 0] += start_row
            contact_indices.extend(chunk_contact_indices)
            return None

        for _ in pairwise_distances_chunked(
                node_locations,
                reduce_func=extract_contact_indices,
                working_memory=(2 * 1024)):  # Limit calculations to 2 GB memory
            pass

        # Create temporary graph to find disconnected sub-graphs and remove
        # them based on the minimum node count
        if min_num_nodes > 1:
            temp_graph = Graph()
            temp_graph.add_edges_from(contact_indices)
            filtered_subgraphs = [temp_graph.subgraph(c)
                                  for c in connected_components(temp_graph)
                                  if len(c) >= min_num_nodes]
            contact_indices = [e for s in filtered_subgraphs for e in s.edges]

        # Create nodes
        node_mask = list(set([i for c in contact_indices for i in c]))
        node_locations = node_locations[node_mask]
        node_densities = neural_map.get_densities(node_locations, devices)
        nodes = [(
            i,
            {
                "x": location[0],
                "y": location[1],
                "z": location[2],
                "density:": float(density)
            }) for i, (location, density) in enumerate(zip(node_locations, node_densities))]

        # Create edges
        index_mapping = {k: v for v, k in enumerate(node_mask)}
        edges = [(
            index_mapping[i],
            index_mapping[j],
            {
                "length": np.sqrt(
                        ((node_locations[index_mapping[i]] -
                          node_locations[index_mapping[j]]) ** 2).sum(axis=0))
            }) for i, j in contact_indices]

        return DensityGraph(neural_map.map_id, nodes, edges)

    @staticmethod
    def from_voxel_density_map(
        voxel_map_file: str,
        contour: float = None,
        adjacency_threshold: float = 4.0,
        min_num_nodes: int = 10,
        devices: List[str] = []
    ) -> DensityGraph:
        """Creates a DensityGraph directly from a voxel density map

        This method creates a DensityGraph from a voxel density map without
        the intermediary neural density map. The NeuralDensityMap is created
        in this function, but it is not saved. The creation of the
        NeuralDensityMap necessitates some input arguments for the percentile
        reduction of the voxel data and the devices used for training the
        neural representations.

        Args:
            voxel_map_file: Filename of the 3DEM map to create a graph from.
            contour: Optional; If given, the area of the voxel map is reduced
                to the area that contains voxel values above the contour value
                with a 5 Angstrom buffer in each axis. The default value of
                None means that the entire voxel map is used to create the
                NeuralDensityMap.
            adjacency_threshold: Optional; This distance is the maximum
                Euclidean distance nodes can be from one another to be
                considered adjacent. Adjacent nodes are connected by an edge.
            min_num_nodes: Optional; Used as a filter mechanism in the graph
                creation, it ensures that no disconnected sub-graphs exist that
                are too small.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A constructed DensityGraph.
        """
        # Train a NeuralDensityMap from the given voxel map
        neural_map = NeuralDensityMap.from_voxel_map(
            voxel_map_file, contour, devices)

        # Create a DensityGraph from the neural map
        return DensityGraph.from_neural_density_map(neural_map,
                                                    contour,
                                                    adjacency_threshold,
                                                    min_num_nodes,
                                                    devices)

    @staticmethod
    def load(filename: str) -> DensityGraph:
        """Load a DensityGraph from a file.

        This method loads a file created by this class's save() method
        to avoid re-creating the graph from the original density data. If the
        filename ends with ".gz" the file will be uncompressed before loading.

        Args:
            filename: Name of the file to be loaded.

        Returns:
            A constructed DensityGraph.
        """
        graph = read_gpickle(filename)
        return DensityGraph(graph.graph["map_id"],
                            graph.nodes.data(),
                            graph.edges.data())
