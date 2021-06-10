"""Continuous, non-linear representation of 3DEM maps using neural networks.

This module provides a data format of 3DEM density maps that is fully
continuous and differentiable. Using the original voxel data of a 3DEM density
map, a "neural density map" uses the SIREN neural network architecture to learn
a non-linear interpolation of the map. Maps are composed of many bespoke neural
density regions trained on separate portions of the voxel data. The properties
of SIREN models make it trivial to retrieve not only the density value at any
point but also the gradient vector at any point. The neural density maps
represent data in Cartesian coordinate space.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
import os
import queue
from typing import Dict, List, Tuple
import tarfile
import time

import mrcfile
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from siren import Siren


# To avoid running out of file descriptors when creating many
# NeuralDensityMaps using multiple devices.
# https://github.com/pytorch/pytorch/issues/973
mp.set_sharing_strategy('file_system')


class ContinuousRegion:
    """Boundary and sampling logic for a continuous region of xyz-space.

    Attributes:
        x_start: Smallest x-axis value in the region
        x_end: Largest x-axis value in the region
        y_start: Smallest y-axis value in the region
        y_end: Largest y-axis value in the region
        z_start: Smallest z-axis value in the region
        z_end: Largest z-axis value in the region
    """
    def __init__(self,
                 x_start: float,
                 x_end: float,
                 y_start: float,
                 y_end: float,
                 z_start: float,
                 z_end: float) -> None:
        """Constructs a ContinuousRegion object.

        Args:
            x_start: Starting value of the region's x-axis.
            x_end: Ending value of the regions's x-axis. It must by greater
                than the value of the x_start argument.
            y_start: Starting value of the region's y-axis.
            y_end: Ending value of the regions's y-axis. It must by greater
                than the value of the y_start argument.
            z_start: Starting value of the region's z-axis.
            z_end: Ending value of the regions's z-axis. It must by greater
                than the value of the z_start argument.

        Raises:
            ValueError: An end value is less than or equal to a start value.
        """
        # Verify the end values are less than the start values for each axis
        if (x_end <= x_start) or (y_end <= y_start) or (z_end <= z_start):
            raise ValueError("End values must be greater than start values")

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.z_start = z_start
        self.z_end = z_end

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Determines whether points exist within the region.

        Args:
            points: The points to be checked for region membership. The array
                must be of shape (n, 3), where n is the number of points and
                points[:, 0], points[:, 1], and points[:, 2] are the points'
                x, y, and z values respectively.
        Returns:
            Boolean array corresponding to the input points whose value
            indicates the point's presence in the region.
        """
        start = np.array([self.x_start, self.y_start, self.z_start])
        end = np.array([self.x_end, self.y_end, self.z_end])
        return np.logical_and(
            np.all((start <= points), axis=1),
            np.all((points <= end), axis=1))

    def get_sample_shape(self,
                         step_x: float,
                         step_y: float,
                         step_z: float) -> Tuple[int, int, int]:
        """Returns the number of samples for each axis given a step size.

        Args:
            step_x: Size of spacing between values on the x-axis
            step_y: Size of spacing between values on the y-axis
            step_z: Size of spacing between values on the z-axis

        Returns:
            A tuple of integers with the number of samples for the x, y, and z
            axes respectively.

        Raises:
            ValueError: If any of the step arguments are less than or equal to
            0.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        sample_parameters = self.get_sample_parameters(step_x, step_y, step_z)
        _, _, x_samples = sample_parameters[0]
        _, _, y_samples = sample_parameters[1]
        _, _, z_samples = sample_parameters[2]
        return (x_samples, y_samples, z_samples)

    def generate_sample_points(self,
                               step_x: float,
                               step_y: float,
                               step_z: float) -> np.ndarray:
        """Generate a set of points that sample the region given the step size.

        If the step size is not a whole number divisor of the axis size, the
        start and end values of the axis will be adjusted to maintain constant
        spacing between samples.

        Args:
            step_x: Size of spacing between values on the x-axis
            step_y: Size of spacing between values on the y-axis
            step_z: Size of spacing between values on the z-axis

        Returns:
            Points that sample the region at the given step size. The array
            has the shape (n, 3), where n is the number of points.

        Raises:
            ValueError: If any of the step arguments are less than or equal to
            0.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        sample_parameters = self.get_sample_parameters(step_x, step_y, step_z)
        sample_x_start, sample_x_end, x_samples = sample_parameters[0]
        sample_y_start, sample_y_end, y_samples = sample_parameters[1]
        sample_z_start, sample_z_end, z_samples = sample_parameters[2]
        axes = (np.linspace(sample_x_start, sample_x_end, x_samples),
                np.linspace(sample_y_start, sample_y_end, y_samples),
                np.linspace(sample_z_start, sample_z_end, z_samples))
        return np.stack(
            np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)

    def get_sample_parameters(
        self,
        step_x: float,
        step_y: float,
        step_z: float
    ) -> Tuple[Tuple[float, float, int],
               Tuple[float, float, int],
               Tuple[float, float, int]]:
        """Calculates parameters for sampling region at the given step sizes.

        Args:
            step_x: Size of spacing between values on the x-axis
            step_y: Size of spacing between values on the y-axis
            step_z: Size of spacing between values on the z-axis

        Returns:
            Tuple of three tuples, each containing the start point, end point,
            and number of samples for the x, y, and z axes respectively.

        Raises:
            ValueError: If any of the step arguments are less than or equal to
            0.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        # Assume sample begins on region starting points, calculate the end
        # values from step size
        sample_x_end = math.floor(self.x_end / step_x) * step_x
        x_samples = int(round((sample_x_end - self.x_start) / step_x)) + 1
        sample_y_end = math.floor(self.y_end / step_y) * step_y
        y_samples = int(round((sample_y_end - self.y_start) / step_y)) + 1
        sample_z_end = math.floor(self.z_end / step_z) * step_z
        z_samples = int(round((sample_z_end - self.z_start) / step_z)) + 1
        return ((self.x_start, sample_x_end, x_samples),
                (self.y_start, sample_y_end, y_samples),
                (self.z_start, sample_z_end, z_samples))


class NeuralDensityMap:
    """Fully continuous and differentiable neural representation of a 3DEM map.

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

        neural_map = NeuralDensityMap.from_voxel_map("./emd_0420.map")

    The neural representation can be saved to prevent the need for re-training
    of the neural regions by using the `save()` method:

        neural_map.save("./emd_0420.neural")

    To visualize the NeuralDensityMap, a voxel density map can be created
    using the `save_as_voxel_map()` function. This function is also useful
    for resamping a density map with a different voxel size. For example,

        neural_map.save_as_voxel_map("./emd_0420_resampled.map",
                                     (0.5, 0.5, 0.5))

    For querying values in the map, the methods `get_xxx()` are for arbitrary
    points within the map. For uniform sampling of density values and
    gradients, use the `sample_xxx()` methods as they are optimized for whole
    map querying.

    Attributes:
        map_id: Identifier of the map, normally the EMD-xxxx ID tag.
        voxel_shape: Shape of the original voxel data.
        voxel_size: Size of the original voxels.
        voxel_origin: Offset from the origin of the first voxel in the
            original voxel data.
        region: ContinuousRegion encapsulating the entire map region.
        original_mean: Mean of the original voxel data.
        original_std: Standard deviation of the original voxel data.
    """
    def __init__(self,
                 map_id: str,
                 regions: List[NeuralDensityRegion],
                 voxel_shape: Tuple[int, int, int] = None,
                 voxel_size: Tuple[float, float, float] = None,
                 voxel_origin: Tuple[float, float, float] = None,
                 original_mean: float = None,
                 original_std: float = None) -> None:
        """Constructs a NeuralDensityMap.

        Args:
            map_id: Identifier of the map, normally the EMD-xxxx ID tag.
            regions: List of NeuralDensityRegions that compose the map.
            voxel_shape: Optional; Shape of the original voxel data.
            voxel_size: Optional; Size of the original voxels.
            voxel_origin: Optional; Offset from the origin of the first voxel
                in the original voxel data.
            original_mean: Mean of the original voxel data.
            original_std: Standard deviation of the original voxel data.
        """
        # Set map properties
        self.map_id = map_id
        self.voxel_shape = voxel_shape
        self.voxel_size = voxel_size
        self.voxel_origin = voxel_origin
        self.original_mean = original_mean
        self.original_std = original_std

        # Calculate the total region info
        self.region = ContinuousRegion(
            x_start=min([r.region.x_start for r in regions]),
            x_end=max([r.region.x_end for r in regions]),
            y_start=min([r.region.y_start for r in regions]),
            y_end=max([r.region.y_end for r in regions]),
            z_start=min([r.region.z_start for r in regions]),
            z_end=max([r.region.z_end for r in regions]))

        # Initialize the neural sub-regions dictionary
        self._sub_regions = {
            k: {
                "overlap_pos_x": None,
                "overlap_neg_x": None,
                "overlap_pos_y": None,
                "overlap_neg_y": None,
                "overlap_pos_z": None,
                "overlap_neg_z": None,
            } for k in regions}

        # Assume each neural sub-region is the same size along a given axis,
        # and assume that each sub-regions overlap only along the coordinate
        # axes with at most one other sub-region per axis direction (meaning
        # at most two total per axis, one on the positive side and another on
        # the negative side). Map each sub-region to the next ones along each
        # axis and calculate the area of overlap
        for neural_region in regions:
            # Iterate through other regions looking for overlaps
            overlaps = self._sub_regions[neural_region]
            for other_neural_region in regions:
                # Skip ourselves
                if neural_region is other_neural_region:
                    continue

                r1 = neural_region.region
                r2 = other_neural_region.region

                # Determine axis alignment
                same_x = (r1.x_start == r2.x_start and r1.x_end == r2.x_end)
                same_y = (r1.y_start == r2.y_start and r1.y_end == r2.y_end)
                same_z = (r1.z_start == r2.z_start and r1.z_end == r2.z_end)

                # Check X overlap
                if (same_y and same_z):
                    if (r1.x_start < r2.x_start < r1.x_end):
                        overlaps["overlap_pos_x"] = ContinuousRegion(
                                r2.x_start, r1.x_end,
                                r1.y_start, r1.y_end,
                                r1.z_start, r1.z_end)
                    if (r1.x_start < r2.x_end < r1.x_end):
                        overlaps["overlap_neg_x"] = ContinuousRegion(
                                r1.x_start, r2.x_end,
                                r1.y_start, r1.y_end,
                                r1.z_start, r1.z_end)

                # Check Y overlap
                if (same_x and same_z):
                    if (r1.y_start < r2.y_start < r1.y_end):
                        overlaps["overlap_pos_y"] = ContinuousRegion(
                                r1.x_start, r1.x_end,
                                r2.y_start, r1.y_end,
                                r1.z_start, r1.z_end)
                    if (r1.y_start < r2.y_end < r1.y_end):
                        overlaps["overlap_neg_y"] = ContinuousRegion(
                                r1.x_start, r1.x_end,
                                r1.y_start, r2.y_end,
                                r1.z_start, r1.z_end)

                # Check Z overlap
                if (same_x and same_y):
                    if (r1.z_start < r2.z_start < r1.z_end):
                        overlaps["overlap_pos_z"] = ContinuousRegion(
                                r1.x_start, r1.x_end,
                                r1.y_start, r1.y_end,
                                r2.z_start, r1.z_end)
                    if (r1.z_start < r2.z_end < r1.z_end):
                        overlaps["overlap_neg_z"] = ContinuousRegion(
                                r1.x_start, r1.x_end,
                                r1.y_start, r1.y_end,
                                r1.z_start, r2.z_end)

    def _get_values(self, points: np.ndarray, device: str) -> np.ndarray:
        """Patches together the density and gradient values."""
        # Create arrays to accumulate the results and calculate the average
        # from the underlying neural sub-regions
        accum = np.zeros((points.shape[0], 4), dtype=np.float32)
        div = np.zeros((points.shape[0]), dtype=np.int32)

        # For each sub-region calculate the weights for values in the
        # overlapped regions. Then calculate the values of the remaining
        # points in the sub-region.
        for sub_region, overlaps in self._sub_regions.items():
            sub_region_mask = sub_region.region.contains_points(points)
            sub_region_points = points[sub_region_mask]
            weights = np.zeros((sub_region_points.shape[0]), dtype=np.float32)
            remaining_mask = np.full((sub_region_points.shape[0]), True)

            # Update the weights for points in the positive overlapped regions
            # Note: Since each point in an overlapped region has two weighted
            # components, the arbitrary decision was made to increment the
            # divisor array when calculating the weights of the point in the
            # upper region.
            if overlaps["overlap_pos_x"] is not None:
                overlap = overlaps["overlap_pos_x"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap.x_end - overlap_points[:, 0]) /
                    (overlap.x_end - overlap.x_start))
            if overlaps["overlap_pos_y"] is not None:
                overlap = overlaps["overlap_pos_y"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap.y_end - overlap_points[:, 1]) /
                    (overlap.y_end - overlap.y_start))
            if overlaps["overlap_pos_z"] is not None:
                overlap = overlaps["overlap_pos_z"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap.z_end - overlap_points[:, 2]) /
                    (overlap.z_end - overlap.z_start))

            # Update the weights for points in the negative overlapped regions
            # Note: The divisor is updated here.
            if overlaps["overlap_neg_x"] is not None:
                overlap = overlaps["overlap_neg_x"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap_points[:, 0] - overlap.x_start) /
                    (overlap.x_end - overlap.x_start))
                div_mask = sub_region_mask.copy()
                div_mask[div_mask] = point_mask
                div[div_mask] += 1
            if overlaps["overlap_neg_y"] is not None:
                overlap = overlaps["overlap_neg_y"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap_points[:, 1] - overlap.y_start) /
                    (overlap.y_end - overlap.y_start))
                div_mask = sub_region_mask.copy()
                div_mask[div_mask] = point_mask
                div[div_mask] += 1
            if overlaps["overlap_neg_z"] is not None:
                overlap = overlaps["overlap_neg_z"]
                point_mask = overlap.contains_points(sub_region_points)
                remaining_mask[point_mask] = False
                overlap_points = sub_region_points[point_mask]
                weights[point_mask] += (
                    (overlap_points[:, 2] - overlap.z_start) /
                    (overlap.z_end - overlap.z_start))
                div_mask = sub_region_mask.copy()
                div_mask[div_mask] = point_mask
                div[div_mask] += 1

            # Update the weights and divisor of the points in the sub-region
            # that are not in an overlapped area
            weights[remaining_mask] = 1
            div_mask = sub_region_mask.copy()
            div_mask[div_mask] = remaining_mask
            div[div_mask] += 1
            accum[sub_region_mask] += sub_region.get_values(
                sub_region_points, device) * weights[:, np.newaxis]

        # Set remaining point values to NaN, comput averages, and return
        accum[div == 0] = np.nan
        accum[div != 0] /= div[div != 0][:, np.newaxis]
        return accum

    def get_values(self,
                   points: np.ndarray,
                   devices: List[str] = []) -> np.ndarray:
        """Gets the density values and gradient vectors for the given points.

        This method is for arbitrarily sampling the density and gradient
        vector. For uniform, full-map sampling use the `sample_values()`
        method.

        Args:
            points: Array of points with shape (n, 3), where n is the number
                of points. Each point has an x, y, and z coordinate
                respectively.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (n, 4), where n is the number of input
            points and each output n has the following format:
                [density, gradient_x, gradient_y, gradient_z].
            The order of output is preserved with the input's order of points.
            If an input point is not in the region, the corresponding output
            values for density and gradient are each np.nan.

        Raises:
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # If using only one device for computation simply perform the operation
        if len(devices) == 0:
            return self._get_values(points, "cuda")
        elif len(devices) == 1:
            return self._get_values(points, devices[0])

        # Validate/sanitize the device names
        device_names = set(devices)
        if ("cuda" in device_names) and ("cuda:0" in device_names):
            device_names.remove("cuda")
        device_names = list(device_names)

        # Scale down multiprocessing if less than 1,000,000 points per process
        while (len(device_names) > 1 and
               points.shape[0] / len(device_names) < 1000000):
            device_names = device_names[:-1]
        if len(device_names) == 1:
            return self._get_values(points, device_names[0])

        # Must use "spawn" method for CUDA support
        multiprocessing_context = mp.get_context("spawn")

        # Split up the data evenly to process with each device. We can use
        # starmap because there is a 1:1 mapping of data to devices.
        data_splits = np.array_split(points, len(device_names))
        with multiprocessing_context.Pool(len(device_names)) as pool:
            values = pool.starmap(
                self._get_values,
                [(data_splits[i], d) for i, d in enumerate(device_names)])

        return np.concatenate(values)

    def get_densities(self,
                      points: np.ndarray,
                      devices: List[str] = []) -> np.ndarray:
        """Gets the density values for the given points.

        This method is for arbitrarily sampling the density. For uniform,
        full-map sampling use the `sample_densities()` method.

        Args:
            points: Array of points with shape (n, 3), where n is the number
                of points. Each point has an x, y, and z coordinate
                respectively.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (n, 1), where n is the number of input
            points and each output n is the corresponding density value. The
            order of output is preserved with the input's order of points. If
            an input point is not in the region, the corresponding output
            value is np.nan.

        Raises:
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Slice off the first column from the more general get_values()
        # function
        return self.get_values(points, devices)[:, 0]

    def get_gradients(self,
                      points: np.ndarray,
                      devices: List[str] = []) -> np.ndarray:
        """Gets the gradient vectors for the given points.

        This method is for arbitrarily sampling the gradient vector. For
        uniform, full-map sampling use the `sample_values()` method.

        Args:
            points: Array of points with shape (n, 3), where n is the number
                of points. Each point has an x, y, and z coordinate
                respectively.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (n, 3), where n is the number of input
            points and each output n has the following format:
                [gradient_x, gradient_y, gradient_z].
            The order of output is preserved with the input's order of points.
            If an input point is not in the region, the corresponding output
            gradient components are each np.nan.

        Raises:
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Slice off the last three columns from the more general get_values()
        # function
        return self.get_values(points, devices)[:, 1:4]

    def sample_values(self,
                      step_x: float,
                      step_y: float,
                      step_z: float,
                      devices: List[str] = []) -> np.ndarray:
        """Sample the density values and gradient vectors across the whole map.

        This method samples the entire map for the density values and gradient
        vector. To maintain a constant step size in each axis, the starting
        and ending points are adjusted such that they will not lie on the
        map's region boundaries.

        Args:
            step_x: Size of spacing between the x-component of sample points.
            step_y: Size of spacing between the y-component of sample points.
            step_z: Size of spacing between the z-component of sample points.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (m, n, p, 4), where m, n, and p are the
            number of samples in the x, y, and z axes respectively. Each index
            [m, n, p] has the following format:
                [density, gradient_x, gradient_y, gradient_z].

        Raises:
            ValueError: If any of the arguments are less than or equal to 0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        # Calculate points across entire region
        x_samples, y_samples, z_samples = self.region.get_sample_shape(
            step_x, step_y, step_z)
        points = self.region.generate_sample_points(step_x, step_y, step_z)
        values = self.get_values(points, devices)
        values = values.reshape(x_samples, y_samples, z_samples, 4)
        return values

    def sample_densities(self,
                         step_x: float,
                         step_y: float,
                         step_z: float,
                         devices: List[str] = []) -> np.ndarray:
        """Sample the density values across the whole map.

        This method samples the entire map for the density values. To
        maintain a constant step size in each axis, the starting and ending
        points are adjusted such that they will not lie on the map's region
        boundaries.

        Args:
            step_x: Size of spacing between the x-component of sample points.
            step_y: Size of spacing between the y-component of sample points.
            step_z: Size of spacing between the z-component of sample points.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (m, n, p, 1), where m, n, and p are the
            number of samples in the x, y, and z axes respectively. Each index
            [m, n, p] is a density value.

        Raises:
            ValueError: If any of the arguments are less than or equal to 0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        # Slice the gradient data out of the output of sample_values()
        return self.sample_values(
            step_x, step_y, step_z, devices)[:, :, :, 0]

    def sample_gradients(self,
                         step_x: float,
                         step_y: float,
                         step_z: float,
                         devices: List[str] = []) -> np.ndarray:
        """Sample the gradient vectors across the whole map.

        This method samples the entire map for the gradient vectors. To
        maintain a constant step size in each axis, the starting and ending
        points are adjusted such that they will not lie on the map's region
        boundaries.

        Args:
            step_x: Size of spacing between the x-component of sample points.
            step_y: Size of spacing between the y-component of sample points.
            step_z: Size of spacing between the z-component of sample points.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A numpy array of the shape (m, n, p, 3), where m, n, and p are the
            number of samples in the x, y, and z axes respectively. Each index
            [m, n, p] has the following format:
                [gradient_x, gradient_y, gradient_z].

        Raises:
            ValueError: If any of the arguments are less than or equal to 0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        # Slice the density data out of the output of sample_values()
        return self.sample_values(
            step_x, step_y, step_z, devices)[:, :, :, 1:4]

    def sample_points(self,
                      step_x: float,
                      step_y: float,
                      step_z: float,
                      density_threshold: float,
                      devices: List[str] = []) -> np.ndarray:
        """Returns the points above a given density value threshold.

        This method samples the map at the given step sizes per axis. Then it
        applies a filter based on the given threshold of density values. In
        general, maps are very sparse, and the sparsity generally increases the
        higher the map resolution.

        Args:
            step_x: Size of spacing between the x-component of sample points.
            step_y: Size of spacing between the y-component of sample points.
            step_z: Size of spacing between the z-component of sample points.
            density_threshold: Density value where every sampled point with a
                density value less that the given value is removed from the
                output.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            Array of points with the shape (n, 3), where n is the number of
            points found to be above the density threshold determined by the
            input arguments.

        Raises:
            ValueError: If any of the step arguments are less than or equal to
            0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if step_x <= 0 or step_y <= 0 or step_z <= 0:
            raise ValueError("Step arguments must be greater than 0")

        # Sample the densities
        densities = self.sample_densities(step_x, step_y, step_z, devices)

        # Use ContinuousRegion to get the points of the densities
        x_samples, y_samples, z_samples = self.region.get_sample_shape(
            step_x, step_y, step_z)
        points = self.region.generate_sample_points(step_x, step_y, step_z)
        points = points.reshape(x_samples,
                                y_samples,
                                z_samples,
                                3)
        return points[densities >= density_threshold]

    def to_voxel_density_map(
        self,
        voxel_size: Tuple[float, float, float] = None,
        devices: List[str] = []
    ) -> np.ndarray:
        """Produces a voxel representation of the NeuralDensityMap.

        Use this method to produce the voxel representation using the given
        voxel size. Calling this method with the default argument will
        re-create the original voxel data from the neural representation. Note
        that the axis ordering for the input voxel size is in the XYZ ordering
        native to the MRC/MAP file format. However, the output voxel array is
        in swaps the X and Z axes to align with the preferred alignment of
        voxel data for use in a MRC/MAP file.

        Args:
            voxel_size: Optional; Tuple of values that are the side lengths of
                the voxels in the X, Y, and Z directions respectively. The
                default value of None signals the method to use the voxel sizes
                of the original voxel map, if possible, otherwise the voxels
                have length 1 on each side.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            The voxel data for a 3DEM map, in IJK ordering. There is no need to
            change axis order before writing to an MRC/MAP file.

        Raises:
            ValueError: If a provided voxel_size does not contain three values.
            ValueError: If any of the provided voxel_size values are less than
                or equal to 0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if voxel_size is not None:
            if len(voxel_size) != 3:
                raise ValueError("voxel_size must be three dimensional")
            if voxel_size[0] <= 0 or voxel_size[1] <= 0 or voxel_size[2] <= 0:
                raise ValueError("voxel_size values must be greater than 0")
        else:
            voxel_size = (self.voxel_size
                          if self.voxel_size is not None else (1., 1., 1.))

        # Simply sample the map at the original voxel size
        voxel_data = self.sample_densities(voxel_size[0],
                                           voxel_size[1],
                                           voxel_size[2],
                                           devices)

        # Swap the X and Z axes to transform into IJK format
        return np.swapaxes(voxel_data, 0, 2)

    def save_as_voxel_map(
        self,
        filename: str,
        voxel_size: Tuple[float, float, float] = None,
        devices: List[str] = []
    ) -> None:
        """Saves as a voxel 3DEM map using the given voxel size.

        Use this method to save the NeuralDensityMap as a voxel density map
        using the given voxel size as the step size for sampling the map. The
        default voxel_size argument will re-create the original voxel map from
        the neural representation. Note that the axis ordering for the input
        voxel size is in the XYZ ordering native to the MRC/MAP file format.

        Args:
            filename: Name of the output file. If the file already exists, it
                will be overwritten.
            voxel_size: Optional; Tuple of values that are the side lengths of
                the voxels in the X, Y, and Z directions respectively. The
                default value of None signals the method to use the voxel sizes
                of the original voxel map, if possible, otherwise the voxels
                have length 1 on each side.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Raises:
            ValueError: If a provided voxel_size does not contain three values.
            ValueError: If any of the provided voxel_size values are less than
                or equal to 0.
            ValueError: If a CUDA device is given and CUDA is not available.
        """
        # Validate the input arguments
        if voxel_size is not None:
            if len(voxel_size) != 3:
                raise ValueError("voxel_size must be three dimensional")
            if voxel_size[0] <= 0 or voxel_size[1] <= 0 or voxel_size[2] <= 0:
                raise ValueError("voxel_size values must be greater than 0")
        else:
            voxel_size = (self.voxel_size
                          if self.voxel_size is not None else (1., 1., 1.))

        # Create the voxel density map file
        voxel_data = self.to_voxel_density_map(voxel_size, devices)
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(voxel_data.astype(np.float32))
            mrc.voxel_size = voxel_size
            mrc.header.origin = self.voxel_origin
            mrc.header.nzstart = 0
            mrc.header.nystart = 0
            mrc.header.nxstart = 0

    def save(self, filename: str = None) -> None:
        """Saves the NeuralDensityMap to a file.

        This method save the NeuralDensityMap and the underlying
        NeuralDensityRegions to files such that they may each be re-created
        by the class's `load()` method. If no filename argument is provided,
        the default save location is the current directory with the map ID as
        the name. The default file extension is ".neural". If the filename
        ends with ".gz" the file will be compressed using gzip compression.

        Args:
            filename: Optional; Name of save file. If not provided, the
            default file name is "./{self.map_id}.neural".
        """
        if filename is None:
            filename = f"./{self.map_id}.neural"

        dir_name = os.path.dirname(filename)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        write_mode = "w:gz" if filename.endswith(".gz") else "w"
        with tarfile.open(filename, write_mode) as archive:
            metadata_filename = f"{os.path.join(dir_name, 'metadata')}.json"
            metadata = {
                "map_id": self.map_id,
                "regions": [],
                "voxel_shape": {
                    "x": self.voxel_shape[0],
                    "y": self.voxel_shape[1],
                    "z": self.voxel_shape[2],
                } if self.voxel_shape is not None else {},
                "voxel_size": {
                    "x": self.voxel_size[0],
                    "y": self.voxel_size[1],
                    "z": self.voxel_size[2],
                } if self.voxel_size is not None else {},
                "voxel_origin": {
                    "x": self.voxel_origin[0],
                    "y": self.voxel_origin[1],
                    "z": self.voxel_origin[2],
                } if self.voxel_origin is not None else {},
                "original_mean":
                    self.original_mean if self.original_mean is not None else -1,
                "original_std":
                    self.original_std if self.original_std is not None else -1,
            }
            for region in self._sub_regions.keys():
                region_filename = f"{os.path.join(dir_name, region.name)}.pt"
                region.save(region_filename)
                archive.add(region_filename)
                os.remove(region_filename)
                metadata["regions"].append(os.path.basename(region_filename))
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            archive.add(metadata_filename)
            os.remove(metadata_filename)

    @staticmethod
    def _create_region_proc(map_id: str,
                            input_queue: mp.Queue,
                            device_name: str,
                            output_queue: mp.Queue) -> None:
        """Process for creating NeuralDensityRegions using Queue.

        This method is to be used as a process in a multiprocessing
        environment to create NeuralDensityRegions. The process
        receives the creation parameters from an input queue and returns
        the created regions in an output queue.

        Args:
            map_id: Identifier of the original 3DEM map.
            input_queue: Queue containing input creation parameters
            device_name: Name of the device to perform the
                NeuralDensityRegion creation. It must be either "cpu",
                "cuda" (for the default CUDA device), or "cuda:{x}", where
                x is the zero-based index of the CUDA device.
            output_queue: Queue for placing created
                NeuralDensityRegions
        """
        while True:
            # Block waiting for queue data
            dequeue = input_queue.get()
            if dequeue is None:
                # Value of None in queue signals the process to exit
                break

            # Unpack the queue data
            region_id, voxels, region_info = dequeue
            name = f"{map_id}_region_{region_id}"

            # Place created region in the output queue
            output_queue.put(NeuralDensityRegion.create_from_voxels(
                name, voxels, region_info, device_name))

    @staticmethod
    def _create_regions_mp(
        map_id: str,
        regions: List[Tuple[np.ndarray, ContinuousRegion]],
        device_names: List[str]
    ) -> List[NeuralDensityRegion]:
        """Creates NeuralDensityRegions in a multiprocessing use case.

        This method creates a process for each of the given devices that
        is used to construct NeuralDensityRegions until the entire map is
        represented by NeuralDensityRegions.

        Args:
            map_id: Identifier of the original 3DEM map.
            regions: List of the voxel data and ContinuousRegion information
                to be used to construct NeuralDensityRegions.
            device_names: List of available devices for training the underlying
                neural representations. They must be either "cpu", "cuda" (for
                the default CUDA device), or "cuda:{x}", where x is the
                zero-based index of the CUDA device. The device names should
                not be in the list more than once, and mutually exclusive
                names such as "cuda" and "cuda:0" are merged automatically.

        Returns:
            List of created NeuralDensityRegions.
        """
        # Must use "spawn" method for CUDA support
        multiprocessing_context = mp.get_context("spawn")

        # Create multiprocessing queues for process input and output
        input_queue = multiprocessing_context.Queue()
        output_queue = multiprocessing_context.Queue()

        # Validate/sanitize the device names
        device_names = set(device_names)
        if ("cuda" in device_names) and ("cuda:0" in device_names):
            device_names.remove("cuda")

        # Create the processes
        processes = list()
        for device_name in device_names:
            # Create and start process
            process = multiprocessing_context.Process(
                target=NeuralDensityMap._create_region_proc,
                args=(map_id,
                      input_queue,
                      device_name,
                      output_queue))
            process.start()
            processes.append(process)

        # Pack the queue with clones of the voxel regions and region info,
        # then add markers to denote the end of the queue for subprocesses to
        # know when to exit
        neural_regions = list()
        for i, region in enumerate(regions):
            voxels, region_info = region
            put_success = False
            while not put_success:
                try:
                    input_queue.put_nowait((i, voxels, region_info))
                    put_success = True
                except queue.Full:
                    # If queue is full, receive some results from a process to
                    # prevent deadlock in the process
                    neural_regions.append(output_queue.get())

        # Retrieve the remaining results from the output queue
        remaining_regions = len(regions) - len(neural_regions)
        for _ in range(remaining_regions):
            neural_regions.append(output_queue.get())

        # Close processes
        for _ in range(len(processes)):
            input_queue.put(None)
        for process in processes:
            process.join()

        # Retrieve the NeuralDensityRegion from the output queue
        return neural_regions

    @staticmethod
    def _create_regions(map_id: str,
                        regions: List[Tuple[np.ndarray, ContinuousRegion]],
                        device: str) -> List[NeuralDensityRegion]:
        """NeuralDensityRegion creation method for a single threaded case.

        Args:
            map_id: Identifier of the original 3DEM map.
            regions: List of the voxel data and ContinuousRegion information
                to be used to construct NeuralDensityRegions.
            device: Name of the device to perform the NeuralDensityRegion
                creation. It must be either "cpu", "cuda" (for the default
                CUDA device), or "cuda:{x}", where x is the zero-based index
                of the CUDA device.

        Returns:
            List of created NeuralDensityRegions.
        """
        neural_regions = list()
        for i, r in enumerate(regions):
            voxels, region = r
            region_name = f"{map_id}_region_{i}"
            neural_regions.append(
                NeuralDensityRegion.create_from_voxels(
                    region_name, voxels, region, device))
        return neural_regions

    @staticmethod
    def from_voxel_data(
        map_id: str,
        voxels: np.ndarray,
        voxel_size: Tuple[float, float, float] = (1., 1., 1.),
        origin: Tuple[float, float, float] = (0., 0., 0.),
        devices: List[str] = []
    ) -> NeuralDensityMap:
        """Creates a NeuralDensityMap from arbitrary voxel data.

        This is a creation method for a NeuralDensityMap from arbitrary
        voxel data, and it requires some context parameters from the original
        voxel source. Negative voxel values are modified to be 0. The default
        use case creates the NeuralDensityMap using the default CUDA device
        (cuda:0) in a single-threaded manner. Passing in multiple device names
        will use a multiprocessing workflow that creates a process per device.

        Note that the ordering of the axes in the arguments is important since
        the native array data and metadata in MRC/MAP files is often not
        consistent in axis ordering. This function does not change the voxel
        axis ordering and assumes consistency in axis ordering between the
        voxels, voxel_size and origin arguments. To create a NeuralDensityMap
        from a voxel map source, it is recommended to use the
        `from_voxel_map()` function.

        Args:
            map_id: String identifier for the voxel region.
            voxels: 3-dimensional voxel data.
            voxel_size: Optional; Size of the voxels.
            origin: Optional; Location of the origin relative to the first
                voxel in order to adequately set-up spatial coordinates for
                the NeuralDensityMap.
            devices: Optional; List of available devices for performing the
                underlying neural representation training. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:{x}", where x is the zero-based index of the CUDA
                device.

        Returns:
            A newly created and constructed NeuralDensityMap.

        Raises:
            ValueError: If voxel data is empty.
            ValueError: If voxel data, voxel size, or origin is not
                3-dimensional.
            ValueError: If value in voxel size is less than or equal to 0.
        """
        # Validate input
        if voxels.size == 0:
            raise ValueError("Voxel data is empty")
        if len(voxels.shape) != 3:
            raise ValueError("Voxel data must be three dimensional")
        if len(voxel_size) != 3:
            raise ValueError("Voxel size must be three dimensional")
        if len(origin) != 3:
            raise ValueError("Origin location must be three dimensional")
        if voxel_size[0] <= 0 or voxel_size[1] <= 0 or voxel_size[2] <= 0:
            raise ValueError("voxel_size value must be greater than 0")

        # Negative voxel values are rubbish
        voxels[voxels < 0] = 0

        # Normalize voxel values to [0, 1]
        if voxels.max() == voxels.min():
            voxels = np.zeros(voxels.shape)  # Avoid NaN
        else:
            voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())

        # Get original mean and standard deviation
        original_mean = float(np.mean(voxels))
        original_std = float(np.std(voxels))

        # Create regions for SIREN training
        # Regions of maximum size 64 x 64 x 64 voxels
        # Regions overlap by minimum 4 voxels on each side (except map edges)
        original_shape = voxels.shape
        shape = np.array(original_shape)
        region_shape = np.minimum([64, 64, 64], shape)
        minimum_region_overlap = np.array([4, 4, 4])
        regions_per_axis = np.ceil(
            (shape - region_shape) /
            (region_shape - minimum_region_overlap)).astype(int) + 1
        # Calculate interval for starting new regions, handling the case
        # where there is only one region needed in a given axis
        interval_divisor = np.maximum(regions_per_axis - 1, [1, 1, 1])
        interval = (shape - region_shape) / interval_divisor

        # Split up the density voxels into regions and get the region info in
        # spatial XYZ-space
        regions = list()
        for i in range(regions_per_axis[0]):
            for j in range(regions_per_axis[1]):
                for k in range(regions_per_axis[2]):
                    slice_start = np.array(
                        [int(i * interval[0]),
                         int(j * interval[1]),
                         int(k * interval[2])])
                    slice_end = np.array(
                        [slice_start[0] + int(region_shape[0]),
                         slice_start[1] + int(region_shape[1]),
                         slice_start[2] + int(region_shape[2])])
                    voxel_slice = voxels[slice_start[0]:slice_end[0],
                                         slice_start[1]:slice_end[1],
                                         slice_start[2]:slice_end[2]]
                    # Calculate the region's spatial bounds. Subtract 1 from
                    # the slice_end since that value is non-inclusize
                    xyz_start = ((slice_start * np.array(voxel_size)) +
                                 np.array(origin))
                    xyz_end = (((slice_end - 1) * np.array(voxel_size)) +
                               np.array(origin))
                    region = ContinuousRegion(xyz_start[0], xyz_end[0],
                                              xyz_start[1], xyz_end[1],
                                              xyz_start[2], xyz_end[2])
                    regions.append((voxel_slice, region))

        # Perform the SIREN training using multiple devices and processes
        # if possible
        if len(devices) > 1:
            neural_regions = NeuralDensityMap._create_regions_mp(
                map_id, regions, devices)
        else:
            if len(devices) == 0:
                # If no devices given, assume default cuda device
                device = "cuda"
            else:
                device = devices[0]
            neural_regions = NeuralDensityMap._create_regions(
                map_id, regions, device)

        return NeuralDensityMap(map_id,
                                neural_regions,
                                original_shape,
                                voxel_size,
                                origin,
                                original_mean,
                                original_std)

    @staticmethod
    def from_voxel_map(voxel_map_file: str,
                       contour: float = None,
                       devices: List[str] = []) -> NeuralDensityMap:
        """Creates a NeuralDensityMap from the given 3DEM map file.

        This is the normal creation method for a NeuralDensityMap. The default
        use case creates the NeuralDensityMap only of the region within the map
        that contains the highest 1% of density values and using the default
        CUDA device (cuda:0) in a single-threaded manner. Passing in multiple
        device names will use a multiprocessing workflow that creates a
        process per device.

        Args:
            voxel_map_file: Filename of the 3DEM map to create a neural
                representation from.
            contour: Optional; If given, the area of the voxel map is reduced
                to the area that contains voxel values above the contour value
                with a 5 Angstrom buffer in each axis. The default value of
                None means that the entire voxel map is used to create the
                NeuralDensityMap.
            devices: Optional; List of available devices for performing the
                underlying neural representation operations. If no device names
                are given, the default CUDA device is used. Possible device
                name options are "cpu", "cuda" (for the default CUDA device),
                or "cuda:x", where x is the zero-based index of the CUDA
                device.

        Returns:
            A newly created and constructed NeuralDensityMap.
        """
        # Load density map from the file and extract the Column, Row, Section
        # (CRS) data which will be used to correctly interpret and permute the
        # volume data. Also convert the underlying data to Python types
        # instead of the np.recarray format.
        map_id = os.path.basename(voxel_map_file).split('.')[0]
        with mrcfile.open(voxel_map_file) as voxel_map:
            # Get voxel data and swap axes to be consistent with other CRS
            # header data
            voxels = np.swapaxes(deepcopy(voxel_map.data), 0, 2)

            # Get the voxel offset in CRS order
            voxel_offset = (int(voxel_map.header.nxstart),
                            int(voxel_map.header.nystart),
                            int(voxel_map.header.nzstart))

            # Get the voxel size
            voxel_size = (float(voxel_map.voxel_size.x),
                          float(voxel_map.voxel_size.y),
                          float(voxel_map.voxel_size.z))

            # Get the origin data
            origin = (float(voxel_map.header.origin.x),
                      float(voxel_map.header.origin.y),
                      float(voxel_map.header.origin.z))

            # Get the axis ordering, this specifies the Cartesian axis for the
            # respective C, R, and S values. Values of 1, 2, and 3 correspond
            # to the X, Y, and Z axes respectively. Subtract 1 from the value
            # to align the value with a zero-based index.
            axis_order = (int(voxel_map.header.mapc) - 1,
                          int(voxel_map.header.mapr) - 1,
                          int(voxel_map.header.maps) - 1)
            # If invalid axis order data, use default ordering
            if ((0 not in axis_order) or
                    (1 not in axis_order) or
                    (2 not in axis_order)):
                axis_order = (0, 1, 2)

        # Permute the axes of the voxel map data and metadata to align with
        # the XYZ Cartesian coordinate system
        if axis_order[0] != 0 or axis_order[1] != 1 or axis_order[2] != 2:
            # Calculate the conversion from CRS to XYZ
            axis_converter = [None, None, None]
            axis_converter[axis_order[0]] = 0
            axis_converter[axis_order[1]] = 1
            axis_converter[axis_order[2]] = 2

            # Convert the voxel array data to XYZ, voxel size and origin are
            # already in XYZ order
            voxels = np.transpose(voxels, axis_converter)
            voxel_offset = (voxel_offset[axis_converter[0]],
                            voxel_offset[axis_converter[1]],
                            voxel_offset[axis_converter[2]])

        # Reduce the voxel region to the area surrounding the given contour,
        # if given
        if contour is not None:
            x_sorted, y_sorted, z_sorted = [
                sorted(axis_indices)
                for axis_indices in np.nonzero(voxels >= contour)]

            # Calculate slice boundaries with an extra 5 Angstrom buffer
            extra_voxels = np.ceil(5 / np.array(voxel_size)).astype(int)
            x_lower = int(max(x_sorted[0] - extra_voxels[0], 0))
            x_upper = int(min(
                x_sorted[-1] + extra_voxels[0], voxels.shape[0] - 1))
            y_lower = int(max(y_sorted[0] - extra_voxels[1], 0))
            y_upper = int(min(
                y_sorted[-1] + extra_voxels[1], voxels.shape[1] - 1))
            z_lower = int(max(z_sorted[0] - extra_voxels[2], 0))
            z_upper = int(min(
                z_sorted[-1] + extra_voxels[2], voxels.shape[2] - 1))

            # Slice the voxel data into the reduced frame
            voxels = voxels[x_lower:x_upper + 1,
                            y_lower:y_upper + 1,
                            z_lower:z_upper + 1]

            # Modify the voxel offset based on the new region
            voxel_offset = (voxel_offset[0] + x_lower,
                            voxel_offset[1] + y_lower,
                            voxel_offset[2] + z_lower)

        # Calculate the true origin using the voxel offset and original origin
        origin = ((voxel_offset[0] * voxel_size[0]) + origin[0],
                  (voxel_offset[1] * voxel_size[1]) + origin[1],
                  (voxel_offset[2] * voxel_size[2]) + origin[2])

        return NeuralDensityMap.from_voxel_data(map_id,
                                                voxels,
                                                voxel_size,
                                                origin,
                                                devices)

    @staticmethod
    def load(filename: str) -> NeuralDensityMap:
        """Loads a NeuralDensityMap from file.

        This static construction method loads a NeuralDensityMap from a
        file created by the class method `save()`. It recursively loads the
        member NeuralDensityRegions and uses the json data to construct
        the NeuralDensityMap. If the filename ends with ".gz" the file will be
        uncompressed first before loading.

        Args:
            filename: Name of the file to load.

        Returns:
            Returns a constructed NeuralDensityMap.
        """
        read_mode = "r:gz" if filename.endswith(".gz") else "r"
        with tarfile.open(filename, read_mode) as archive:
            dir_name = os.path.dirname(filename)
            with archive.extractfile(os.path.join(dir_name, "metadata.json")) as f:
                load_data = json.load(f)
                map_id = load_data["map_id"]
                regions = list()
                for region_file in load_data["regions"]:
                    region_filename = os.path.join(dir_name, region_file)
                    archive.extract(region_filename)
                    regions.append(NeuralDensityRegion.load(region_filename))
                    os.remove(region_filename)
                voxel_shape = None
                voxel_size = None
                voxel_origin = None
                original_mean = None
                original_std = None
                if len(load_data["voxel_shape"]) != 0:
                    voxel_shape = (load_data["voxel_shape"]["x"],
                                   load_data["voxel_shape"]["y"],
                                   load_data["voxel_shape"]["z"])
                if len(load_data["voxel_size"]) != 0:
                    voxel_size = (load_data["voxel_size"]["x"],
                                  load_data["voxel_size"]["y"],
                                  load_data["voxel_size"]["z"])
                if len(load_data["voxel_origin"]) != 0:
                    voxel_origin = (load_data["voxel_origin"]["x"],
                                    load_data["voxel_origin"]["y"],
                                    load_data["voxel_origin"]["z"])
                if load_data["original_mean"] != -1:
                    original_mean = load_data["original_mean"]
                if load_data["original_std"] != -1:
                    original_std = load_data["original_std"]
                return NeuralDensityMap(map_id,
                                        regions,
                                        voxel_shape,
                                        voxel_size,
                                        voxel_origin,
                                        original_mean,
                                        original_std)


class NeuralDensityRegion:
    """Encapsulates a SIREN model trained to represent a region of a 3DEM map.

    This class is used to create a neural representation using the SIREN
    network architecture. The underlying neural representation is trained on
    the voxel data but allows for the querying of any coordinate in the region.
    Each NeuralDensityRegion may represent voxel data with a limit of 64 voxels
    per axis. Output density values are normalized to lie in the range of 0 and
    1 inclusively.

    The normal construction method is to utilize the static
    `create_from_voxels()` function. Some of the model parameters given as
    class attributes are specifc to the machine that trained the SIREN model.

    Attributes:
        name: Identifer for the region. Used as default save filename.
        region: ContinuousRegion describing the underlying representation.
        in_features: Number of input features to underlying SIREN model.
        out_features: Number of output features to underlying SIREN model.
        hidden_features: Number of hidden features of the underlying SIREN
            model.
        hidden_layers: Number of hidden layers of the underlying SIREN model.
        epochs: The number of epochs elapsed in the training of the underlying
            SIREN model.
        loss: The average L1 loss of the trained SIREN model.
        training_time: The training time in seconds of the underlying SIREN
            model.
        model_state: The PyTorch model state dict of the SIREN model. This may
            potentially be of use as a way to continue/alter the model
            training.
    """
    def __init__(self,
                 name: str,
                 region: ContinuousRegion,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 epochs: int,
                 loss: float,
                 training_time: float,
                 model_state: Dict) -> None:
        """Constructs a NeuralDensityRegion.

        Args:
            name: Identifer for the region. Used as default save filename.
            region: ContinuousRegion describing the underlying representation.
            in_features: Number of input features to underlying SIREN model.
            out_features: Number of output features to underlying SIREN model.
            hidden_features: Number of hidden features of the underlying SIREN
                model.
            hidden_layers: Number of hidden layers of the underlying SIREN
                model.
            epochs: The number of epochs elapsed in the training of the
                underlying SIREN model.
            loss: The average L1 loss of the trained SIREN model.
            training_time: The training time in seconds of the underlying
                SIREN model.
            model_state: The PyTorch model state dict of the SIREN model. This
                may potentially be of use as a way to continue/alter the model
                training.
        """
        self.name = name
        self.region = region
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.loss = loss
        self.training_time = training_time
        self.model_state = model_state

    def _load_model(self, device: torch.device) -> Siren:
        """Returns the SIREN model loaded onto the specified device.

        Args:
            device: The torch.device the model should be loaded onto. It
                should be pre-constructed before being passed as argument,
                for example: device = torch.device("cuda:0").
        """
        model = Siren(in_features=self.in_features,
                      out_features=self.out_features,
                      hidden_features=self.hidden_features,
                      hidden_layers=self.hidden_layers)
        model.load_state_dict(self.model_state)
        model = model.eval()
        model.to(device)
        return model

    def _to_model_coordinates(self, points: np.ndarray) -> torch.Tensor:
        """Returns a Tensor of the input points converted to model coordinates.

        Model coordinates range between -1 and 1 inclusively corresponding
        to the underlying ConinuousRegion. The output of this function is a
        Tensor that is ready to be used as input to the SIREN model. It
        converts the input numpy array, which is likely float64, to the
        model-supported float32.

        Args:
            points: Array of points with shape (n, 3), where n is the number
                of points.
        """
        ranges = np.array(
            [self.region.x_end - self.region.x_start,
             self.region.y_end - self.region.y_start,
             self.region.z_end - self.region.z_start])
        starts = np.array(
            [self.region.x_start,
             self.region.y_start,
             self.region.z_start])
        return torch.Tensor((2 * ((points - starts) / ranges)) - 1)

    def _postprocess_output_densities(
        self,
        densities: torch.Tensor
    ) -> torch.Tensor:
        """Shifts the nominal range of the Tensor from [-1, 1] to [0, 1].

        The output of the model for density is a value nominally in the range
        [-1, 1]. This method shifts the given values to the nominal range of
        [0, 1].

        Args:
            densities: Tensor whose values need to be normalized.

        Returns:
            Tensor with range shifted to nominal range of [0, 1]. Values may
            still exceed the range if the network produced values outside of
            the nominal range.
        """
        # Shift from [-1, 1] to [0, 1]
        return (densities + 1) / 2

    def get_values(self,
                   points: np.ndarray,
                   device: str = "cuda") -> np.ndarray:
        """Returns the density and gradient vector for the given points.

        The output concatenates the density with the gradient vector for each
        input point. It is assumed that the given points are within the
        region, and the behavior of using points outside the region is
        undefined.

        Args:
            points: Array of points with shape (n, 3). It is assumed the
                points are in the region.
            device: Optional; Device on which to perform nerual network
                operations. If no device names are given, the default CUDA
                device is used. Possible device name options are "cpu",
                "cuda" (for the default CUDA device), or "cuda:x", where x is
                the zero-based index of the CUDA device.

        Returns:
            A numpy array of the shape (n, 4), where n is the number of input
            points and each output n has the following format:
                [density, gradient_x, gradient_y, gradient_z].
            The order of output is preserved with the input's order of points.

        Raises:
            ValueError: If the device is a CUDA device and CUDA is not
            available.
        """
        # Perform check for any input points, return empty array if none
        if points.size == 0:
            return np.zeros((0, 4))

        # Determine the device used for model operations
        if device != "cpu" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, specify 'cpu' as device")
        compute_device = torch.device(device)

        # Load the model onto the device
        model = self._load_model(compute_device)

        # Convert points to model coordinates tensor
        coordinates = self._to_model_coordinates(points).to(compute_device)

        # Break up the coordinates into chunks to prevent overloading the
        # model prediction
        values = np.full((coordinates.shape[0], 4), np.nan, dtype=np.float32)
        max_chunk_size = 300000
        for i in range(0, coordinates.shape[0], max_chunk_size):
            model_input = coordinates[i:i + max_chunk_size]
            model_output, inputs = model(model_input)
            grad_outputs = torch.ones_like(model_output)
            grads = torch.autograd.grad(
                model_output, [inputs], grad_outputs=grad_outputs)[0]
            model_output = self._postprocess_output_densities(model_output)
            values[i:i + max_chunk_size] = torch.cat(
                (model_output, grads), 1).detach().cpu().numpy()
        return values

    def save(self, filename: str = None) -> None:
        """Saves the NeuralDensityRegion to a file.

        This method saves the underlying SIREN creation parameters, training
        metrics, and model state using the PyTorch save method. The
        recommended file extension is ".pt" according to PyTorch docs
        (https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended).

        Args:
            filename: Optional; The filename of the saved NeuralDensityRegion.
                The default filename is "./{self.name}.pt".
        """
        if filename is None:
            filename = f"./{self.name}.pt"

        dir_name = os.path.dirname(filename)
        os.makedirs(dir_name, exist_ok=True)

        save_data = {
            "name": self.name,
            "region": {
                "x_start": self.region.x_start,
                "x_end": self.region.x_end,
                "y_start": self.region.y_start,
                "y_end": self.region.y_end,
                "z_start": self.region.z_start,
                "z_end": self.region.z_end,
            },
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "loss": self.loss,
            "training_time": self.training_time,
            "model_state_dict": self.model_state,
        }
        torch.save(save_data, filename)

    @staticmethod
    def create_from_voxels(name: str,
                           voxels: np.ndarray,
                           region: ContinuousRegion,
                           device_name: str) -> NeuralDensityRegion:
        """Creates a NeuralDensityRegion from voxel data.

        This static initialization method is the normal method for the
        creation of a NeuralDensityRegion. It implements the training of the
        SIREN neural representation and gathering of training metrics.
        Performance is dramatically improved by using a GPU device. The voxel
        data must not extend beyond 64 voxels in each axis and the values must
        be in the range of 0 to 1 inclusive.

        Args:
            name: Identifier for the voxel region.
            voxels: A 3-dimensional array of map voxel data. The maximum size
                per axis is 64 voxels. The voxel data must be within the range
                of [0, 1].
            region: A ContinuousRegion object describing the spatial context of
                the voxel data.
            device_name: Name of the device used to train the SIREN model. It
                must be either "cpu", "cuda" (for the default CUDA device), or
                "cuda:{x}", where x is the zero-based index of the CUDA device.

        Returns:
            A constructed NeuralDensityRegion.

        Raises:
            ValueError: If any of the input voxels axes exceed 64 voxels.
            ValueError: If any voxel value is outside the [0, 1] range
        """
        # Create data loader
        if np.any(np.array(voxels.shape) > 64):
            raise ValueError("Input voxel axis exceeds 64 voxels")
        if np.any(voxels < 0.0) or np.any(voxels > 1.0):
            raise ValueError("Input voxel values must be in the range [0, 1]")
        region_dataset = _VoxelDensityRegionDataset(voxels)
        dataloader = DataLoader(region_dataset)

        # Setup device
        device = torch.device(device_name)

        # Get training data
        model_input, ground_truth = next(iter(dataloader))
        model_input = model_input.to(device)
        ground_truth = ground_truth.to(device)

        # Setup model
        in_features = 3
        out_features = 1
        hidden_features = 256
        hidden_layers = 4
        model = Siren(in_features=in_features,
                      out_features=out_features,
                      hidden_features=hidden_features,
                      hidden_layers=hidden_layers)
        model = model.train()
        model.to(device)

        # Setup training methods
        optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
        loss_fn = torch.nn.MSELoss()

        # Training loop tracks the number of epochs in which the model has not
        # improved overall, and it will exit when the number of stagnant
        # epochs has reached a threshold. There is an automatic training
        # cutoff point when loss reaches min_exit_loss.
        epoch = 0
        max_exit_loss = 0.0004
        min_exit_loss = 0.00001
        max_stagnant_epochs = 25
        stagnant_epochs = 0
        lowest_loss = float('inf')
        best_model_state = dict()
        start_ns = time.time_ns()
        while True:
            # Perform model operation
            model_output, _ = model(model_input)
            loss = loss_fn(model_output, ground_truth)
            loss_value = float(loss)

            # Reset count of stagnant epochs when model is improved
            if loss_value < lowest_loss:
                lowest_loss = loss_value
                best_model_state = model.state_dict()
                stagnant_epochs = 0
            # Count of stagnant epochs is increased if model is not improved
            else:
                stagnant_epochs += 1

            # Check the exit criteria for a properly overfitted network
            if lowest_loss < min_exit_loss:
                break
            if (stagnant_epochs >= max_stagnant_epochs and
                    lowest_loss < max_exit_loss):
                break

            # Update model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch += 1
        end_ns = time.time_ns()
        training_time = (end_ns - start_ns) / 1000000000

        # Transfer model state dict Tensors to CPU for more portable storage
        for k, v in best_model_state.items():
            best_model_state[k] = v.cpu()

        return NeuralDensityRegion(name,
                                   region,
                                   in_features,
                                   out_features,
                                   hidden_features,
                                   hidden_layers,
                                   epoch,
                                   lowest_loss,
                                   training_time,
                                   best_model_state)

    @staticmethod
    def load(filename: str) -> NeuralDensityRegion:
        """Loads and constructs a NeuralDensityRegion from a file.

        This static method loads a NeuralDensityRegion from a file, assuming
        that the file was originally created by the class method save().

        Args:
            filename: Name of file from NeuralDensityRegion.save() method.

        Returns:
            A constructed NeuralDensityRegion.
        """
        load_data = torch.load(filename)
        region = ContinuousRegion(
            x_start=load_data["region"]["x_start"],
            x_end=load_data["region"]["x_end"],
            y_start=load_data["region"]["y_start"],
            y_end=load_data["region"]["y_end"],
            z_start=load_data["region"]["z_start"],
            z_end=load_data["region"]["z_end"])
        return NeuralDensityRegion(load_data["name"],
                                   region,
                                   load_data["in_features"],
                                   load_data["out_features"],
                                   load_data["hidden_features"],
                                   load_data["hidden_layers"],
                                   load_data["epochs"],
                                   load_data["loss"],
                                   load_data["training_time"],
                                   load_data["model_state_dict"])


class _VoxelDensityRegionDataset(Dataset):
    """Implements a PyTorch Dataset using voxel density data.

    This class inherits and implements a PyTorch Dataset such that the voxel
    coordinates and density data may be used by a DataLoader to train a SIREN
    neural representation. The batch size is always 1.
    """
    def __init__(self, density_voxels: np.ndarray) -> None:
        # Create coordinates
        axes = (torch.linspace(-1, 1, steps=density_voxels.shape[0]),
                torch.linspace(-1, 1, steps=density_voxels.shape[1]),
                torch.linspace(-1, 1, steps=density_voxels.shape[2]))
        mgrid = torch.meshgrid(*axes)
        coordinates = torch.stack(mgrid, dim=-1)
        self._coordinates = coordinates.reshape(-1, 3)

        # Create a tensor from the numpy data, making sure it is float32
        densities = torch.as_tensor(density_voxels).float()

        # Assume values are in the range [0, 1] extend the range to [-1, 1]
        densities = (2 * densities) - 1

        # Flatten to match the corresponding coordinates
        self._densities = densities.reshape(-1, 1)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index > 0:
            raise IndexError
        return (self._coordinates, self._densities)
