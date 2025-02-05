# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Definition of aggregation methods and AggregationState."""

import dataclasses
import math
from typing import Collection, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from weatherbenchX import binning
from weatherbenchX import weighting
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


def _combining_sum(
    data_arrays: Sequence[Optional[xr.DataArray]],
) -> Optional[xr.DataArray]:
  """A sum which combines / aligns coordinates in case they don't match.

  It's semantically equivalent to

  sum(xarray.align(data_arrays, join='outer', fill_value=0))

  but when all summands have the same coordinates it's just a plain sum.

  Args:
    data_arrays: To be summed/combined. All must all have the same set of
      dimensions and must have coordinates for all dimensions so we can be sure
      we have aligned them correctly. None values are allowed and will be
      dropped from the sum (treated as empty/zero).

  Returns:
    A data_array whose index coordinates along each dimension are the union of
    the index coordinates of all the arguments along that dimension, or None
    if there were no non-None arguments.
  """
  # Arrays for individual statistics can be None if an AggregationMethod wasn't
  # able to handle that statistic (e.g. aggregation.Aggregator will return None
  # for statistics that don't contain the requested preserve_dims).
  data_arrays = [s for s in data_arrays if s is not None]
  if not data_arrays:
    return None

  if len(data_arrays) == 1:
    return data_arrays[0]

  dims = set(data_arrays[0].dims)
  for a in data_arrays[1:]:
    if set(a.dims) != dims:
      raise ValueError(
          f'Different dims encountered by _combining_sum: {a.dims} vs {dims}.'
      )

  for a in data_arrays:
    for dim in dims:
      if dim not in a.coords:
        raise ValueError(
            'All dimensions must have coordinates to ensure alignment when '
            f'summing statistics, but dimemsion {dim} lacked coordinates.'
        )

  # Fast path when index coordinates are all the same.
  with xr.set_options(arithmetic_join='exact'):
    try:
      return sum(data_arrays[1:], start=data_arrays[0])
    except ValueError:
      # Coordinates were not exactly aligned.
      pass

  # Potentially-slow but general path, the other paths above do the same thing
  # as this but may be faster.
  # This will extend each array to use the union of all the coordinates, padding
  # with zeros for any missing coordinates, and only sum after padding each
  # array. As such it may be quadratic in len(data_arrays) in the worst case.
  data_arrays = xr.align(*data_arrays, join='outer', fill_value=0, copy=False)
  return sum(data_arrays[1:], start=data_arrays[0])


def _fast_dot(
    a: xr.DataArray, b: xr.DataArray, reduce_dims: set[str]
) -> xr.DataArray:
  """Implementation of dot product semantically almost equivalent to xr.dot.

  Difference one is that this will only work if all reduce dims are present in
  both arrays. xr.dot will also sum out any reduce dims present in only one of
  the arrays. Here this is already taken care of in the calling function, so
  an error would be raised if this is not the case.

  Difference two is that this implementation requires that all common dims
  between a and b are also part of reduce_dims. In other words, any binning
  dimension must also be reduced.

  Args:
    a: First array.
    b: Second array.
    reduce_dims: Set of dimensions to reduce over.

  Returns:
    The dot product of a and b.
  """
  # If data is empty, reshaping array below will fail. In this case, we can just
  # use the xarray implementation.
  if a.size == 0 or b.size == 0:
    return xr.dot(a, b, dim=reduce_dims)

  assert (
      not set(a.dims).intersection(set(b.dims)) - reduce_dims
  ), 'Not all common dims between a and b are also part of reduce_dims.'

  def reshape_data(x):
    """Transformations to apply to both arrays."""
    array_dims = set(x.dims)
    assert reduce_dims.issubset(
        array_dims
    ), 'Not all reduce_dims are present in the array.'
    non_reduce_dims = sorted(list(array_dims - reduce_dims))
    coords = {c: x.coords[c] for c in non_reduce_dims}
    ordered_dims = non_reduce_dims + sorted(list(reduce_dims))

    # Make sure we are also preserving non-dimension coordinates. But only if
    # they don't contain any of the reduce_dims.
    other_coords = {
        c: x.coords[c]
        for c in x.coords
        if c not in x.dims
        and not set(x.coords[c].dims) - set(non_reduce_dims)  # Must be empty.
    }
    len_dims = [x.sizes[c] for c in non_reduce_dims]
    x = x.transpose(*ordered_dims)
    # Switch to numpy for reshaping since this is faster than xarray's stack.
    # Array should how have two dimensions with shape:
    # (product(non_reduce_dims), product(reduce_dims)).
    x = x.values.reshape(math.prod(len_dims), -1)
    return x, non_reduce_dims, coords, other_coords, len_dims

  a, non_reduce_dims_a, coords_a, other_coords_a, len_dims_a = reshape_data(a)
  b, non_reduce_dims_b, coords_b, other_coords_b, len_dims_b = reshape_data(b)
  len_dims_out = len_dims_a + len_dims_b

  out = np.dot(a, b.T)
  out = out.reshape(len_dims_out)
  out = xr.DataArray(
      out,
      dims=non_reduce_dims_a + non_reduce_dims_b,
      coords=coords_a | coords_b | other_coords_a | other_coords_b,
  )
  return out


@dataclasses.dataclass
class AggregationState:
  """An object that contains sum of weighted statistics and sum of weights.

  Allows for aggregation over multiple chunks, e.g. in a Beam pipeline.

  Attributes:
    sum_weighted_statistics: Structure containing summed/aggregated statistics.
    sum_weights: Structure containing the corresponding summed weights.
  """

  sum_weighted_statistics: Optional[
      Mapping[str, Mapping[Hashable, xr.DataArray]]
  ]
  sum_weights: Optional[Mapping[str, Mapping[Hashable, xr.DataArray]]]

  @classmethod
  def zero(cls) -> 'AggregationState':
    """An initial/'zero' aggregation state."""
    return cls(sum_weighted_statistics=None, sum_weights=None)

  def __add__(self, other: 'AggregationState') -> 'AggregationState':
    return self.sum([self, other])

  @classmethod
  def sum(
      cls, aggregation_states: list['AggregationState']
  ) -> 'AggregationState':
    """Sum of aggregation states."""
    sum_weighted_statistics_and_sum_weights_tuples = [
        (a.sum_weighted_statistics, a.sum_weights)
        for a in aggregation_states
        if a.sum_weighted_statistics is not None
    ]

    # Sometimes beam does a reduction with only Zero states. In this case, we
    # end up with an empty collection. In these cases, we need to return a zero
    # state.
    if not sum_weighted_statistics_and_sum_weights_tuples:
      return cls.zero()

    # Sum over each element in the nested dictionaries
    sum_weighted_statistics, sum_weights = xarray_tree.map_structure(
        lambda *a: _combining_sum(a),
        *sum_weighted_statistics_and_sum_weights_tuples,
    )

    return cls(sum_weighted_statistics, sum_weights)

  def mean_statistics(self) -> Mapping[str, Mapping[Hashable, xr.DataArray]]:
    """Returns the statistics normalized by their corresponding weights."""

    def normalize(sum_weighted_statistics, sum_weights):
      return sum_weighted_statistics / sum_weights

    return xarray_tree.map_structure(
        normalize, self.sum_weighted_statistics, self.sum_weights
    )

  def metric_values(
      self, metrics: Mapping[Hashable, metrics_base.Metric]
  ) -> xr.Dataset:
    """Returns metrics computed from the normalized statistics.

    Args:
      metrics: Dictionary of metric names and instances.

    Returns:
      values: Combined dataset with naming convention <metric>.<variable>
    """

    mean_statistics = self.mean_statistics()
    values = xr.Dataset()
    for metric_name, metric in metrics.items():
      values_for_metric = metric.values_from_mean_statistics(mean_statistics)
      for var_name, da in values_for_metric.items():
        values[f'{metric_name}.{var_name}'] = da
    return values


@dataclasses.dataclass
class Aggregator:
  """Defines aggregation over set of dataset dimensions.

  Note on NaNs: By default, all reductions are performed with skipna=False,
  meaning that the aggregated statistics will be NaN if any of the input
  statistics are NaN. Currently, there is one awkward use case, where even if
  the input NaNs are outside the binning mask, e.g. if NaNs appear in a
  different region from the binning region, the aggregated statistics will
  still be NaN. Use the masking option to avoid this.

  Attributes:
    reduce_dims: Dimensions to average over. Any variables that don't have these
      dimensions will be filtered out during aggregation.
    bin_by: List of binning instances. All bins will be multiplied.
    weigh_by: List of weighting instance. All weights will be multiplied.
    masked: If True, aggregation will only be performed for non-masked (True on
      the mask) values. This requires a 'mask' coordinate on the statistics
      passed to aggregate_statistics.
    skipna: If True, NaNs will be omitted in the aggregation. This option is not
      recommended, as it won't catch unexpected NaNs.
  """

  reduce_dims: Collection[str]
  bin_by: Sequence[binning.Binning] | None = None
  weigh_by: Sequence[weighting.Weighting] | None = None
  masked: bool = False
  skipna: bool = False

  def aggregation_fn(
      self,
      stat: xr.DataArray,
  ) -> xr.DataArray | None:
    """Returns the aggregation function."""
    # Recall that masked out values have already been set to zero in
    # aggregate_statistics. The logic below has to respect this.

    # Take product of all weights and bins.
    if self.weigh_by is not None:
      weights = math.prod(
          [weighting_method.weights(stat) for weighting_method in self.weigh_by]
      )
      stat = stat * weights

    reduce_dims_set = set(self.reduce_dims)
    eval_unit_dims = set(stat.dims)
    if not reduce_dims_set.issubset(eval_unit_dims):
      # Can't reduce over dims that aren't present as evaluation unit dims.
      return None

    if self.bin_by is not None:
      bin_masks = xr.DataArray(
          math.prod([
              binning_method.create_bin_mask(stat)
              for binning_method in self.bin_by
          ])
      )
      bin_dim_names = set([binning.bin_dim_name for binning in self.bin_by])
      if len(bin_dim_names) != len(self.bin_by):
        raise ValueError('Bin dimension names must be unique.')

      # bin_masks_dims are all of the dims the mask operate with on the input
      # data (e.g. the actual bin dimension does not count).
      bin_masks_dims = set(bin_masks.dims) - set(bin_dim_names)

      if not bin_masks_dims.issubset(eval_unit_dims):
        # Can't bin based on dims that aren't present as evaluation unit dims:
        return None

      # These dimensions don't need preserving, and are also not explicitly
      # used by the masks, so we just sum them first.
      non_bin_index_reduce_dims = reduce_dims_set - bin_masks_dims
      stat = stat.sum(non_bin_index_reduce_dims, skipna=False)

      # Finally we compute the element-wise product, reducing only across
      # the bin masks dimensions that we are not preserving. The
      # bin_dim_names dimensions will always be preserved.
      bin_index_reduce_dims = reduce_dims_set - non_bin_index_reduce_dims
      binned_data = _fast_dot(bin_masks, stat, bin_index_reduce_dims)

      return binned_data

    else:  # Simple sum when no binning is applied.
      return stat.sum(reduce_dims_set, skipna=False)

  def aggregate_statistics(
      self,
      statistics: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> AggregationState:
    """Aggregate all statistics for a batch.

    Args:
      statistics: Full statistics for a batch.

    Returns:
      AggregationState instance with a sum of weighted statistics and a sum of
      weights for the current batch. These can be summed over multiple batches,
      and then used to compute weighted mean statistics, and from these the
      final values of the metrics.
    """

    # Different aggregator for each variable
    def batch_aggregator_for_var_and_stat(stat):
      if self.skipna:
        # Set NaNs to zero, so that they will be ignored in the sum.
        stat = stat.where(~stat.isnull(), 0)

      if self.masked and hasattr(stat, 'mask'):
        # Set masked values to Zero for stat and weights, which will therefore
        # be ignored in mean_statistics(). this is equivalent to multiplying by
        # the mask, but avoids NaN * 0 -> NaN in cases where there are NaNs in
        # masked positions. Only for variables with a mask attribute.
        if hasattr(stat, 'mask'):
          stat = stat.where(stat.mask, 0)

      return self.aggregation_fn(stat)

    def batch_aggregator_weights_for_var_and_stat(stat):
      ones = xr.ones_like(stat)
      # Make sure the weights are also zero for skipna and masked aggregation.
      if self.skipna:
        ones = ones.where(~stat.isnull(), 0)
      if self.masked and hasattr(stat, 'mask'):
        ones = ones.where(stat.mask, 0)
      return batch_aggregator_for_var_and_stat(ones)

    def filter_nones(x):
      result = {}
      for name, values in x.items():
        if isinstance(values, xr.Dataset):
          # Dataset has already had None's filtered out by xarray_tree,
          # but we want to preserve its type:
          result[name] = values
        else:
          result[name] = {k: v for k, v in values.items() if v is not None}
      return result

    sum_weighted_statistics = filter_nones(
        xarray_tree.map_structure(batch_aggregator_for_var_and_stat, statistics)
    )
    sum_weights = filter_nones(
        xarray_tree.map_structure(
            batch_aggregator_weights_for_var_and_stat, statistics
        )
    )

    # Aggregator for every dataset in statistics
    return AggregationState(sum_weighted_statistics, sum_weights)


def compute_metric_values_for_single_chunk(
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: Aggregator,
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
  """Convenience function to compute metric results for a given predictions/targets pair.
  
  This is not intended to accumulate over multiple chunks.

  Args:
    metrics: Dictionary of metrics instances.
    aggregator: Aggregator instance.
    predictions: Xarray Dataset or dictionary of DataArrays.
    targets: Xarray Dataset or dictionary of DataArrays.

  Returns:
    results: Xarray Dataset of metric values.
  """
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions, targets
  )
  aggregation_state = aggregator.aggregate_statistics(statistics)
  results = aggregation_state.metric_values(metrics)
  return results
