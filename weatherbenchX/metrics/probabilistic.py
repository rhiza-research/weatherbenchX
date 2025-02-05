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
"""Implementation of probabilistic metrics and assiciated statistics."""

from typing import Hashable, Mapping
import numpy as np
from weatherbenchX.metrics import base
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import wrappers
import xarray as xr

### Statistics


class CRPSSkill(base.PerVariableStatistic):
  """The skill measure associated with CRPS, E|X - Y|."""

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'CRPSSkill_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return np.abs(predictions - targets).mean(
        self._ensemble_dim, skipna=self._skipna_ensemble
    )


def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
  """Version of (ordinal) scipy.rankdata from V13."""
  x = np.asarray(x)
  x = np.swapaxes(x, axis, -1)
  j = np.argsort(x, axis=-1)
  ordinal_ranks = np.broadcast_to(
      np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
  )
  ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
  np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
  return np.swapaxes(ordered_ranks, axis, -1)


def _rank_da(da: xr.DataArray, dim: str) -> np.ndarray:
  return da.copy(data=_rankdata(da.values, axis=da.dims.index(dim)))


class CRPSSpread(base.PerVariableStatistic):
  """The spread measure associated with CRPS, E|X - X`|."""

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'CRPSSpread_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    n_ensemble = predictions.sizes[self._ensemble_dim]
    if n_ensemble < 2:  # CRPS equates to MAE in case of n_ensemble == 1.
      return xr.zeros_like(predictions.isel({self._ensemble_dim: 0}, drop=True))

    # one_half_spread is ̂̂λ₂ from Zamo. That is, with n_ensemble = M,
    #   λ₂ = 1 / (2 M (M - 1)) Σ_{i,j=1}^M |Xi - Xj|
    # See the definition of eFAIR and then
    # eqn 3 (appendix B), which shows that this double summation of absolute
    # differences can be written as a sum involving sorted elements multiplied
    # by their index. That is, if X1 < X2 < ... < XM,
    #   λ₂ = 1 / (M(M-1)) Σ_{i,j=1}^M (2*i - M - 1) Xi.
    # The term (2*i - M - 1) is +1 times the number of terms Xi is greater than,
    # and -1 times the number of terms Xi is less than.
    # Here we do not sort but instead compute the rank of each element, multiply
    # appropriately, then sum. We prefer this second form, since it involves an
    # O(M Log[M]) compute and O(M) memory usage, whereas the first is O(M²) in
    # compute and memory.
    rank = _rank_da(predictions, self._ensemble_dim)
    return (
        2
        * (
            ((2 * rank - n_ensemble - 1) * predictions).mean(
                self._ensemble_dim, skipna=self._skipna_ensemble
            )
        )
        / (n_ensemble - 1)
    )


class EnsembleVariance(base.PerVariableStatistic):
  """Computes the variance in the ensemble dimension."""

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'EnsembleVariance_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return predictions.var(
        dim=self._ensemble_dim, ddof=1, skipna=self._skipna_ensemble
    )


class UnbiasedEnsembleMeanSquaredError(base.PerVariableStatistic):
  """Computes the unbiased ensemble mean squared error.

  This class estimates E(X - Y)² with no bias. This is done by subtracting the
  sample variance divided by n. As such, you must have n > 1 or the result will
  be NaN.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'UnbiasedEnsembleMeanSquaredError_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    unbiased_variance = predictions.var(
        dim=self._ensemble_dim, ddof=1, skipna=self._skipna_ensemble
    )
    predictions_mean = predictions.mean(
        dim=self._ensemble_dim, skipna=self._skipna_ensemble
    )
    biased_mse = (predictions_mean - targets) ** 2
    n_ensemble = predictions.sizes[self._ensemble_dim]
    return biased_mse - unbiased_variance / n_ensemble


### Metrics


class CRPSEnsemble(base.PerVariableMetric):
  """Continuous ranked probabilisty score for an ensemble prediction.

  Given ground truth scalar random variable Y, and two iid predictions X, X`,
  the Continuously Ranked Probability Score is defined as
    CRPS = E|X - Y| - 0.5 * E|X - X`|
  where `E` is mathematical expectation, and | ⋅ | is the absolute value. CRPS
  has a unique minimum when X is distributed the same as Y.

  If N ensemble members are available, the ensemble mean is taken using the PWM
  method from [Zamo & Naveau, 2018].

  So long as 2 or more ensemble members are used, the estimates of spread, skill
  and CRPS are unbiased at each time.

  References:

  - [Gneiting & Raftery, 2012], Strictly Proper Scoring Rules, Prediction, and
    Estimation
  - [Zamo & Naveau, 2018], Estimation of the Continuous Ranked Probability Score
    with Limited Information and Applications to Ensemble Weather Forecasts.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, ensemble members with NaN values will be ignored
        in the ensemble mean computations. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['CRPSSkill'] - 0.5 * statistic_values['CRPSSpread']


class UnbiasedEnsembleMeanRMSE(base.PerVariableMetric):
  """Unbiased estimate of the ensemble mean RMSE."""

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'UnbiasedEnsembleMeanSquaredError': UnbiasedEnsembleMeanSquaredError()
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(statistic_values['UnbiasedEnsembleMeanSquaredError'])


class SpreadSkillRatio(base.PerVariableMetric):
  """Computes the (biased) spread-skill ratio.

  The spread skill ratio is defined as the ensemble standard deviation divided
  by the RMSE of the ensemble mean.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, ensemble members with NaN values will be ignored
        in the ensemble mean computations. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'EnsembleVariance': EnsembleVariance(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'EnsembleMeanSquaredError': wrappers.WrappedStatistic(
            deterministic.SquaredError(),
            wrappers.EnsembleMean(
                which='predictions',
                ensemble_dim=self._ensemble_dim,
                skipna=self._skipna_ensemble,
            ),
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(
        statistic_values['EnsembleMeanSquaredError']
        / statistic_values['EnsembleVariance']
    )


class UnbiasedSpreadSkillRatio(base.PerVariableMetric):
  """Computes the spread-skill ratio based on the unbiased skill estimator.

  This is analogous to the regular spread skill ratio but using the unbiased
  estimator of the ensemble mean squared error. This is useful for estimating
  the spread skill ratio for differing ensemble sizes.

  Note that the ratio and square root are still biased, however, this is
  negligible if the number of time points is large.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, ensemble members with NaN values will be ignored
        in the ensemble mean computations. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'EnsembleVariance': EnsembleVariance(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'UnbiasedEnsembleMeanSquaredError': UnbiasedEnsembleMeanSquaredError(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(
        statistic_values['UnbiasedEnsembleMeanSquaredError']
        / statistic_values['EnsembleVariance']
    )
