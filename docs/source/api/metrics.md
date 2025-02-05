# Metrics

*Use the right sidebar to navigate.*

## Base
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.base

.. autoclass:: Statistic
.. autoclass:: PerVariableStatistic
.. autoclass:: Metric
.. autoclass:: PerVariableMetric
.. autoclass:: PerVariableStatisticWithClimatology
```

## Deterministic
### Statistics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.deterministic

.. autoclass:: Error
.. autoclass:: AbsoluteError
.. autoclass:: SquaredError
.. autoclass:: PredictionPassthrough
.. autoclass:: TargetPassthrough
.. autoclass:: WindVectorSquaredError
.. autoclass:: SquaredPredictionAnomaly
.. autoclass:: SquaredTargetAnomaly
.. autoclass:: AnomalyCovariance
```

### Metrics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.deterministic

.. autoclass:: Bias
.. autoclass:: MAE
.. autoclass:: MSE
.. autoclass:: RMSE
.. autoclass:: PredictionAverage
.. autoclass:: TargetAverage
.. autoclass:: WindVectorRMSE
.. autoclass:: ACC
.. autoclass:: PredictionActivity
```

## Probabilistic
### Statistics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.probabilistic

.. autoclass:: CRPSSkill
.. autoclass:: CRPSSpread
.. autoclass:: EnsembleVariance
.. autoclass:: UnbiasedEnsembleMeanSquaredError
```

### Metrics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.probabilistic

.. autoclass:: CRPSEnsemble
.. autoclass:: UnbiasedEnsembleMeanRMSE
.. autoclass:: SpreadSkillRatio
.. autoclass:: UnbiasedSpreadSkillRatio
```

## Categorical
### Statistics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.categorical

.. autoclass:: TruePositives
.. autoclass:: TrueNegatives
.. autoclass:: FalsePositives
.. autoclass:: FalseNegatives
.. autoclass:: SEEPSStatistic
```

### Metrics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.categorical

.. autoclass:: CSI
.. autoclass:: Accuracy
.. autoclass:: Recall
.. autoclass:: Precision
.. autoclass:: F1Score
.. autoclass:: FrequencyBias
.. autoclass:: SEEPS
```

## Spatial
### Statistics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.spatial

.. autoclass:: SquaredFractionsError
.. autoclass:: SquaredPredictionFraction
.. autoclass:: SquaredTargetFraction
```

### Metrics
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.spatial

.. autoclass:: FSS
```

## Wrappers
```{eval-rst}
.. currentmodule:: weatherbenchX.metrics.wrappers

.. autoclass:: InputTransform
.. autoclass:: EnsembleMean
.. autoclass:: ContinuousToBinary
.. autoclass:: WrappedStatistic
.. autoclass:: WrappedMetric
```
