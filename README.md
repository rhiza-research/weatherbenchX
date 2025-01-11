# WeatherBench-X: A modular framework for evaluating weather forecasts

## Why WeatherBench-X?

WeatherBench-X is an evaluation framework that enables flexible evaluation of various kinds of forecast and ground truth data, including sparse datasets like those coming from weather stations or satellites. It is the successor to the [WeatherBench 2 evaluation code](https://github.com/google-research/weatherbench2). However, the [WeatherBench 2 benchmark](https://sites.research.google/weatherbench/) remains active (and is now also powered by the WeatherBench-X evaluation framework).

WeatherBench related datasets can be found [here](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).

The core design principles behind WeatherBench-X are:
- Modularity: Data loaders, Interpolations, Metrics and the Aggregation can be defined through interoperable classes.
- Xarray: All internal logic is based on xarray DataArrays.
- Scalability: Each operation can be split into small chunks allowing scalable evaluation of very large datasets through Apache Beam.

To get started using WeatherBench-X, check out the quickstart notebook.

## License

This is not an officially supported Google product.

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
