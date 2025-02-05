# Running distributed evaluation jobs on GCP

To evaluate large datasets, distributed computing is necessary. WeatherBench-X uses Apache Beam to build distributed workflows. These can then be run on Google Cloud using [Dataflow](https://cloud.google.com/dataflow). Here is a quick guide to running Beam jobs.

## Local execution
For testing or very small dataset, it makes sense to run the Beam pipeline locally. For this, chose the `DirectRunner` as your Beam runner. You can also set how many local works to use at the same time:
```
python run_example_evaluation.py \
  --prediction_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --target_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --time_start=2020-01-01 \
  --time_stop=2020-01-02 \
  --output_path=./results.nc \
  --runner=DirectRunner \
  -- \
  --direct_num_workers 2
```
For a full list of how to configure the direct runner, please review [this page](https://beam.apache.org/documentation/runners/direct/).

## Dataflow execution

To run Beam pipelines on the cloud, we are using Google Cloud Dataflow. For this you will need an active Google Cloud account.

To run on Google Cloud Dataflow, use the `DataflowRunner`. Additionally, a few parameters need to be specified.

* `--runner`: The `PipelineRunner` to use. This field can be either `DirectRunner` or `DataflowRunner`.
  Default: `DirectRunner` (local mode)
* `--project`: The project ID for your Google Cloud Project. This is required if you want to run your pipeline using the
  Dataflow managed service (i.e. `DataflowRunner`).
* `--temp_location`: Cloud Storage path for temporary files. Must be a valid Cloud Storage URL, beginning with `gs://`.
* `--region`: Specifies a regional endpoint for deploying your Dataflow jobs. Default: `us-central1`.
* `--job_name`: The name of the Dataflow job being executed as it appears in Dataflow's jobs list and job details.
* `--setup_file`: To make sure all the required packages are installed on the workers, pass the `setup.py` file to GCP.

Example:
```
export BUCKET=<your-bucket>
export PROJECT=<your-project>
export REGION=us-central1

python run_example_evaluation.py \
  --prediction_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --target_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --time_start=2020-01-01 \
  --time_stop=2020-01-02 \
  --output_path=gs://$BUCKET/results.nc \
  --runner=DataflowRunner \
  -- \
  --project=$PROJECT \
  --region=$REGION \
  --temp_location=gs://$BUCKET/tmp/ \
  --setup_file=../setup.py \
  --job_name=wbx-eval
```

For a full list of how to configure the Dataflow pipeline, please review
[this table](https://cloud.google.com/dataflow/docs/reference/pipeline-options).

When running Dataflow, you
can [monitor jobs through UI](https://cloud.google.com/dataflow/docs/guides/using-monitoring-intf),
or [via Dataflow's CLI commands](https://cloud.google.com/dataflow/docs/guides/using-command-line-intf):

For example, to see all outstanding Dataflow jobs, simply run:

```shell
gcloud dataflow jobs list
```

To describe stats about a particular Dataflow job, run:

```shell
gcloud dataflow jobs describe $JOBID
```

In addition, Dataflow provides a series
of [Beta CLI commands](https://cloud.google.com/sdk/gcloud/reference/beta/dataflow).

These can be used to keep track of job metrics, like so:

```shell
JOBID=<enter job id here>
gcloud beta dataflow metrics list $JOBID --source=user
```

You can even [view logs via the beta commands](https://cloud.google.com/sdk/gcloud/reference/beta/dataflow/logs/list):

```shell
gcloud beta dataflow logs list $JOBID
```