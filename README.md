[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://travis-ci.org/polyaxon/polyaxon-quick-start.svg?branch=master)](https://travis-ci.org/polyaxon/polyaxon-quick-start)
[![Slack](https://img.shields.io/badge/chat-on%20slack-aadada.svg?logo=slack&longCache=true)](https://join.slack.com/t/polyaxon/shared_invite/enQtMzQ0ODc2MDg1ODc0LWY2ZTdkMTNmZjBlZmRmNjQxYmYwMTBiMDZiMWJhODI2ZTk0MDU4Mjg5YzA5M2NhYzc5ZjhiMjczMDllYmQ2MDg)

# polyaxon-quick-start

A quick start project for polyaxon.

This example is used for the quick start [section in the documentation](https://docs.polyaxon.com/concepts/quick_start/)

> If you are looking for the quick-start example used for v0.4, please go to the [v0.4 branch](https://github.com/polyaxon/polyaxon-quick-start/tree/v0.4) 

This example also includes different `polyaxonfiles`:

   * A simple polyaxonfile for running the default values in the model.py.
   * A polyaxonfile that defines the params in the declaration sections.
   * A polyaxonfile that defines declarations and a matrix, and will generate an experiment group for hyperparameters search.


Cluster Instructions

* Multiple data volumes are mounted including "/data" of node and S3 bucketes,
  use `tracking.get_data_paths` to see all data paths.

  * Access images and mpacks (indexed by future DB) by
    `store.download_file(filename, local_path)`,  maybe will write a wrapper.

* Only one output storage is supported by community edition; S3 is set
  as the default. No need to call `tracking.get_output_path`.

  * Always store output locally in "/outputs" of the container
    with your preferred structure, and upload to S3 when finished
      ```
      experiment.log_artifact(file_path)
      experiment.log_artifacts(dir_path)
      ```
    which are equivalent to deprecated
      ```
      experiment.outputs_store.upload_file(file_path)
      experiment.outputs_store.upload_dir(dir_path)
      ```
    Mimic the log_artifact section in `model.py` rather than uploading the
    `/outputs` directory to avoid an extra level.

* Make sure the script supports CLI arguments by argparse in order to take
  advantage of hyperparameter search as an experiment group.
  Use the params section in the config; the inputs section seems to overlap
  with argparse and is not necessary.