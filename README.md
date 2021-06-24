[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://travis-ci.org/polyaxon/polyaxon-quick-start.svg?branch=master)](https://travis-ci.org/polyaxon/polyaxon-quick-start)
[![Slack](https://img.shields.io/badge/chat-on%20slack-aadada.svg?logo=slack&longCache=true)](https://polyaxon.com/slack/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg?style=flat)](https://polyaxon.com/docs/)
[![GitHub](https://img.shields.io/badge/issue_tracker-github-blue?logo=github)](https://github.com/polyaxon/polyaxon/issues)
[![GitHub](https://img.shields.io/badge/roadmap-github-blue?logo=github)](https://github.com/polyaxon/polyaxon/milestones)

> N.B. This is a read-only repo, you can create issues or PRs in the [main repo](https://github.com/polyaxon/polyaxon/issues).

# polyaxon-quick-start

A quick start project for polyaxon.

This example is used for the quick start [section in the documentation](https://polyaxon.com/docs/intro/quick-start/). This example also includes different `polyaxonfiles`:

   * A simple polyaxonfile for running the default values in the model.py.
   * A polyaxonfile that defines the inputs and outputs sections to parametrize the model.py.
   * Several polyaxonfiles that define hyperparameter search algorithms.
   * A polyaxonfile with a DAG for automating the complete process.

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

* Using logger: calling getLogger without a name will return the same logger
  used by boto3, thus showing all kinds of S3 debug messages if level is set
  to logging.DEBUG, so always call it with a non-empty name, unless you want
  to invetigate S3 issues.

  Furthermore, experiments in a group are executed apparently in the same
  namespace. Calling getLogger with __name__, will cause the threads to log
  to the same logger.
  For polyaxon logging interface to work, each experiment must have a unique
  logger, so don't re-use any logger from any other threads.
  This can be ensured by hashing the input args,
  as done in the example `model.py`.

