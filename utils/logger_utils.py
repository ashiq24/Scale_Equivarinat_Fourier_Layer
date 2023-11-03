import os
import torch
from datetime import datetime
import pytorch_lightning.loggers as pl_loggers


# Set logger
def set_logger(
    logger, name, params, log_path,
    source_files=False,capture_stdout=False,
    capture_stderr=False,capture_hardware_metrics=False,
    tags=None, project_name="project_name",
    version=None, log_model_checkpoints=False,
):
  if logger == "tb":
    # Set Tensorboard
    ret = pl_loggers.TensorBoardLogger(save_dir=log_path,
                                       name=name,
                                       version=version,
                                       )
  elif logger == "neptune":
    # Set Neptune
    ret = set_neptune(
      name=name,
      params=params,
      tags=tags,
      project_name=project_name,
      version=version,
      source_files=source_files,
      capture_stdout=capture_stdout,
      capture_stderr=capture_stderr,
      capture_hardware_metrics=capture_hardware_metrics,
      log_model_checkpoints=log_model_checkpoints,
    )
  else:
    raise ValueError("Undefined logger. Check 'logger' input argument.")
  return ret


# Set Neptune
def set_neptune(
  name,params,tags,
  version,project_name,source_files,
  capture_stdout,capture_stderr,capture_hardware_metrics,
  log_model_checkpoints,
  ):
  if version:
    # Restore run
    kwargs = {
      "capture_stdout":capture_stdout,
      "capture_stderr":capture_stderr,
      "capture_hardware_metrics":capture_hardware_metrics,
    }
    if not(source_files): kwargs["source_files"] = []

    # Load previous run
    run = pl_loggers.neptune.neptune.init_run(
      project=project_name,
      run=version,
      **kwargs,
    )
    nl = pl_loggers.NeptuneLogger(run=run)

    # Override log_model_checkpoints attribute
    nl._log_model_checkpoints = log_model_checkpoints
  else:
    # Initialize run
    kwargs = {
      "capture_stdout":capture_stdout,
      "capture_stderr":capture_stderr,
      "capture_hardware_metrics":capture_hardware_metrics,
      "log_model_checkpoints":log_model_checkpoints,
    }
    if not(source_files): kwargs["source_files"] = []
    nl = pl_loggers.NeptuneLogger(
      api_key= os.environ["NEPTUNE_API_TOKEN"],
      project=project_name,
      name=name,
      **kwargs,
    )
    nl.log_hyperparams(params)
  return nl


def read_tags(FLAGS):
  _tags = FLAGS.checkpoint.rsplit('/', 5)[-4]
  _tags = _tags.rsplit('-', 3)
  if FLAGS.model is None:
    FLAGS.model = _tags[0]
  if FLAGS.optimizer is None:
    FLAGS.optimizer = _tags[1]
  if FLAGS.pool_method is None:
    FLAGS.pool_method = _tags[2]

  # From dictionary
  cp = torch.load(FLAGS.checkpoint)["hyper_parameters"]
  return


def set_tags(FLAGS, mode="train"):
  assert mode in ["train", "test"]

  tags = [FLAGS.dataset, FLAGS.model]
  if hasattr(FLAGS, "optimizer"):
    tags.append(FLAGS.model)
  tags = [str(k) for k in tags]
  return tags
