import os
import logging
from pathlib import Path
import pytorch_lightning as pl
LOG = logging.getLogger(__name__)


class AutoResumeState(pl.Callback):
  """Maintains a temporary file to keep track across
     run if terminated before training ends naturally."""

  def __init__(
      self, checkpoint_cb, resume_cp=None,
      enabled=False, max_epochs=None, single_epoch_mode=False,
      state_file='.train_incomplete', **kwargs,
  ):
    super().__init__()
    LOG.info("Initializing autoresume")
    self._resume_checkpoint_file = None
    self.mc = checkpoint_cb
    self.enabled = enabled
    self.checkpoints = []
    self.checkpoint_dir = None
    self.max_epochs = max_epochs
    self.single_epoch_mode = single_epoch_mode
    self.resume_cp = resume_cp

    if enabled:
      LOG.info(
          f"Autoresume enabled, checkpoint state will be saved in {state_file}")
      self.state_file = state_file
      if os.path.isfile(self.state_file):
        with open(self.state_file, 'r') as f:
          self.checkpoint_dir = f.readline()
          LOG.info(f'Previous checkpoints at {self.checkpoint_dir}')
          self.checkpoints = sorted(
              Path(self.checkpoint_dir).glob("*.ckpt"),
              key=os.path.getmtime,
          )
          LOG.info(f"Found {len(self.checkpoints)} checkpoints")
          if len(self.checkpoints) > 0:
            LOG.info(f"Newest {self.checkpoints[-1]}")
      else:
        LOG.info(f"Previous session does not exist. Trainer will start from scratch")

    super().__init__(**kwargs)

  def on_train_epoch_end(self, trainer, pl_module):
    if self.enabled:
      with open(self.state_file, 'w') as f:
        f.write(self.mc.dirpath)
        LOG.debug(
            f"Logged last checkpoint {self.mc.dirpath} in {self.state_file}")
      # Only keep the latest 5 and best checkpoint.
      if os.path.isdir(self.mc.dirpath) and trainer.global_rank == 0:
        all_ckpts = os.listdir(self.mc.dirpath)
        top_k_num = 5
        if len(all_ckpts) < top_k_num:
          return
        else:
          top_k = []
          for key in self.mc.best_k_models:
            top_k.append((float(self.mc.best_k_models[key]), key))
          # Sorts in asscending order.
          top_k.sort()
          LOG.info(
              "on_train_epoch_end. Deleting old checkpoints (assuming metric to be higher the better)")
          for curr_ckpt in all_ckpts:
            curr_ckpt_path = "%s/%s" % (self.mc.dirpath, curr_ckpt)
            if curr_ckpt_path not in [tmp[1] for tmp in top_k[-top_k_num:]]:
              os.remove(curr_ckpt_path)

  def on_fit_end(self, trainer, pl_module):
    if self.enabled:
      if self.single_epoch_mode and trainer.current_epoch < (self.max_epochs):
        return
      LOG.info("trainer.fit() finished normally. Removing state file")
      if os.path.isfile(self.state_file) and trainer.global_rank == 0:
        os.remove(self.state_file)
        LOG.info(f"{self.state_file} removed")
      else:
        LOG.info(f"{self.state_file} not found. If in DDP mode, it may have been removed "
                 "by another process already.")

  def get_epoch(self):
    # Read epoch from cp filename
    epoch = str(self.checkpoints[-1]
                ).split('/')[-1].split('-')[0]  # 'epoch=xxxx'
    epoch = int(epoch.split('=')[-1])
    return epoch

  @property
  def resume_checkpoint_file(self):
    # Checkpoints[-1] state has priority over resume_cp state
    if self.enabled and len(self.checkpoints) > 0:
      # Load latest state associated to .train_incomplete
      return self.checkpoints[-1]
    elif self.resume_cp is not None:
      # Load resume_cp state
      return self.resume_cp
    return None

  @property
  def resume_version(self):
    # Checkpoints[-1] state has priority over resume_cp state
    if self.checkpoint_dir:
      # Autoresume: Load version from Checkpoints[-1]
      _d_path = self.checkpoint_dir.split('/')
      for p in _d_path:
        if 'version' in p:
          version = int(p.split('_')[-1])
          LOG.info(f"Resuming version {version}")
          return version
        elif 'SEQT' in p:
          LOG.info(f"Resuming Neptune experiment_id {p}")
          return p
    elif self.resume_cp is not None:
      # TODO: Generate new version based on resume_cp
      #version = self.resume_cp.rsplit('/')[-3] + "-R"
      # return version
      pass


class OneEpochStop(pl.Callback):
  def __init__(self, num_epoch_stop=1):
    super().__init__()
    self.num_epoch_stop = num_epoch_stop

  def on_validation_end(self, trainer, pl_module):
    if (trainer.current_epoch+1) % self.num_epoch_stop == 0:
      trainer.should_stop = True
