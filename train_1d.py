import os
import pytorch_lightning as pl
import lightning_fabric as lf
from absl import app, flags, logging
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from models.fourier_seq_mnist import FourierSeqMist
import torchvision.transforms as T
from utils.core_utils import AddScaling
from callbacks import AutoResumeState, OneEpochStop
import clargs.train_opts
import clargs.data_opts
import clargs.logger_opts
from dataloader import get_pl_datamodule, get_available_pl_modules
from models import get_model, get_available_models
from utils.logger_utils import set_logger
from torch.optim.lr_scheduler import StepLR
from tests.test_equievarience import TestEquivarinecError, GetAccuracy
import numpy as np
from layers.complex_modules import get_activation
from dataloader.mnist_data_module import get_augmentation
AVAILABLE_DATASETS = get_available_pl_modules()
AVAILABLE_MODELS = get_available_models()

# Dataset
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data', 'Dataset location')
flags.DEFINE_enum('dataset', 'yesno',AVAILABLE_DATASETS, 'Dataset')


flags.DEFINE_string('model', 'fourier_1d', 'Model names: fourier_1d')
flags.DEFINE_integer('input_channel', 1, 'number of input Channel')
flags.DEFINE_integer('C1', 32, 'Width/number of channel in 1nd Convolutional layer')
flags.DEFINE_integer('C2', 64, 'Width/number of channel in 2nd Convolutional layer')
flags.DEFINE_integer('C3', 100, 'Width/number of channel in 3rd Convolutional layer')
flags.DEFINE_integer('C4', 100, 'Width/number of channel in 4th Convolutional layer')
flags.DEFINE_integer('mixer_band', 3, 'width of window for frequency mixer')

flags.DEFINE_integer('FC1', 256, 'Width of 1st Fully connected NN following the Convolutional Layer')
flags.DEFINE_float('dropout_fc1', 0.0, 'Dropout in the input of the 1st layer of FNN')
flags.DEFINE_float('dropout_fc2', 0.75, 'Dropout in the input of the 2nd layer of FNN')
flags.DEFINE_string('activation_cov', 'relu', 'Activation after Convolutional layers')
flags.DEFINE_string('activation_mlp', 'relu', 'Activation after MLP layers')
flags.DEFINE_string('normalizer', 'instance', 'Type of normlaization to use.')




flags.DEFINE_integer('batch_size', 1, 'Batch size') #dataset yes has different sizes, so batch_size should be 1
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate')
flags.DEFINE_float('weight_decay', 3e-2, 'Weight Decay')
flags.DEFINE_integer('epochs', 60, 'Training epochs')
flags.DEFINE_integer('step_size_lr', 20, 'Step size of lr scheduler')
flags.DEFINE_float('step_gamma_lr', 0.1, 'gamma for lr scheduler')


# Misc.
flags.DEFINE_integer(
    'seed', 42, 'Random seed')
flags.DEFINE_boolean(
    'dryrun', False, 'Run sanity check only on 1 batch of each split')
flags.DEFINE_boolean('autoresume', False,
                     'enables autoresume from last checkpoint if previous run was incomplete')
flags.DEFINE_string('autoresume_statefile', '.train_incomplete',
                    'State file used by autoresume feature; Must be unqiue per job.')
flags.DEFINE_boolean(
    'oneepoch', False, 'Stops training after K epoch regardless of max epochs')
flags.DEFINE_integer('oneepoch_k', 1, 'The K interger for oneepoch')
flags.DEFINE_string('resume_cp',None,'Checkpoint path to resume training.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    print('Unprocessed args:', argv[1:])
  gpu_device = int(argv[1])
  print("Running on GPU ", gpu_device)

  # Model checkpoint
  mc = pl.callbacks.ModelCheckpoint(
      filename='{epoch}-{val_acc:.5f}',
      monitor='val_acc',
      # 'Save_last' disabled for autoresume
      save_last=not (FLAGS.autoresume),
      # Save all checkpoints for autoresume
      # to guarantee last epoch is saved
      save_top_k=-1 if FLAGS.autoresume else 1,
      mode='max',
  )
  
  # Autoresume
  ar_cb = AutoResumeState(
      checkpoint_cb=mc,
      resume_cp=FLAGS.resume_cp,
      enabled=FLAGS.autoresume,
      max_epochs=FLAGS.epochs,
      single_epoch_mode=FLAGS.oneepoch,
      state_file=FLAGS.autoresume_statefile,
  )
  print("ar_cb.resume_version: {s}".format(s=ar_cb.resume_version))

  # Logging
  log_path = os.path.join(FLAGS.log_path, FLAGS.logger)
  name = f'{FLAGS.model}'
  params = {"FLAGS": FLAGS.flag_values_dict()}

  logger = set_logger(
      logger=FLAGS.logger,
      log_path=log_path,
      name=name,
      params=params,
      project_name='ashiq/scale-eq-public',
      version=None,
      source_files=FLAGS.source_files,
      capture_stdout=FLAGS.capture_stdout,
      capture_stderr=FLAGS.capture_stderr,
      capture_hardware_metrics=FLAGS.capture_hardware_metrics,
      log_model_checkpoints=FLAGS.log_model_checkpoints,
  )
  
  # Seed
  _seed = FLAGS.seed if ar_cb.checkpoint_dir is None else\
    FLAGS.seed+ar_cb.get_epoch()+1
  lf.utilities.seed.seed_everything(_seed)
  
  # Lr monitor
  lr_monitor = LearningRateMonitor(
    logging_interval='step',
    log_momentum=True,
  )
  
  # Callbacks
  cb = [mc,lr_monitor,ar_cb] 
  if FLAGS.oneepoch:
    cb.append(OneEpochStop(FLAGS.oneepoch_k))


  # Dataset
  dm = get_pl_datamodule(FLAGS.dataset)(
      data_dir=FLAGS.data_path,
      batch_size=FLAGS.batch_size)

  dm.prepare_data()
  dm.setup()

  activation_cov = get_activation(FLAGS.activation_cov)
  activation_mlp = get_activation(FLAGS.activation_mlp)

  model = get_model(FLAGS.model)(1, learning_rate = FLAGS.learning_rate, weight_decay = FLAGS.weight_decay,\
                                C1 =FLAGS.C1, C2 = FLAGS.C2, C3 = FLAGS.C3, C4 = FLAGS.C4, FC1 = FLAGS.FC1,\
                                dropout_fc1 = FLAGS.dropout_fc1,dropout_fc2 = FLAGS.dropout_fc2, activation_con = activation_cov,\
                                activation_mlp = activation_mlp,mixer_band = FLAGS.mixer_band,scheduler = StepLR, normalizer = 'instance',\
                                scheduler_kwargs={'step_size':FLAGS.step_size_lr, 'gamma':FLAGS.step_gamma_lr} )


  trainer = pl.Trainer(
      accelerator=FLAGS.accelerator,
      benchmark=True,
      callbacks=cb,
      default_root_dir=log_path,
      devices= [gpu_device],
      fast_dev_run=FLAGS.dryrun,
      val_check_interval = None,
      check_val_every_n_epoch= 2,
      logger=logger,
      # No sanity_val_steps for oneepoc
      num_sanity_val_steps = 0 if FLAGS.oneepoch else 2,
      max_epochs=FLAGS.epochs,
      precision=FLAGS.precision,
      strategy=FLAGS.strategy,
  ) 
  trainer.fit(model, dm, ckpt_path=ar_cb.resume_checkpoint_file)

if __name__ == '__main__':
  app.run(main)
