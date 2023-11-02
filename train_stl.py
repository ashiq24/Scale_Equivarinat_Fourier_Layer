"""Main training loop."""

import os
import pytorch_lightning as pl
import lightning_fabric as lf
#from torchsummary import summary
from absl import app, flags, logging
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import sys
from callbacks import AutoResumeState, OneEpochStop
import clargs.train_opts
from dataloader.stl_data_module import ScaleSTLDataModule
import clargs.data_opts
from models.fourier_seq_stl import FourierSeqStl
from torchvision import transforms
import clargs.logger_opts
from dataloader import get_pl_datamodule, get_available_pl_modules
from models import get_model, get_available_models
from utils.logger_utils import set_logger
from torch.optim.lr_scheduler import StepLR
from tests.test_equievarience import TestEquivarinecError, GetAccuracy
import numpy as np
from utils.core_utils import Cutout
from layers.complex_modules import get_activation
from dataloader.mnist_data_module import get_augmentation
AVAILABLE_DATASETS = get_available_pl_modules()
AVAILABLE_MODELS = get_available_models()

# Dataset
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data', 'Dataset location')
flags.DEFINE_enum('dataset', 'scale_stl',AVAILABLE_DATASETS, 'Dataset')
flags.DEFINE_integer('training_size',7000, "Training set size.")
flags.DEFINE_integer('test_size',5000, "Test set size.")
flags.DEFINE_string('data_resize_mode', 'None', 'Dataset padding mode - None, pad, or resize')
flags.DEFINE_string('data_downscale_mode', 'ideal', 'Dataset downscale mode - "ideal"/"bicubic" ')
flags.DEFINE_string('augmentation', 'circular_shift', 'data Augmentation scheme - "add_noise", "circular_shift"  or "None"') #Defined in core_util

flags.DEFINE_string('model', 'fourier_stl', 'Model name. fourier_stl')
flags.DEFINE_integer('input_channel', 3, 'number of input Channel')
flags.DEFINE_integer('C1', 32, 'Width/number of channel in 1nd Convolutional layer')
flags.DEFINE_integer('C2', 128, 'Width/number of channel in 2nd Convolutional layer')
flags.DEFINE_integer('C3', 256, 'Width/number of channel in 3rd Convolutional layer')
flags.DEFINE_integer('C4', 512, 'Width/number of channel in 4th Convolutional layer')
flags.DEFINE_integer('mixer_band', 3, 'width of window for frequency mixer')

flags.DEFINE_integer('FC1', 256, 'Width of 1st Fully connected NN following the Convolutional Layer')
flags.DEFINE_float('dropout_fc1', 0.00, 'Dropout in the input of the 1st layer of FNN')
flags.DEFINE_float('dropout_fc2', 0.75, 'Dropout in the input of the 2nd layer of FNN')
flags.DEFINE_string('activation_cov', 'relu', 'Activation after Convolutional layers')
flags.DEFINE_string('activation_mlp', 'relu', 'Activation after MLP layers')
flags.DEFINE_string('normalizer', 'instance', 'Normalization to be used')

flags.DEFINE_integer('increment', 8, 'increment')
flags.DEFINE_integer('max_res', 97, 'max_res')
flags.DEFINE_integer('base_res', 40, 'base_res')
flags.DEFINE_integer('pool_size', 13, 'pool_size')


# Trainer
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('weight_decay', 1e-2, 'Weight Decay')
flags.DEFINE_integer('epochs', 200, 'Training epochs')
flags.DEFINE_integer('step_size_lr', 100, 'Step size of lr scheduler')
flags.DEFINE_float('step_gamma_lr', 0.1, 'gamma for lr scheduler')


# Misc.
flags.DEFINE_integer(
    'seed', 0, 'Random seed for shift-consistency experiments')
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
  if len(argv)>2:
    FLAGS.model = argv[2]
  # Model checkpoint

  mc = pl.callbacks.ModelCheckpoint(
      filename='{epoch}-{val_acc:.5f}',
      monitor='val_acc',
      save_last=not (FLAGS.autoresume),
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

  augmentation = get_augmentation(FLAGS.augmentation)
  augmentation = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        Cutout(1, 32),
    ])
  # Experiment set up
  # seletion training and testing scales
  train_scales =[i for i in range(49,98,1)]
  test_scales = [i for i in range(49,98,1)]
  print(train_scales)
  print(test_scales)
  # Dataset
  dm = ScaleSTLDataModule(
      data_dir=FLAGS.data_path,
      training_size = FLAGS.training_size,
      test_size = FLAGS.test_size,
      train_scales = train_scales,
      test_scales = test_scales,
      data_resize_mode = FLAGS.data_resize_mode,
      data_downscale_mode = FLAGS.data_downscale_mode,
      batch_size=FLAGS.batch_size,
      num_workers= 4,
      train_transform = augmentation)

  dm.prepare_data()
  dm.setup()
  assert len(dm.val_dataloader().dataset)>0

  activation_cov = get_activation(FLAGS.activation_cov)
  activation_mlp = get_activation(FLAGS.activation_mlp)
  model = get_model(FLAGS.model)(FLAGS.input_channel, learning_rate = FLAGS.learning_rate, weight_decay = FLAGS.weight_decay,\
                                C1 =FLAGS.C1, C2 = FLAGS.C2, C3 = FLAGS.C3, C4 = FLAGS.C4, FC1 = FLAGS.FC1, dropout_fc1 = FLAGS.dropout_fc1,\
                                dropout_fc2 = FLAGS.dropout_fc2, activation_con = activation_cov, activation_mlp = activation_mlp,\
                                mixer_band = FLAGS.mixer_band, increment = FLAGS.increment, max_res =FLAGS.max_res, base_res = FLAGS.base_res,\
                                pool_size = FLAGS.pool_size,scheduler = StepLR, normalizer = 'instance',\
                                scheduler_kwargs={'step_size':FLAGS.step_size_lr, 'gamma':FLAGS.step_gamma_lr} )



  trainer = pl.Trainer(
      accelerator=FLAGS.accelerator,
      benchmark=True,
      callbacks=cb,
      default_root_dir=log_path,
      devices= [gpu_device],
      fast_dev_run=FLAGS.dryrun,
      val_check_interval = None,
      check_val_every_n_epoch= 1,
      logger=logger,
      # No sanity_val_steps for oneepoc
      num_sanity_val_steps = 0 if FLAGS.oneepoch else 2,
      max_epochs=FLAGS.epochs,
      precision=FLAGS.precision,
      #resume_from_checkpoint=ar_cb.resume_checkpoint_file,
      strategy=FLAGS.strategy,
  )

  trainer.fit(model, dm, ckpt_path=ar_cb.resume_checkpoint_file)
  equ_testter = TestEquivarinecError(model)
  

  l = equ_testter.get_equivarience_error(dm.test_dataloader(),[i for i in range(49,98,8)], mode=FLAGS.data_downscale_mode, samples= 500)
  print("Equivarience Error", l)
  
  print("...............Test ACC....................")
  GT = GetAccuracy(model, dm.test_dataloader())
  print("Accuracty", GT.get_acc())
  
  print("...............Train ACC....................")
  GT = GetAccuracy(model, dm.train_dataloader())
  print("Accuracty", GT.get_acc())


if __name__ == '__main__':
  app.run(main)
