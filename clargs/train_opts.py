"""Train config."""

from absl import flags

flags.DEFINE_integer('devices',1,'number of gpus to use')
flags.DEFINE_integer('num_cpu',-1,'number of CPUs to use per node; < 0 uses all CPUs.')
flags.DEFINE_enum('accelerator','gpu',['cpu','gpu'],'Hardare: {cpu, gpu, etc.}')
flags.DEFINE_enum('strategy','auto',['dp','ddp','auto'],'Training strategy: {None (default), dp, ddp}')
flags.DEFINE_integer('accumulate_grad_batches',1,'Batch accumulation')
flags.DEFINE_boolean('lr_scale',False,'Linearly scale learn, warmup and decay rates.')
flags.DEFINE_float('lr_scale_factor',512,'Learning rate factor to scale w.r.t. number of GPUs, batch size, batch accum., etc.')
flags.DEFINE_boolean('lr_scale_skip_scheduler',False,'Skip scaling scheduler learning rates.')
flags.DEFINE_enum('lr_scheduler_interval','epoch',['step','epoch'],'Learn rate scheduler update interval.')
flags.DEFINE_boolean('lr_subtract_warmup_t',False,'[MViT] Subtract warmup steps from total scheduler steps.')
flags.DEFINE_float('gradient_clip_val',5,'Gradient clipping value')
flags.DEFINE_integer('precision',32,'Trainer precision')
flags.DEFINE_boolean('update_scheduler',False,'Replace the scheduler resumed from a checkpoint with an arbitrary one.')