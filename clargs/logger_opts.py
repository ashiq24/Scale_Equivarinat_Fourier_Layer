from absl import flags


# Logger config
flags.DEFINE_string('log_path', './logs', 'log directory')
flags.DEFINE_enum('logger', 'neptune', [
                  'tb', 'neptune'], 'Logger to use. Neptune logger requires NEPTUNE_API_TOKEN set.')
flags.DEFINE_boolean('source_files', False, '[Neptune] capture source code.')
flags.DEFINE_boolean('capture_stdout', False,
                     '[Neptune] capture shell standard output.')
flags.DEFINE_boolean('capture_stderr', False,
                     '[Neptune] capture shell standard error.')
flags.DEFINE_boolean('capture_hardware_metrics', False,
                     '[Neptune] capture hardware metrics.')
flags.DEFINE_boolean('log_model_checkpoints', False,
                     '[Neptune] Upload checkpoints to workspace.')
flags.DEFINE_boolean('log_rank_zero_only', False,
                     '[All] Log exclusivelly using node 0 (avoid possible deadlocks).')
flags.DEFINE_boolean('log_sync_dist', True,
                     '[All] Reduces the metric across devices (may lead to a significant communication overhead).')
flags.DEFINE_integer('log_every_n_steps', 50, 'Log stats every n steps')
