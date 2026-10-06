import os
from collections import deque

import numpy as np
from shutil import copyfile

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam

from utils.files import get_best_model_name, get_model_stats

import config

class SelfPlayCallback(MaskableEvalCallback):
  def __init__(self, opponent_type, threshold, env_name, logger, *args, base_eval_env=None, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.log = logger
    # fixed reference env (opponent_type='start'): the model this run starts from, frozen,
    # so the progress metric is independent of promotions
    self.base_eval_env = base_eval_env
    self.opponent_type = opponent_type
    self.model_dir = os.path.join(config.MODELDIR, env_name)
    self.generation, self.base_timesteps, bmr = get_model_stats(get_best_model_name(env_name))

    #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
    self.best_mean_reward = -np.inf

    self.threshold = threshold # the threshold is a constant

    if self.base_eval_env is not None and self.base_eval_env.opponent_type == 'start':
      ref = f"best_model.zip (generation {self.generation})" if os.path.exists(os.path.join(self.model_dir, 'best_model.zip')) else "base.zip"
      self.log.info(f"Fixed eval reference (eval/*_vs_base): {ref}")


  def _on_step(self) -> bool:

    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

      # Progress metric: evaluate against the frozen start model (base.zip on a fresh
      # run, best_model.zip as it was at launch otherwise; tags keep the historical
      # "_vs_base" name). Unlike the self-play eval below, this reference never moves, so the curve in
      # tensorboard shows whether the agent is actually improving even when no
      # generation gets promoted. Recorded before super()._on_step() so the parent's
      # logger.dump() flushes everything at the same timestep.
      if self.base_eval_env is not None:
        self.base_eval_env.episode_history = deque(maxlen=4096)
        ep_rewards, _ = evaluate_policy(
            self.model,
            self.base_eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=False,
        )
        # zero-sum reward: terminal +/-1 dominates the accumulated shaping (|sum| < 1),
        # so the sign of the episode reward identifies the winner
        win_rate_vs_base = float(np.mean([r > 0 for r in ep_rewards]))
        mean_reward_vs_base = float(np.mean(ep_rewards))
        self.logger.record("eval/win_rate_vs_base", win_rate_vs_base)
        self.logger.record("eval/mean_reward_vs_base", mean_reward_vs_base)
        self.log.info("Eval vs start model: win_rate={:.2f}, mean_reward={:.2f}".format(win_rate_vs_base, mean_reward_vs_base))
        self._record_per_count(self.base_eval_env.episode_history, "eval/mean_reward_vs_base")

      self.eval_env.set_attr('episode_history', deque(maxlen=4096))
      result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
      # the parent has already dumped its metrics: dump the per-count ones on their own
      if self._record_per_count([g for h in self.eval_env.get_attr('episode_history') for g in h],
                                "eval/mean_reward"):
        self.logger.dump(self.num_timesteps)

      self.log.info("Eval num_timesteps={}, episode_reward={:.2f}".format(self.num_timesteps, self.best_mean_reward))
      self.log.info("Total episodes ran={}".format(self.n_eval_episodes))

      #compare the latest reward against the threshold
      if result and self.best_mean_reward > self.threshold:
        self.generation += 1
        self.log.info(f"New best model: {self.generation}\n")

        generation_str = str(self.generation).zfill(5)
        rewards_str = str(round(self.best_mean_reward,3))
        
        source_file = os.path.join(config.TMPMODELDIR, f"best_model.zip") # this is constantly being written to - not actually the best model
        target_file = os.path.join(self.model_dir,  f"_model_{generation_str}_{rewards_str}_{str(self.base_timesteps + self.num_timesteps)}_.zip")
        copyfile(source_file, target_file)
        target_file = os.path.join(self.model_dir,  f"best_model.zip")
        copyfile(source_file, target_file)
        
      #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
      self.best_mean_reward = -np.inf

    return True
  
  def _record_per_count(self, history, tag) -> bool:
    """Log the mean reward per player count of the evaluation games (`tag`_<n>p),
    for the games whose player count varies (jamaica). False if only one count."""
    counts = sorted({n for n, _ in history})
    if len(counts) < 2:
      return False
    means = {n: float(np.mean([r for k, r in history if k == n])) for n in counts}
    for n, mean in means.items():
      self.logger.record(f"{tag}_{n}p", mean)
    self.log.info("  {} per player count: {}".format(tag, ", ".join(
        f"{n}p={mean:.2f} ({sum(k == n for k, _ in history)} games)" for n, mean in means.items())))
    return True

  def _on_training_start(self) -> None:
    hparam_dict = {
        "gamma": self.model.gamma,
        "ent_coef": self.model.ent_coef,
        "n_epochs": self.model.n_epochs,
        "clip_range": self.model.clip_range(0),
        "batch_size": self.model.batch_size,
    }
    # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
    # Tensorbaord will find & display metrics from the `SCALARS` tab
    metric_dict = {
        "rollout/ep_rew_mean": 0.0,
        "eval/mean_reward": 0.0,
    }
    self.logger.record(
        "hparams",
        HParam(hparam_dict, metric_dict),
        exclude=("stdout", "log", "json", "csv"),
    )