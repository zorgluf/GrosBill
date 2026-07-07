import os
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
    # fixed baseline env (opponent_type='base'): progress metric independent of promotions
    self.base_eval_env = base_eval_env
    self.opponent_type = opponent_type
    self.model_dir = os.path.join(config.MODELDIR, env_name)
    self.generation, self.base_timesteps, bmr = get_model_stats(get_best_model_name(env_name))

    #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
    self.best_mean_reward = -np.inf

    self.threshold = threshold # the threshold is a constant


  def _on_step(self) -> bool:

    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

      # Progress metric: evaluate against the frozen base (random-init) opponent.
      # Unlike the self-play eval below, this baseline never moves, so the curve in
      # tensorboard shows whether the agent is actually improving even when no
      # generation gets promoted. Recorded before super()._on_step() so the parent's
      # logger.dump() flushes everything at the same timestep.
      if self.base_eval_env is not None:
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
        self.log.info("Eval vs base: win_rate={:.2f}, mean_reward={:.2f}".format(win_rate_vs_base, mean_reward_vs_base))

      result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf

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