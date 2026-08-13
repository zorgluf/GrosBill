"""AlphaZero-style refinement of an existing PPO agent.

Warm-starts from zoo/<env>/best_model.zip (train PPO first with train.py) and
refines it with search-improved targets instead of the PPO surrogate:

  1. Self-play with determinized MCTS (hidden cards re-dealt at every state load,
     see train_mcts.DeterminizedGymctsNeuralAgent), leaves evaluated by the VALUE
     HEAD (no random rollouts), Dirichlet noise at the root and temperature-based
     move sampling for the first moves.
  2. For every visited state, record the root visit-count distribution pi and,
     at the end of the game, the outcome z = +/-1.
  3. Train the SAME MaskableActorCriticPolicy with the AlphaZero loss:
     cross-entropy(policy, pi) + vf_coef * MSE(value, z). No importance ratios,
     no clipping, no GAE.

The output stays a plain MaskablePPO zip: promotion, play.py, test.py and the
self-play opponents all keep working unchanged (they only call model.predict).
"""
import os
import sys
import argparse
import inspect
import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed

from utils.files import get_best_model_name, get_model_stats
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config

import gymnasium.wrappers as _gym_wrappers
import gymcts.gymcts_deepcopy_wrapper as _gymcts_dcw
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log

from train_mcts import DeterminizedGymctsNeuralAgent

# gymcts targets gymnasium >= 1.0 (RecordEpisodeStatistics(buffer_length=...));
# on gymnasium 0.29 the argument is named deque_size. Shim so both work.
if 'buffer_length' not in inspect.signature(_gym_wrappers.RecordEpisodeStatistics.__init__).parameters:
    class _CompatRecordEpisodeStatistics(_gym_wrappers.RecordEpisodeStatistics):
        def __init__(self, env, buffer_length=100, **kwargs):
            super().__init__(env, deque_size=buffer_length, **kwargs)
    _gymcts_dcw.RecordEpisodeStatistics = _CompatRecordEpisodeStatistics


class ValueLeafMCTSWrapper(DeepCopyMCTSGymEnvWrapper):
    """AlphaZero-style leaf evaluation: cumulative return so far + value head,
    instead of a random playout to the end of the game. One forward pass per
    leaf instead of a ~25-move playout (each move of which is an opponent
    predict), and the evaluation improves as the value head is trained.

    The accumulated return keeps leaf values comparable with terminal leaves
    (which return the plain cumulative episode return), like the
    RecordEpisodeStatistics-based rollout() this replaces.
    """

    value_model = None  # class attribute: shared by all deepcopied node states
    _acc_return: float = 0.0

    def reset(self, **kwargs):
        self._acc_return = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        step_tuple = super().step(action)
        self._acc_return += float(step_tuple[1])
        return step_tuple

    def rollout(self) -> float:
        if self.is_terminal():
            return self._acc_return
        obs = self.env.unwrapped.observation
        obs_tensor, _ = ValueLeafMCTSWrapper.value_model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            value = ValueLeafMCTSWrapper.value_model.policy.predict_values(obs_tensor)
        return self._acc_return + float(value.item())


class AlphaZeroMCTSAgent(DeterminizedGymctsNeuralAgent):
    """Determinized MCTS with the AlphaZero data-generation ingredients:
    Dirichlet noise on the root priors (exploration across games), temperature
    sampling of the played move for the first moves (diversity), and capture of
    the root visit-count distribution (the policy training target)."""

    def __init__(self, *args, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 temperature_moves=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature_moves = temperature_moves
        self.move_idx = 0
        self.last_visit_probs = None  # full action-space vector, set each step

    def perform_mcts_step(self, search_start_node=None, num_simulations=None,
                          render_tree_after_step=None):
        if search_start_node is None:
            search_start_node = self.search_root_node
        if num_simulations is None:
            num_simulations = self.number_of_simulations_per_step

        # expand the root up-front so Dirichlet noise can be mixed into its priors
        if search_start_node.is_leaf():
            self.expand_node(search_start_node)
        children = search_start_node.children
        if self.dirichlet_eps > 0 and len(children) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
            for child, eta in zip(children.values(), noise):
                child._selection_score_prior = (
                    (1 - self.dirichlet_eps) * child._selection_score_prior
                    + self.dirichlet_eps * float(eta))

        self.vanilla_mcts_search(search_start_node=search_start_node,
                                 num_simulations=num_simulations)

        # visit-count distribution over the root children = the policy target
        actions = np.array(list(children.keys()))
        visits = np.array([c.visit_count for c in children.values()], dtype=np.float64)
        if visits.sum() == 0:  # defensive: cannot happen with num_simulations >= 1
            visits = np.ones_like(visits)
        pi = visits / visits.sum()
        self.last_visit_probs = np.zeros(self.env.action_space.n, dtype=np.float32)
        self.last_visit_probs[actions] = pi

        # temperature move selection: sample early for diversity, argmax after
        if self.move_idx < self.temperature_moves:
            action = int(np.random.choice(actions, p=pi))
        else:
            action = int(actions[np.argmax(visits)])
        self.move_idx += 1

        next_node = children[action]
        if self.clear_mcts_tree_after_step:
            next_node.reset()
        elif not self.keep_whole_tree_till_initial_root:
            next_node.remove_parent()
        self.search_root_node = next_node
        return action, next_node


def generate_episode(agent):
    """Play one self-play game with the MCTS agent.
    Returns (obs_list, mask_list, pi_list, z) where pi is the root visit
    distribution for each state and z = +/-1 the final outcome for the agent."""
    obs_list, mask_list, pi_list = [], [], []
    node = agent.search_root_node
    total_reward = 0.0
    done = False
    while not done:
        obs_list.append(node._obs)
        mask_list.append(np.array(node.state.action_masks(), dtype=bool))
        action, node = agent.perform_mcts_step()
        pi_list.append(agent.last_visit_probs)
        total_reward += float(node.state._step_tuple[1])
        done = node.terminal
    # zero-sum env: the sign of the episode reward identifies the winner
    z = 1.0 if total_reward > 0 else -1.0
    return obs_list, mask_list, pi_list, z


def train_network(model, positions, batch_size, n_epochs, lr, vf_coef,
                  max_grad_norm=0.5):
    """AlphaZero loss on recorded positions:
    cross-entropy(policy, visit distribution) + vf_coef * MSE(value, outcome).
    Plain supervised learning on model.policy — no PPO machinery."""
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(positions)
    policy_losses, value_losses = [], []
    for _ in range(n_epochs):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch = [positions[i] for i in order[start:start + batch_size]]
            obs_t = obs_as_tensor(
                {k: np.stack([b[0][k] for b in batch]) for k in batch[0][0]},
                model.device)
            masks = np.stack([b[1] for b in batch])
            pis = torch.as_tensor(np.stack([b[2] for b in batch]),
                                  dtype=torch.float32, device=model.device)
            zs = torch.as_tensor(np.array([b[3] for b in batch]),
                                 dtype=torch.float32, device=model.device)

            distribution = policy.get_distribution(obs_t, action_masks=masks)
            probs = distribution.distribution.probs
            # pi is zero on illegal actions, so clamping keeps 0*log(0) at 0
            policy_loss = -(pis * torch.log(probs.clamp_min(1e-9))).sum(dim=1).mean()
            values = policy.predict_values(obs_t).flatten()
            value_loss = F.mse_loss(values, zs)

            loss = policy_loss + vf_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss))
            value_losses.append(float(value_loss))
    policy.set_training_mode(False)
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def main(args):
    log.setLevel(20)

    model_dir = os.path.join(config.MODELDIR, args.env_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(config.DEBUG if args.debug else config.INFO)
    log_name = args.log_name if args.log_name else f"{args.env_name}_az"

    seed = args.seed if args.seed != 0 else random.randint(0, 1000)
    set_random_seed(seed)

    base_env = get_environment(args.env_name)
    env_self = selfplay_wrapper(base_env)(opponent_type=args.opponent_type, logger=logger, device=args.device)
    eval_env = selfplay_wrapper(base_env)(opponent_type=args.opponent_type, logger=logger, device=args.device)
    base_eval_env = selfplay_wrapper(base_env)(opponent_type='base', logger=logger, device=args.device)

    # warm start is the whole point of "refine": search quality depends on the
    # priors and the value head, so we require an already-trained model.
    model_path = os.path.join(model_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        sys.exit(f"{model_path} not found: train a PPO agent first (train.py), "
                 f"then refine it with this script.")
    logger.info('Warm-starting from best_model.zip...')
    model = MaskablePPO.load(model_path, env_self, device=args.device)
    ValueLeafMCTSWrapper.value_model = model

    generation, base_timesteps, _ = get_model_stats(get_best_model_name(args.env_name))
    sb3_logger = configure(os.path.join(config.LOGDIR, log_name), ["tensorboard"])

    positions = deque(maxlen=args.buffer_size)
    total_positions = 0

    for iteration in range(args.nb_improve_loop):
        logger.info(f"Iteration {iteration + 1}/{args.nb_improve_loop}")

        # 1. self-play with search
        model.policy.set_training_mode(False)
        new_positions, wins = 0, 0
        for g in range(args.nb_episode_gen):
            env_self.reset()
            wrapped = ValueLeafMCTSWrapper(env_self, action_mask_fn=lambda env: env.env.action_masks())
            agent = AlphaZeroMCTSAgent(
                env=wrapped,
                model=model,
                number_of_simulations_per_step=args.nb_sim_mcts,
                clear_mcts_tree_after_step=True,
                render_tree_after_step=False,
                render_tree_max_depth=3,
                exclude_unvisited_nodes_from_render=False,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                temperature_moves=args.temperature_moves,
            )
            obs_l, mask_l, pi_l, z = generate_episode(agent)
            for o, m, p in zip(obs_l, mask_l, pi_l):
                positions.append((o, m, p, z))
            new_positions += len(obs_l)
            wins += int(z > 0)
            logger.info(f"  episode {g + 1}/{args.nb_episode_gen}: {len(obs_l)} positions, outcome {z:+.0f}")
        total_positions += new_positions
        model.num_timesteps += new_positions
        search_win_rate = wins / args.nb_episode_gen

        # 2. AlphaZero loss on the replay buffer
        policy_loss, value_loss = train_network(
            model, list(positions), args.batch_size, args.n_epochs, args.lr, args.vf_coef)
        logger.info(f"  trained on {len(positions)} positions: "
                    f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                    f"search_win_rate={search_win_rate:.2f}")

        # 3. eval (same metrics as SelfPlayCallback) + promotion into the zoo
        ep_rewards, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes,
                                        deterministic=False, return_episode_rewards=True, warn=False)
        mean_reward = float(np.mean(ep_rewards))
        base_rewards, _ = evaluate_policy(model, base_eval_env, n_eval_episodes=args.n_eval_episodes,
                                          deterministic=False, return_episode_rewards=True, warn=False)
        win_rate_vs_base = float(np.mean([r > 0 for r in base_rewards]))
        logger.info(f"  eval: mean_reward={mean_reward:.3f} win_rate_vs_base={win_rate_vs_base:.2f}")

        sb3_logger.record("az/policy_loss", policy_loss)
        sb3_logger.record("az/value_loss", value_loss)
        sb3_logger.record("az/search_win_rate", search_win_rate)
        sb3_logger.record("az/buffer_positions", len(positions))
        sb3_logger.record("eval/mean_reward", mean_reward)
        sb3_logger.record("eval/win_rate_vs_base", win_rate_vs_base)
        sb3_logger.dump(base_timesteps + model.num_timesteps)

        if mean_reward > args.threshold:
            generation += 1
            logger.info(f"  New best model: generation {generation}\n")
            generation_str = str(generation).zfill(5)
            rewards_str = str(round(mean_reward, 3))
            target = os.path.join(model_dir,
                                  f"_model_{generation_str}_{rewards_str}_{base_timesteps + model.num_timesteps}_.zip")
            model.save(target)
            model.save(os.path.join(model_dir, 'best_model.zip'))
            # env_self/eval_env pick up the new best opponent at their next reset
            # (setup_opponents watches get_best_model_name)


def cli() -> None:
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                   description="AlphaZero-style refinement of an existing best_model.zip")

  parser.add_argument("--env_name", "-e", type = str, default = 'stotten'
              , help="Which gym environment to train in (needs redeterminize() for correct search): stotten")
  parser.add_argument("--opponent_type", "-o", type = str, default = 'mostly_best'
              , help="best / mostly_best / random / base - the type of opponent to train against")
  parser.add_argument("--debug", "-d", action = 'store_true', default = False
              , help="Debug logging")
  parser.add_argument("--log_name", "-log", type = str, default = None
              , help="Name of the experiment in tensorboard")
  parser.add_argument("--seed", "-s",  type = int, default = 0
            , help="Random seed. If 0, random")

  parser.add_argument("--nb_improve_loop", "-niloop",  type = int, default = 100
            , help="Number of improvement iterations (self-play generation + training + eval)")
  parser.add_argument("--nb_episode_gen", "-negen",  type = int, default = 20
            , help="Self-play games generated per iteration (each game costs nb_sim_mcts searches per move)")
  parser.add_argument("--nb_sim_mcts", "-simmcts",  type = int, default = 100
            , help="MCTS simulations per move")
  parser.add_argument("--dirichlet_alpha", "-dira", type = float, default = 0.3
            , help="Dirichlet noise concentration on root priors (~10/branching factor)")
  parser.add_argument("--dirichlet_eps", "-dire", type = float, default = 0.25
            , help="Fraction of Dirichlet noise mixed into root priors (0 disables)")
  parser.add_argument("--temperature_moves", "-tmoves", type = int, default = 8
            , help="Number of opening moves sampled proportionally to visit counts (argmax after)")

  parser.add_argument("--buffer_size", "-buf", type = int, default = 10000
            , help="Replay buffer size in positions (~25 positions per game)")
  parser.add_argument("--batch_size", "-ob",  type = int, default = 256
            , help="Minibatch size for the supervised update")
  parser.add_argument("--n_epochs", "-oe",  type = int, default = 2
            , help="Passes over the replay buffer per iteration")
  parser.add_argument("--lr", "-lr", type = float, default = 1e-4
            , help="Learning rate (low: this refines an already-trained model)")
  parser.add_argument("--vf_coef", "-vf", type = float, default = 1.0
            , help="Weight of the value loss (AlphaZero uses 1.0)")

  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
            , help="Episodes per evaluation (vs best and vs base)")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.2
            , help="Mean eval reward needed to promote a new generation (same scale as train.py)")

  parser.add_argument("--device", "-dev",  type = str, default = "cpu"
            , help="The device to use")

  args = parser.parse_args()
  main(args)


if __name__ == '__main__':
  cli()
