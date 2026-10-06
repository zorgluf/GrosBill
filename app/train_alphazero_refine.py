"""AlphaZero-style refinement of an existing PPO agent.

Warm-starts from zoo/<env>/best_model.zip (train PPO first with train.py) and
refines it with search-improved targets instead of the PPO surrogate:

  1. Self-play with determinized MCTS (hidden cards re-dealt at every state load,
     like train_mcts.DeterminizedGymctsNeuralAgent), leaves evaluated by the VALUE
     HEAD (no random rollouts), Dirichlet noise at the root and temperature-based
     move sampling for the first moves. The games are played in parallel worker
     processes (--n_workers).
  2. For every visited state, record the root visit-count distribution pi and,
     at the end of the game, the outcome z = +/-1.
  3. Train the SAME MaskableActorCriticPolicy with the AlphaZero loss:
     cross-entropy(policy, pi) + vf_coef * MSE(value, z). No importance ratios,
     no clipping, no GAE.

The search is a small MCTS of its own (no gymcts) with LAZY expansion, as in
AlphaZero: expanding a node only stores the priors of its children, and a child's
env state is built the first time the search selects it. gymcts' eager expansion
stepped every legal child (one opponent forward pass each) and evaluated one of
them, so most of the work went into children the search never visited. The
selection rule and the training targets are unchanged (gymcts' PUCT_v0).

The output stays a plain MaskablePPO zip: promotion, play.py, test.py and the
self-play opponents all keep working unchanged (they only call model.predict).
"""
import os
import sys
import argparse
import copy
import logging
import math
import multiprocessing
import random
import signal
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

# gymcts defaults (GymctsNode.ubc_c / best_action_weight), kept from the gymcts-based search
C_PUCT = 0.707
BEST_ACTION_WEIGHT = 0.05

logger = logging.getLogger(__name__)
_warned_no_redeterminize = False


class Node:
    """MCTS node. Values are the learner's: the opponents' moves happen inside the
    self-play env step, so there is no sign flip between tree levels."""

    __slots__ = ('prior', 'state', 'obs', 'acc_return', 'terminal', 'children',
                 'visit_count', 'mean_value', 'max_value')

    def __init__(self, prior: float):
        self.prior = prior
        self.state = None         # self-play env at this node, built on first visit
        self.obs = None
        self.acc_return = 0.0     # learner's return from the start of the game to this node
        self.terminal = False
        self.children = None      # {action: Node}, set by evaluate()
        self.visit_count = 0
        self.mean_value = 0.0
        self.max_value = -math.inf

    def score(self, parent_visits: int) -> float:
        """gymcts PUCT_v0: Q = 0 while unvisited, else a mean/max mix."""
        q = 0.0 if self.visit_count == 0 else (
            (1 - BEST_ACTION_WEIGHT) * self.mean_value + BEST_ACTION_WEIGHT * self.max_value)
        return q + C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)


def load_state(state):
    """Copy of a node's env with the hidden information re-dealt from the learner's
    point of view (determinized MCTS, see train_mcts.DeterminizedGymctsNeuralAgent):
    the search cannot plan with the true deck order or the opponent's real cards."""
    global _warned_no_redeterminize
    env = copy.deepcopy(state)
    if hasattr(env, 'redeterminize'):
        if not env.done:
            env.redeterminize(env.agent_player_num)
    elif not _warned_no_redeterminize:
        _warned_no_redeterminize = True
        logger.warning(f"env {env.name} has no redeterminize(): MCTS will plan "
                       f"with perfect information (hidden-info leak)")
    return env


def evaluate(policy, node) -> float:
    """One forward pass (the body of MaskableActorCriticPolicy.forward): sets the
    node's children with their masked priors and returns the value head output."""
    mask = np.asarray(node.state.action_masks(), dtype=bool)
    obs_t, _ = policy.obs_to_tensor(node.obs)
    with torch.no_grad():
        features = policy.extract_features(obs_t)
        if policy.share_features_extractor:
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            latent_pi = policy.mlp_extractor.forward_actor(features[0])
            latent_vf = policy.mlp_extractor.forward_critic(features[1])
        distribution = policy._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(mask[None])
        probs = distribution.distribution.probs[0].cpu().numpy()
        value = float(policy.value_net(latent_vf).item())
    node.children = {int(a): Node(float(probs[a])) for a in np.flatnonzero(mask)}
    return value


def create_child(policy, parent, action) -> float:
    """First visit of parent.children[action]: step a determinized copy of the
    parent's env. Returns the leaf value: return so far + value head (AlphaZero
    leaf evaluation, no rollout). The accumulated return keeps it comparable with
    terminal leaves, whose value is the plain episode return."""
    child = parent.children[action]
    env = load_state(parent.state)
    obs, reward, terminated, truncated, _ = env.step(action)
    child.state, child.obs = env, obs
    child.acc_return = parent.acc_return + float(reward)
    child.terminal = terminated or truncated
    if child.terminal:
        return child.acc_return
    return child.acc_return + evaluate(policy, child)


def simulate(policy, root) -> None:
    """One simulation: PUCT descent to a terminal node or to a child never visited
    (which is built and evaluated), then backup along the path."""
    node, path = root, [root]
    while True:
        if node.terminal:
            value = node.acc_return
            break
        best_score, best = -math.inf, []
        for action, child in node.children.items():
            s = child.score(node.visit_count)
            if s > best_score:
                best_score, best = s, [action]
            elif s == best_score:
                best.append(action)
        action = best[0] if len(best) == 1 else random.choice(best)
        child = node.children[action]
        path.append(child)
        if child.state is None:
            value = create_child(policy, node, action)
            break
        node = child
    for n in path:
        n.mean_value += (value - n.mean_value) / (n.visit_count + 1)
        n.visit_count += 1
        n.max_value = max(n.max_value, value)


def generate_episode(env, policy, n_simulations, dirichlet_alpha, dirichlet_eps,
                     temperature_moves):
    """Play one self-play game with the MCTS agent (env is reset here).
    Returns (obs_list, mask_list, pi_list, z) where pi is the root visit
    distribution for each state and z = +/-1 the final outcome for the agent."""
    obs, _ = env.reset()
    root = Node(1.0)
    root.state, root.obs = env, obs  # never stepped itself: the search steps copies
    obs_list, mask_list, pi_list = [], [], []
    move_idx = 0
    while not root.terminal:
        obs_list.append(root.obs)
        mask_list.append(np.array(root.state.action_masks(), dtype=bool))

        # fresh tree every move: (re)build the root's children, then mix
        # Dirichlet noise into their priors (exploration across games)
        evaluate(policy, root)
        children = root.children
        if dirichlet_eps > 0 and len(children) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(children))
            for child, eta in zip(children.values(), noise):
                child.prior = (1 - dirichlet_eps) * child.prior + dirichlet_eps * float(eta)
        if len(children) > 1:  # a forced move needs no search
            for _ in range(n_simulations):
                simulate(policy, root)

        # visit-count distribution over the root children = the policy target
        actions = np.array(list(children.keys()))
        visits = np.array([c.visit_count for c in children.values()], dtype=np.float64)
        if visits.sum() == 0:  # forced move: no simulation was run
            visits = np.ones_like(visits)
        pi = visits / visits.sum()
        pi_full = np.zeros(env.action_space.n, dtype=np.float32)
        pi_full[actions] = pi
        pi_list.append(pi_full)

        # temperature move selection: sample early for diversity, argmax after
        if move_idx < temperature_moves:
            action = int(np.random.choice(actions, p=pi))
        else:
            action = int(actions[np.argmax(visits)])
        move_idx += 1

        if children[action].state is None:  # forced move: never built by the search
            create_child(policy, root, action)
        root = children[action]
        root.visit_count, root.mean_value, root.max_value = 0, 0.0, -math.inf
    # zero-sum env: the sign of the episode reward identifies the winner
    z = 1.0 if root.acc_return > 0 else -1.0
    return obs_list, mask_list, pi_list, z


# --- self-play worker processes -------------------------------------------------
_worker = {}


def _init_worker(env_name, opponent_type, model_path, search_kwargs):
    """Each worker owns a self-play env and a CPU copy of the policy, whose weights
    come with every game (the main process trains it between iterations)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C stops the main process, which terminates the pool
    torch.set_num_threads(1)  # batch-1 forward passes; the workers already use every core
    env = selfplay_wrapper(get_environment(env_name))(opponent_type=opponent_type, logger=logger, device='cpu')
    model = MaskablePPO.load(model_path, env, device='cpu')
    model.policy.set_training_mode(False)
    _worker.update(env=env, policy=model.policy, search_kwargs=search_kwargs)


def _play_game(task):
    weights, seed = task
    policy = _worker['policy']
    policy.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    set_random_seed(seed)
    return generate_episode(_worker['env'], policy, **_worker['search_kwargs'])


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
    model_dir = os.path.join(config.MODELDIR, args.env_name)
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

    generation, base_timesteps, _ = get_model_stats(get_best_model_name(args.env_name))
    sb3_logger = configure(os.path.join(config.LOGDIR, log_name), ["tensorboard"])

    search_kwargs = dict(n_simulations=args.nb_sim_mcts, dirichlet_alpha=args.dirichlet_alpha,
                         dirichlet_eps=args.dirichlet_eps, temperature_moves=args.temperature_moves)
    n_workers = min(args.n_workers, args.nb_episode_gen)
    pool = None
    if n_workers > 1:
        # spawn, not fork: the parent already runs torch threads
        pool = multiprocessing.get_context('spawn').Pool(
            n_workers, initializer=_init_worker,
            initargs=(args.env_name, args.opponent_type, model_path, search_kwargs))
        logger.info(f'Self-play on {n_workers} worker processes')

    positions = deque(maxlen=args.buffer_size)
    total_positions = 0

    for iteration in range(args.nb_improve_loop):
        logger.info(f"Iteration {iteration + 1}/{args.nb_improve_loop}")

        # 1. self-play with search
        model.policy.set_training_mode(False)
        if pool is None:
            games = (generate_episode(env_self, model.policy, **search_kwargs)
                     for _ in range(args.nb_episode_gen))
        else:
            weights = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
            seeds = np.random.randint(0, 2**31 - 1, size=args.nb_episode_gen)
            games = pool.imap(_play_game, [(weights, int(s)) for s in seeds])
        new_positions, wins = 0, 0
        for g, (obs_l, mask_l, pi_l, z) in enumerate(games):
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
            # the self-play envs (here and in the workers) and eval_env pick up the new
            # best opponent at their next reset (setup_opponents watches get_best_model_name)

    if pool is not None:
        pool.close()
        pool.join()


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
  parser.add_argument("--n_workers", "-nw",  type = int, default = os.cpu_count()
            , help="Processes playing the self-play games in parallel (1 = in the main process)")
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
            , help="The device to use for training (the search always runs on CPU workers, except with -nw 1)")

  args = parser.parse_args()
  main(args)


if __name__ == '__main__':
  cli()
