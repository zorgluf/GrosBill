"""Policy for Small World (`smallw`) — **dummy model, the real network is future work**.

The `smallw` task list (plan T6) only needs a policy that *loads*, so that
`base.zip` can be created and a freshly initialised (therefore essentially
random) agent can drive `play.py` / `test.py` and prove that SB3 accepts the
environment's spaces. Nothing here is tuned for playing strength:

* the observation is a `gym.spaces.Dict` of integer boxes (`regions` (30, 16),
  `players` (5, 13), `combos` (6, 3), `global` (16,)), so SB3's
  `CombinedExtractor` is used: it casts every box to float and flattens it
  (579 features), with no embedding and no normalisation. It has to be asked
  for **explicitly**: SB3 only picks it automatically when the policy is named
  by an alias (`'MultiInputPolicy'`), and every `ActorCriticPolicy` subclass
  passed as a class — as `get_network_arch` does — defaults to
  `FlattenExtractor`, which crashes on a Dict observation
  (`'dict' object has no attribute 'flatten'`),
* a small shared MLP (`pi=[128, 128]`, `vf=[128, 128]`) sits on top,
* the action mask of `SmallWorldEnv.action_masks()` is applied by
  `MaskableActorCriticPolicy`, so a fresh policy plays uniformly at random
  among the legal actions.

**Future work** (a separate task): a real network, i.e. an
`nn.Embedding` over the categorical columns (terrain, owner code, race / power
ids, phase) feeding a per-region encoder (the regions are a graph, see
`envs/map3p.ADJACENCY`) and a factored / pointer action head over the
region-indexed action ranges of `smallw.py` (`A_REGION`, `A_REGION_ALL`,
`A_SORCERER`, `A_DRAGON`). Keep `CustomPolicy` under this name when that
happens only if the weights stay compatible — otherwise add a new class and a
new `get_network_arch` entry, the way `stotten` / `stottentr` do, so the
existing `zoo/smallw` snapshots still load.
"""

from typing import Callable

from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor

#: Hidden layers of the policy / value MLPs on top of the flattened observation.
NET_ARCH = dict(pi=[128, 128], vf=[128, 128])


class CustomPolicy(MaskableActorCriticPolicy):
    """Default maskable actor-critic policy + a small MLP (dummy random agent)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # CombinedExtractor (Dict observation) must be requested explicitly, see
        # the module docstring: the inherited default is FlattenExtractor.
        kwargs.setdefault('features_extractor_class', CombinedExtractor)
        kwargs.setdefault('net_arch', NET_ARCH)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
