from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

class CustomPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization (to test if still useful)
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[ 1000, 500, 300, 100 ],
            *args,
            **kwargs,
        )


