#Custom GrosBill env
import gymnasium as gym

class GBEnv(gym.Env):
    """ Base class for GrosBill environments.
    see https://gymnasium.farama.org/ for parent class documentation."""

    current_player: int = -1
    pov_player: int = -1

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for environment reset. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation sampled from the observation space and an empty info dictionary.
        """
        super().reset(seed=seed, options=options)
        # Implement the reset logic for the environment in subclasses
        return self.observation_space.sample(), {}

    def step(self, action):
        # Apply the action and return the new state, reward, done, and info
        next_state = self.observation_space.sample()
        reward = 1.0  # Example reward
        terminated = False  # Example termination condition
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def render(self, pov_player: int = -1, mode:str = 'human_web', **kwargs):
        """
        Update the render of the environment. Superseeded by subclasses to implement specific rendering logic.
        Args:
            pov_player (int, optional): Player number for point of view rendering. -1 set the pov_player as the current player.
            mode (str, optional): Rendering mode. Defaults to 'human_web'.
        """
        if pov_player == -1:
            pov_player = self.current_player
        else:
            self.pov_player = pov_player
        return

    def close(self):
        pass # Implement any cleanup logic if needed

    def nicegui_page():
        pass  # Implement NiceGUI page rendering