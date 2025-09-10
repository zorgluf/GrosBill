#Custom GrosBill env
import gymnasium as gym

class GBEnv(gym.Env):
    """ Base class for GrosBill environments.
    see https://gymnasium.farama.org/ for parent class documentation."""

    current_player: int = -1 #if -1, means no player move yet for next step
    pov_player: int = -1
    render_mode: str = 'human_web' #default render mode for GrosBill envs
    n_players: int
    player_names: list[str] = None
    done: bool = False #True if the game is finished
    winner_player: int = None #player that win the game if done
    name: str = '' #name of the game, as declared inside the environment directory

    def __init__(self, name: str, n_players: int = 2, player_names: list[str] = None):
        """
        Initializes the GrosBill environment.

        Args:
            name (str): Name of the game, as declared inside the environment directory.
            n_players (int): Number of players in the game.
            player_names (list[str], optional): List of player names. Defaults to None.
        """
        super().__init__()
        self.name = name
        self.n_players = n_players
        if player_names == None:
            self.player_names = [f'Player {i+1}' for i in range(n_players)]
        else:
            self.player_names = player_names
        # Don't forget to set the observation and action spaces in subclasses

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state. Must be called by subclasses.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for environment reset. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation sampled from the observation space and an info dictionary.
        """
        super().reset(seed=seed, options=options)
        self.done = False
        # Implement the reset logic for the environment in subclasses
        return None, {}

    def step(self, action):
        """
        Applies the given action to the environment and returns the resulting state, reward, termination flags, and additional info.
        Args:
            action: The action to be applied to the environment.
        Returns:
            tuple: A tuple containing:
                - next_state: The new state of the environment after the action.
                - reward (List[float]): The rewards obtained from applying the action for each player.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Additional information, including whether the next step requires an action from a player.
        """
        # Apply the action and return the new state, reward, done, and info
        next_state = self.observation_space.sample()
        reward = [1.0, ]  # Example reward
        terminated = False  # Example termination condition
        truncated = False
        info = {
            "next_step_no_action": False #must return this value. Indicate if the next step need an action from one of the players
        }
        raise NotImplementedError("Subclasses must implement action_masks method.")

    def render(self, pov_player: int = None, mode:str = 'human_web', **kwargs):
        """
        Update the render of the environment. Superseeded by subclasses to implement specific rendering logic. Must be called by subclass
        Args:
            pov_player (int, optional): Player number for point of view rendering. -1 activate god mode (see everything). None set the pov_player to the current player
            mode (str, optional): Rendering mode. Defaults to 'human_web'.
            suggested_action (int, optional): Suggested action for human players. Defaults to None.
        """
        if pov_player == None:
            self.pov_player = self.current_player
        else:
            self.pov_player = pov_player
        return

    def close(self):
        pass # Implement any cleanup logic if needed

    def action_masks(self):
        """
        Returns a list of legal actions for the current player.
        This method should be implemented in subclasses to provide specific action masks.
        """
        raise NotImplementedError("Subclasses must implement action_masks method.")

    def nicegui_page(self):
        raise NotImplementedError("Subclasses must implement nicegui page rendering.")