<h1>GrosBill</h1>
<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="images/logo_gb_small.png" alt="Logo" height="120">

  <!-- <h3 align="center">SIMPLE</h3> -->

  <p align="center">
    AI boardgame player
  </p>
</p>
<br>


<!-- TABLE OF CONTENTS -->

  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#tutorial">Tutorial</a></li>
      <ul>
        <li><a href="#prerequisites">Quickstart</a></li>
        <li><a href="#prerequisites">Tensorboard</a></li>
        <li><a href="#custom-environments">Custom Environments</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>



<br>

---
<!-- ABOUT THE PROJECT -->
## About The Project

This project started initialy by a fork of the great project [SIMPLE](https://github.com/davidADSP/SIMPLE) made by David Foster [@davidADSP](https://twitter.com/davidADSP) - david@adsp.ai.
To learn more about this initial project, check out the accompanying [blog post](https://medium.com/applied-data-science/how-to-train-ai-agents-to-play-multiplayer-games-using-self-play-deep-reinforcement-learning-247d0b440717).

Modifications after modifications, it started to diverge with breaking changes, especially to have a better UI interface to really play the games with AI agents. The inittial fork is now a new project, with still the main core logic of training agent.

The main modifications are :
* Support CUDA devices for training
* Migrate from TensorFlow into PyTorch
* Use a web based rendering mode

This guide explains how to get started with the repo, add new custom environments and tune the hyperparameters of the system.

Have fun!

---
<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Install Docker and Docker Compose to make use of the `docker-compose.yml` file

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/zorgluf/GrosBill.git
   cd GrosBill
   ```
2. Build the image and 'up' the container.
   ```sh
   docker-compose up -d
   ```
3. Choose an environment to install in the container (`tictactoe`, `connect4`, `sushigo`, `geschenkt`, `butterfly`, and `flamme rouge` are currently implemented)
   ```sh
   bash ./scripts/install_env.sh sushigo
   ```

Build standalone docker play images :
  ```sh
  docker build . -t simple-play-frouge -f app/Dockerfile_play_frouge
  ```
---
<!-- TUTORIAL -->
## Tutorial

This is a quick tutorial to allow you to start using the two entrypoints into the codebase: `test.py` and `train.py`.

---
<!-- QUICKSTART -->
### Quickstart

#### `test.py` 

This entrypoint allows you to play against a trained AI, pit AIs against eachother or play against baseline random models.

For example, try the following command to play against a baseline random model in the Flamme Rouge environment.
   ```sh
   docker-compose exec app python3 test.py -d -g 1 -a best_model base human best_model best_model -e frouge 
   ```

#### `train.py` 

This entrypoint allows you to start training the AI using selfplay PPO. The underlying PPO engine is from the [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) package.

For example, you can start training the agent to learn how to play SushiGo with the following command:
   ```sh
   docker-compose exec app python3 train.py -r -e frouge -t 250 -os 12800 -ob 256
   ```
*TODO : explain parameters*

After 30 or 40 iterations the process should have achieved above the default threshold score of 0.2 and will output a new `best_model.zip` to the `/zoo/frouge` folder. 

Training runs until you kill the process manually (e.g. with Ctrl-C), so do that now.

You can now use the `test.py` entrypoint to play 100 games silently between the current `best_model.zip` and the random baselines model as follows:

  ```sh
  docker-compose exec app python3 test.py -g 100 -a best_model base base base base -e frouge 
  ```

You should see that the best_model scores better than the two baseline model opponents. 
```sh
Played 100 games: {'best_model_btkce': 31.0, 'base_sajsi': -15.5, 'base_poqaj': -15.5}
```

You can continue training the agent by dropping the `-r` reset flag from the `train.py` entrypoint arguments - it will just pick up from where it left off.

   ```sh
   docker-compose exec app python3 train.py -e frouge -t 250 -os 12800 -ob 256
   ```

Congratulations, you've just completed one training cycle for the game Flamme Rouge! The PPO agent will now have to work out a way to beat the model it has just created...

---
<!-- TENSORBOARD -->
### Tensorboard

To monitor training, you can start Tensorboard with the following command:

  ```sh
  bash scripts/tensorboard.sh
  ```

Navigate to `localhost:6006` in a browser to view the output.

In the `/zoo/pretrained/` folder there is a pre-trained `/<game>/best_model.zip` for each game, that can be copied up a directory (e.g. to `/zoo/frouge/best_model.zip`) if you want to test playing against a pre-trained agent right away.

---
<!-- CUSTOM ENVIRONMENTS -->
### Custom Environments

You can add a new environment by copying and editing an existing environment in the `/environments/` folder.

For the environment to work with the SIMPLE self-play wrapper, the class must contain the following methods (expanding on the standard methods from the OpenAI Gym framework):

`__init__`

In the initiation method, you need to define the usual `action_space` and `observation_space`, as well as two additional variables: 
  * `n_players` - the number of players in the game
  * `current_player_num` - an integer that tracks which player is currently active
   

`step`

The `step` method accepts an `action` from the current active player and performs the necessary steps to update the game environment. It should also it should update the `current_player_num` to the next player, and check to see if an end state of the game has been reached.


`reset`

The `reset` method is called to reset the game to the starting state, ready to accept the first action.


`render`

The `render` function is called to output a visual or human readable summary of the current game state to the log file.


`observation`

The `observation` function returns a numpy array that can be fed as input to the PPO policy network. It should return a numeric representation of the current game state, from the perspective of the current player, where each element of the array is in the range `[-1,1]`.


`legal_actions`

The `legal_actions` function returns a numpy vector of the same length as the action space, where 1 indicates that the action is valid and 0 indicates that the action is invalid.


Please refer to existing environments for examples of how to implement each method.

You will also need to add the environment to the two functions in `/utils/register.py` - follow the existing examples of environments for the structure.

---
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/zorgluf/GrosBill/issues) for a list of proposed features (and known issues).


---
<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---
<!-- LICENSE -->
## License

Distributed under the GPL-3.0. See `LICENSE` for more information.


---
<!-- CONTACT -->
## Contact

François Valley [linkedin](www.linkedin.com/in/francois-valley-1133716)
