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
        <li><a href="#quickstart">Quickstart</a></li>
        <li><a href="#tensorboard">Tensorboard</a></li>
        <li><a href="#custom-environments">Custom Environments</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>



<br>

---
<!-- ABOUT THE PROJECT -->
## About The Project

GrosBill lets you **play board games against AI agents in your browser**, and **train those agents yourself** through self-play reinforcement learning.

Two games are currently implemented:
* **Flamme Rouge** (`frouge`) — the cycling race game, played as 1 human against 4 AI riders
* **Schotten Totten** (`stotten`) — the 2-player card game, played head-to-head against the AI

This project initially started as a fork of the great project [SIMPLE](https://github.com/davidADSP/SIMPLE) made by David Foster [@davidADSP](https://twitter.com/davidADSP) - david@adsp.ai.
To learn more about this initial project, check out the accompanying [blog post](https://medium.com/applied-data-science/how-to-train-ai-agents-to-play-multiplayer-games-using-self-play-deep-reinforcement-learning-247d0b440717).

Modification after modification, it started to diverge with breaking changes, especially to have a better UI to really play the games with AI agents. The initial fork is now a new project, which still keeps the main core logic of agent training.

The main modifications are:
* Support CUDA devices for training
* Migration from TensorFlow to PyTorch ([Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) / [sb3-contrib](https://sb3-contrib.readthedocs.io/) MaskablePPO)
* A web based rendering mode ([NiceGUI](https://nicegui.io/)) to play the games from a browser
* Extra training modes: imitation learning from your own recorded games, MCTS-based training and AlphaZero-style refinement

This guide explains how to get started with the repo, add new custom environments and tune the hyperparameters of the system.

Have fun!

---
<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Either:
* **Docker** (and Docker Compose to make use of the `docker-compose.yml` file) — the easiest way to just play, or
* **Python 3.12** with `pip` if you prefer to run natively (a virtual environment is recommended)

Pre-trained models for both games are shipped in the repo (`app/zoo/pretrained/`), so you can play right away without training anything.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/zorgluf/GrosBill.git
   cd GrosBill
   ```

2. **Play with Docker (recommended to just play).** Build and run the standalone play image, which bundles the pre-trained models:
   ```sh
   docker build . -t grosbill-play -f Dockerfile_play
   docker run -p 8080:8080 grosbill-play
   ```
   Then open http://localhost:8080 in your browser.

3. **Or run natively.** Install the dependencies and launch commands directly from the *app* directory:
   ```sh
   pip install -r app/requirements.txt
   cd app
   python3 play.py
   ```
   Note: PyTorch is installed as a dependency; on a machine without a GPU you can save time and disk space by installing the CPU wheels first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

---
<!-- TUTORIAL -->
## Tutorial

This is a quick tutorial to allow you to start playing and training a new agent.

---
<!-- QUICKSTART -->
### Quickstart

#### `play.py`

This entrypoint launches a web server on port 8080 as a GUI interface to play against a trained AI.

   ```sh
   cd app
   python3 play.py
   ```

Open http://localhost:8080 and pick a game from the home page. Each game loads the current `best_model.zip` from `app/zoo/<env>/` as opponent (falling back on the pre-trained ones). From the home page you can also toggle two options:
* **Suggest action** — the AI shows you the move it would play in your position, useful to learn a game or evaluate the agent.
* **Record for future learning** — your finished games are saved as trajectories (in `app/zoo/trajectories/`) that can later be fed to `train_expert.py` for imitation learning. The home page shows how many trajectories have been recorded for each game.

#### `train.py`

This entrypoint allows you to start training the AI using self-play PPO (the maskable-action variant `MaskablePPO` from [sb3-contrib](https://sb3-contrib.readthedocs.io/)).

You must select the environment to train with `-e` (`frouge`, `stotten`, or `stottentr` — a transformer-policy variant of Schotten Totten kept in its own zoo/logs namespace). For a detailed explanation of the training parameters, please read the help carefully:
   ```sh
   python3 train.py --help
   ```

After a few minutes the process should have achieved above the default threshold score and will output a new `best_model.zip` to the `app/zoo/<env>` folder. Depending on training parameters (`--opponent_type`, default `mostly_best`), it might use this `best_model.zip` for the opponents and train a new model against it. In that case, generating a new "best model" that beats the previous one will take significantly more time.

Training runs until you kill the process manually (e.g. with Ctrl-C), so do that now.

As reference, the following parameters are used for training the best models:
*  For Flamme Rouge (zero-sum ranked reward: +1 for the winner, -0.1..-0.4 for the losers, so `-t 0.3` promotes at roughly a 44% win rate, more than twice the 20% chance level):
   ```sh
   python3 train.py -r -e frouge -t 0.3 -os 2048 -ob 1024 -oe 5
   ```
* For Schotten Totten:
   ```sh
   python3 train.py -r -e stotten -dev cuda -t 0.3 -ent 0.003 -n_envs 4 -os 2048 -ob 1024 -ne 200
   ```

Beware: the `-r` / `--reset` flag deletes the existing models and logs of the environment before starting from scratch. You can continue training an existing agent by dropping it — training will just pick up from where it left off.

#### `test.py`

You can use the `test.py` entrypoint to play games silently between agents, e.g. 100 games between the current `best_model.zip` and random baseline models (`base`) as follows (Flamme Rouge is a 5-player race, hence the 4 opponents):

  ```sh
  python3 test.py -g 100 -a best_model base base base base -e frouge
  ```

You should see that `best_model` scores better than the baseline opponents:
```sh
Played 100 games: {'best_model_btkce': 31.0, 'base_sajsi': -15.5, 'base_poqaj': -15.5, 'base_kfjJs': -15.5, 'base_ldhfk': -15.5}
```

You can also pass `human` as one of the agents to play in the terminal (not supported by current game implementation, still work in progress), or any archived model name from `app/zoo/<env>/` to pit specific model generations against each other.

#### Other training entrypoints (experimental)

Beyond plain self-play PPO, the repo provides some more experimental trainers (all share most of `train.py`'s arguments — see their `--help`):
* `train_expert.py` — imitation learning: bootstraps a model by behavioral cloning from the trajectories you recorded through the play UI, then continues with self-play PPO.
* `train_mcts.py` — MCTS-guided training with determinization for hidden-information games.
* `train_alphazero_refine.py` — AlphaZero-style refinement of an existing model.
* `train_hyper.py` — a simple hyperparameter grid-search launcher; results are compared through Tensorboard.

---
<!-- TENSORBOARD -->
### Tensorboard

To monitor training, you can start Tensorboard with the following command:

  ```sh
  tensorboard --logdir app/logs
  ```

(or `./scripts/tensorboard.sh` when using the development container.)

Navigate to `localhost:6006` in a browser to view the output.

In the `app/zoo/pretrained/` folder there is a pre-trained `<game>/best_model.zip` for each game, that can be copied up a directory (e.g. to `app/zoo/frouge/best_model.zip`) if you want to test playing against a pre-trained agent right away.

---
<!-- CUSTOM ENVIRONMENTS -->
### Custom Environments

You can add a new environment by copying and editing an existing environment in the `app/environments/` folder.

For the environment to work with the GrosBill framework, the class must extend the `GBEnv` class in `app/utils/env.py` — it defines both the game logic side (players, scores, legal-action masking) and the NiceGUI rendering side used by `play.py`. Read carefully the comments inside this class to use the defined class properties and methods. Knowledge of the [gymnasium Env class](https://gymnasium.farama.org/api/env/) is also required.

Then register your environment (and its policy network) in `app/utils/register.py` so that `train.py`, `test.py` and `play.py` can find it by name.

---
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/zorgluf/GrosBill/issues) for a list of proposed features (and known issues).


---
<!-- LICENSE -->
## License

Distributed under the GPL-3.0. See `LICENSE` for more information.


---
<!-- CONTACT -->
## Contact

François Valley [linkedin](https://www.linkedin.com/in/francois-valley-1133716)
