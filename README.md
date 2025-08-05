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

Build standalone docker play images :
  ```sh
  docker build . -t simple-play-frouge -f Dockerfile_play_frouge
  ```
or launch command directly from the *app* directory.
---
<!-- TUTORIAL -->
## Tutorial

This is a quick tutorial to allow you to start using the two entrypoints into the codebase: `test.py` and `train.py`.

---
<!-- QUICKSTART -->
### Quickstart

#### `play.py` 

This entrypoint allows you to play against a trained AI, pit AIs against eachother or play against baseline random models. This will launch a web server as a GUI interface on port 8080.

   ```sh
   python3 play 
   ```

#### `train.py` 

This entrypoint allows you to start training the AI using selfplay PPO. The underlying PPO engine is from the [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) package.

For example, you can start training the agent to learn how to play SushiGo with the following command:
   ```sh
   python3 train.py -r -e frouge -t 0.5 -os 5000 -ob 256
   ```
*TODO : explain parameters*

After 30 or 40 iterations the process should have achieved above the default threshold score of 0.2 and will output a new `best_model.zip` to the `/zoo/frouge` folder. 

Training runs until you kill the process manually (e.g. with Ctrl-C), so do that now.

You can now use the `test.py` entrypoint to play 100 games silently between the current `best_model.zip` and the random baselines model as follows:

  ```sh
  python3 test.py -g 100 -a best_model base base base base -e frouge 
  ```

You should see that the best_model scores better than the two baseline model opponents. 
```sh
Played 100 games: {'best_model_btkce': 31.0, 'base_sajsi': -15.5, 'base_poqaj': -15.5}
```

You can continue training the agent by dropping the `-r` reset flag from the `train.py` entrypoint arguments - it will just pick up from where it left off.

   ```sh
   python3 train.py -e frouge -t 250 -os 12800 -ob 256
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

For the environment to work with the GrosBill framework, the class must extends the GBEnv class in `utils/env.py`. Read carefully the comments inside this class to use the defined class properties and methods. The knowledge of the [gymnasium Env class](https://gymnasium.farama.org/api/env/) is also required.

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
