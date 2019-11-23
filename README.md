# PySC2 Deep RL Agents

<div align="center">
  <a href="https://youtu.be/m2pC9md0ixY" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123695-c69e17dc-104d-11e8-80bd-33726f3f0f12.gif"
         alt="MoveToBeacon"
         width="200" border="1" style="color:white" />
  </a>
  <a href="https://youtu.be/lpOlKfyhIXc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123698-c910b650-104d-11e8-8019-8825187b677f.gif"
         alt="CollectMineralShards"
         width="200" border="1" style="color:white" />
  </a>
  <a href="https://youtu.be/GFRsXx0imHc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123701-cabcb13e-104d-11e8-9aa8-7f1332d3cdb7.gif"
         alt="FindAndDefeatZerglings"
         width="201" border="1" style="color:white" />
  </a>
  <a href="https://youtu.be/-wDhAHkj90A" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36202298-87b14c60-1183-11e8-9b3f-f9bb5c8b1ab7.gif"
         alt="DefeatZerglingsAndBanelings"
         width="202" border="1" style="color:white" />
  </a>
</div>

<!-- <div align="center">
  <a href="https://youtu.be/m2pC9md0ixY" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123695-c69e17dc-104d-11e8-80bd-33726f3f0f12.gif"
         alt="MoveToBeacon"
         width="320" border="10" style="color:white" />
  </a>
  <a href="https://youtu.be/lpOlKfyhIXc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123698-c910b650-104d-11e8-8019-8825187b677f.gif"
         alt="CollectMineralShards"
         width="320" border="10" style="color:white" />
  </a>
</div>

<div align="center">
  <a href="https://youtu.be/GFRsXx0imHc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123701-cabcb13e-104d-11e8-9aa8-7f1332d3cdb7.gif"
         alt="FindAndDefeatZerglings"
         width="320" border="10" style="color:white" />
  </a>
  <a href="https://youtu.be/-wDhAHkj90A" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36202298-87b14c60-1183-11e8-9b3f-f9bb5c8b1ab7.gif"
         alt="DefeatZerglingsAndBanelings"
         width="320" border="10" style="color:white" />
  </a>
</div> -->


This repository implements different Deep Reinforcement Learning Agents for the [pysc2](https://github.com/deepmind/pysc2/) learning environment as described in the [DeepMind StarCraft II paper](https://deepmind.com/documents/110/sc2le.pdf).

We provide implementations for:
- Advantage Actor Critic (A2C) based on A3C [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)
  * Fully Connected Policy
  * Convolutional LSTM Policy               [https://arxiv.org/abs/1506.04214](https://arxiv.org/abs/1506.04214)
- Proximal Policy Optimization (PPO)        [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- FeUdal Networks (FuN)                     [https://arxiv.org/abs/1703.01161](https://arxiv.org/abs/1703.01161)

This repository is part of a student research project which was conducted at the [Autonomous Systems Labs](http://www.ias.informatik.tu-darmstadt.de/), [TU Darmstadt](https://www.tu-darmstadt.de/) by [Daniel Palenicek](https://github.com/danielpalen), [Marcel Hussing](https://github.com/marcelhussing), and [Simon Meister](https://github.com/simonmeister).

**The repository was originally located at [simonmeister/pysc2-rl-agents](https://github.com/simonmeister/pysc2-rl-agents) but has moved to this new location.**

## Content

### FeUdal Networks

### PPO

### A2C

### Reports

We document our results more in-depth in the following reports:
- Daniel Palenicek, Marcel Hussing, Simon Meister (Apr. 2018): [Deep Reinforcement Learning for StarCraft II](reports/1_deep_reinforcement_learning_for_starcraft_ii.pdf)
- Daniel Palenicek, Marcel Hussing (Sep. 2018): [Adapting Feudal Networks for StarCraft II](reports/2_adapting_feudal_networks_for_starcraft_ii.pdf)


<!-- ### Progress
- [x] A2C agent
- [x] FullyConv architecture
- [x] support all spatial screen and minimap observations as well as non-spatial player observations
- [x] support the full action space as described in the DeepMind paper
(predicting all arguments independently)
- [x] support training on all mini games
- [ ] report results for all mini games
- [x] LSTM architecture
- [ ] Multi-GPU training -->


## Results

On the mini games, we report the following results as best mean over score:

| Map                         | FC  | ConvLSTM | PPO | FUN | [DeepMind](https://deepmind.com/documents/110/sc2le.pdf) 
| ---                         | --- | ---      | --- | --- | ---                                                      
| MoveToBeacon                | 26  | 26       | 26  | 26  | 26
| CollectMineralShards        | 97  | 93       | -   | -   | 103
| FindAndDefeatZerglings      | 45  | -        | -   | -   | 45
| DefeatRoaches               | -   | -        | -   | -   | 100
| DefeatZerglingsAndBanelings | 68  | -        | -   | -   | 62
| CollectMineralsAndGas       | -   | -        | -   | -   | 3978
| BuildMarines                | -   | -        | -   | -   | 3

In the following we show plots for the score over episodes.

### FeUdal Networks

### PPO

<img src="https://user-images.githubusercontent.com/29195346/69479708-cd936800-0e00-11ea-8484-3e9e4972efd0.png" width=580>

### A2C

#### Convolutional LSTM 
<img src="https://user-images.githubusercontent.com/29195346/69479603-6d4ff680-0dff-11ea-806e-ef4e9c0946d0.png" width=580>

#### Fully Connected

<img src="https://user-images.githubusercontent.com/29195346/69478636-088f9e80-0df5-11ea-95ce-769bed6be5d8.png">

Note that the DeepMind mean scores are their best individual scores after 100 runs for each
game, where the initial learning rate was randomly sampled for each run.
We use a constant initial learning rate for a much smaller number of runs due to limited hardware.

## Usage

### Hardware Requirements
- for fast training, a GPU is recommended.
We ran each experiment on a single Titan X Pascal (12GB).

### Software Requirements
- Python 3
- pysc2 (tested with v1.2)
- TensorFlow (tested with 1.4.0)
- StarCraft II and mini games (see below or
  [pysc2](https://github.com/deepmind/pysc2/))

### Quick Install Guide
- `pip install numpy tensorflow-gpu pysc2==1.2`
- Install StarCraft II. On Linux, use
[3.16.1](http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip).
- Download the
[mini games](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### Train & run
- run and train: `python run.py my_experiment --map MoveToBeacon`.
- run and evalutate without training: `python run.py my_experiment --map MoveToBeacon --eval`.

You can visualize the agents during training or evaluation with the `--vis` flag.
See `run.py` for all arguments.

Summaries are written to `out/summary/<experiment_name>`
and model checkpoints are written to `out/models/<experiment_name>`.

### License

This project is licensed under the MIT License (refer to the LICENSE file for details).

## Acknowledgments
The code in `rl/environment.py` is based on [OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/a2c), with adaptions from [sc2aibot](https://github.com/pekaalto/sc2aibot). Some of the code in `rl/agents/a2c/runner.py` is loosely based on [sc2aibot](https://github.com/pekaalto/sc2aibot). The Convolutional LSTM Cell implementation is taken from [carlthome]( https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py).
