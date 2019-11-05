This repository allows to run Rail Environment multi agent training with the RLLib Library.

## Installation:

To run scripts of this repository, the deep learning library tensorflow should be installed, along with the following packages:
```sh
pip install gym ray==0.7.0 gin-config opencv-python lz4 psutil
```

To start a training with different parameters, you can create a folder containing a config.gin file (see example in `experiment_configs/config_example/config.gin`.

Then, you can modify the config.gin file path at the end of the `train_experiment.py` file.

The results will be stored inside the folder, and the learning curves can be visualized in 
tensorboard:

```
tensorboard --logdir=/path/to/folder_containing_config_gin_file
```

## Gin config files

In each config.gin files, all the parameters of the `run_experiment` functions have to be specified.
For example, to indicate the number of agents that have to be initialized at the beginning of each simulation, the following line should be added:

```
run_experiment.n_agents = 2
```

If several number of agents have to be explored during the experiment, one can pass the following value to the `n_agents` parameter:

```
run_experiment.n_agents = {"grid_search": [2,5]}
```

which is the way to indicate to the tune library to experiment several values for a parameter.

To reference a class or an object within gin, you should first register it from the `train_experiment.py` script adding the following line:

```
gin.external_configurable(TreeObsForRailEnv)
```

and then a `TreeObsForRailEnv` object can be referenced in the `config.gin` file:

```
run_experiment.obs_builder = {"grid_search": [@TreeObsForRailEnv(), @GlobalObsForRailEnv()]}
TreeObsForRailEnv.max_depth = 2
```

Note that `@TreeObsForRailEnv` references the class, while `@TreeObsForRailEnv()` references instantiates an object of this class.




More documentation on how to use gin-config can be found on the github repository: https://github.com/google/gin-config

## Run an example:
To start a training on a 20X20 map, with different numbers of agents initialized at each episode, on can run the train_experiment.py script:
```
python RLLib_training/train_experiment.py
```
This will load the gin config file in the folder `experiment_configs/config_examples`.

To visualize the result of a training, one can load a training checkpoint and use the policy learned.
This is done in the `render_training_result.py` script. One has to modify the `CHECKPOINT_PATH` at the beginning of this script:

```
CHECKPOINT_PATH = os.path.join(__file_dirname__, 'experiment_configs', 'config_example', 'ppo_policy_two_obs_with_predictions_n_agents_4_map_size_20q58l5_f7',
                               'checkpoint_101', 'checkpoint-101')
```
and load the corresponding gin config file:

```
gin.parse_config_file(os.path.join(__file_dirname__, 'experiment_configs', 'config_example', 'config.gin'))
```


