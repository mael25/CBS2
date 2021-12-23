# CBS2

This project is based on [Learning by Cheating](https://github.com/dotchen/LearningByCheating) (LBC) and [Cheating by Segmentation](https://github.com/thomasvanorden/LBS) (CBS).
It follows the work of CBS which replaced the bird's-eye view semantically segmented
image used to train the teacher in LBC.

CBS2 focus on adding modules within the student network architecture (trained with an RGB image)
in order to favor the perception of traffic lights without decreasing the perception of larger objects.
To do this, we test several approaches used for multi-scale perception such as [Pyramid Pooling Module](https://arxiv.org/abs/1612.01105) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144).
The aim is to be able to integrate these modules as seamlessly as possible in the existent architecture
and keeping it end-to-end.

It has also been adapted to work with Carla 0.9.10.1 which allows to evaluate its performance on Carla's leaderboard.
The dataset is collected on Carla 0.9.10.1, using the agent implemented in World on Rails, slightly modified so that it brakes
closer to traffic lights and gather all the data needed for the CBS models.

### Code source

This repository contains code from other sources
- Not modified:
  - [Carla Leaderboard](https://github.com/carla-simulator/leaderboard)
  - [Carla Scenario Runner](https://github.com/carla-simulator/scenario_runner)

  (Instructions on how to setup Carla and the leaderboard / scenario runner available below and at https://leaderboard.carla.org/get_started)

- Modified:
  - [World on rails](https://github.com/dotchen/WorldOnRails)
  - [Cheating by Segmentation](https://github.com/thomasvanorden/LBS) (branch: segmentation)


### Installing Carla

##### Install Carla in desired location
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz -C carla09101
```
##### Clone CBS2 repository (branch: cbs2) in desired location
```bash
git clone git@github.com:mael25/CBS2.git
cd CBS2
git checkout cbs2
```
##### If not already done, install conda and then create and activate this new virtual environment
```bash
conda env create -f docs/cbs2.yml
conda activate cbs2
```
##### To add more package to the environment

- conda: ```conda install <package>```
- conda-forge: ```conda install -c conda-forge <package>```
- pip: ```~/anaconda3/envs/cbs2/bin/pip install <package>```


##### Add the following environmnet variables to ~/.bashrc
```bash
export CARLA_ROOT=<your_path>/carla09101
export CBS2_ROOT=<your_path>/CBS2
export LEADERBOARD_ROOT=${CBS2_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${CBS2_ROOT}/scenario_runner
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
```
##### Verify the setup by launching Carla (with cbs2 virtual environmnet activated):
```bash
source ~/.bashrc
$CBS2_ROOT/scripts/launch_carla.sh 1 2000
```
Note: be sure to have cbs2 virtual environmnet activated
### Data collection


### Training


### Evaluation
