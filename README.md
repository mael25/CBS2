# CBS2

This project is based on [Learning by Cheating](https://github.com/dotchen/LearningByCheating) (LBC) and [Cheating by Segmentation](https://github.com/thomasvanorden/LBS) (CBS).
It follows the work of CBS which replaced the bird's-eye view semantically segmented
image used to train the teacher in LBC.

CBS2 focus on adding modules within the student network architecture (trained with an RGB image)
in order to favor the perception of traffic lights without decreasing the perception of larger objects.
To do this, we test several approaches used for multi-scale perception such as [Pyramid Pooling Module](https://arxiv.org/abs/1612.01105) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144).
The aim is to be able to integrate these modules as seamlessly as possible in the existent architecture
and keeping it end-to-end.

It has also been adapted to work with Carla 0.9.10.1 which allows to evaluate its performance using Carla's scenario runner.
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
  - [Pytorch GradCAM](https://github.com/jacobgil/pytorch-grad-cam)


### Installing Carla

Install Carla in the  desired location:
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz -C carla09101
```
Clone CBS2 repository (branch: cbs2) in desired location:
```bash
git clone git@github.com:mael25/CBS2.git
cd CBS2
git checkout cbs2
```
If not already done, install conda and then create and activate this new virtual environment:
```bash
conda env create -f docs/cbs2.yml
conda activate cbs2
```
To add more packages to the environment:

- **conda**: `conda install <package>`
- **conda-forge**: `conda install -c conda-forge <package>`
- **pip**: `~/anaconda3/envs/cbs2/bin/pip install <package>`

Add the following environmnet variables to `~/.bashrc`:
```bash
export CARLA_ROOT=<your_path>/carla09101
export CBS2_ROOT=<your_path>/CBS2
export LEADERBOARD_ROOT=${CBS2_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${CBS2_ROOT}/scenario_runner
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
```
Verify the setup by launching Carla (with cbs2 virtual environment activated):
```bash
source ~/.bashrc
$CBS2_ROOT/scripts/launch_carla.sh 1 2000
```

### Launching Carla

Make sure to start Carla in another terminal before proceeding to data collection or evaluation.
```bash
$CBS2_ROOT/scripts/launch_carla.sh <num_runners> <port>
```
Note: if there are multiple runners, `<port>` is also the increment between them.

### Data collection
Data collection configuration (settable in `autoagents/collector_agents/config_data_collection.yml`):
- **num_per_flush**: Amount of timestep data per save
- **noise_collect**: Add noise to the steering command of the agent
- **main_data_dir**: Location where the data should be saved
- ...

To start the data collection:
```bash
python -m rails.data_phase1
```
### Training
##### Teacher
```bash
cd cbs0/training
python -m train_birdview --segmentation --dynamic --batch_size=<batch_size>  --dataset_dir=<path_to_data_dir> --log_dir=<path_to_log_dir> --max_epoch=<max_epoch>
```
##### Student
Phase 0: warm-up stage
```bash
cd cbs0/training
python -m train_image_phase0 --log_dir=<path_to_log_dir> --teacher_path=<path_to_teacher_dir/model-XX.th> --dataset_dir=<path_to_data_dir>
```
Phase 1: actual training
```bash
cd cbs0/training
python -m train_image_phase1 --log_dir=<path_to_log_dir> --teacher_path=<path_to_teacher_dir/model-XX.th> --ckpt=<path_to_phase0_student_dir/model-XX.th> --dataset_dir=<path_to_data_dir> --pretrained --max_epoch=<max_epoch>
```

To train a model with PPM add for example `--ppm=1-2-3-6` (where [1,2,3,6] are the desired bin sizes used for the adaptive pooling). For FPN, add `--fpn`. These arguments must be added both for phase 0 and phase 1.
### Evaluation


- To evaluate on Carla Leaderboard:
```bash
python -m evaluate
```
To evaluate a model trained with PPM or FPN, add respectively `--mod=ppm` or `--mod=fpn`. This will change the configuration file used. The default configuration file for each type of model are located in the `results_lead` folder. The results of the evaluation are saved there as well.

- To evaluate on NoCrash:
```bash
python -m evaluate_nocrash --town=<TownXX> --weather <test/train> --resume
```
To visualize in the W&B logs not only the RGB image with vehicle commands overprinted but also a saliency map made using guided back propagation (implementation: modified from [Pytorch GradCAM](https://github.com/jacobgil/pytorch-grad-cam), paper: J. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller, [Striving for Simplicity](https://arxiv.org/abs/1412.6806): The All Convolutional Net”, Computer Vision and Pattern Recognition (CVPR), 2014), it is necessary to comment the code in `cbs2/autoagents/cbs2_agent.py` at `l.191` and uncomment `l.194-l.199`.
