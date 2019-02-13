## Environment Probing Interaction Policies

This repository contains the implementation for [Environment Probing Interaction Policies](https://openreview.net/pdf?id=ryl8-3AcFX).

#### Setup
Follow [instructions](https://rllab.readthedocs.io/en/latest/user/installation.html) to create a conda 
environment *rllab3* with OpenAI Gym and Mujoco v1.31.

#### Usage

**Training**

```
source activate rllab3

# (Optional) Collect initial dataset for the prediction models. 
# Note that this step is optional since the dataset files are provided under EPI/envs/.
cd collect_initial_data
python collect_data_vine_hopper_8d.py
python collect_data_vine_striker.py

# Train Interaction Policy
cd ..
python train_interaction.py HopperInteraction -e 8
python train_interaction.py StrikerInteraction -e 8

# Train Task Policy
python train.py HopperTaskReset --epi_folder data/Exp190211_HopperInteraction_0/ --epi_itr 100 --params 8
python train.py StrikerTaskReset --epi_folder data/Exp180921_StrikerInteraction_3/ --epi_itr 200

#  Train Average Baseline or Oracle
python train.py HopperAvg
python train.py HopperOracle
```

**Evaluation**
```
 python evaluate_hopper_with_policy.py data/Hopper/HopperTaskReset_0730/Exp180730_HopperTaskReset_6 --epi_folder data/Hopper/HopperInteraction_8/Exp180727_HopperInteraction_0/ --epi_itr 200
```
