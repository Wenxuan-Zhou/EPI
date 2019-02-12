# EPI
Code for Environment Probing Interaction Policies

## Installation
rllab
mujoco

## Run

**Train**
* (Optional) Collect initial dataset

These commands collect the initial dataset for the prediction models. 
It will create csv files under EPI/interaction_policy/models/hopper_8 or striker_2.
Note that this step is optional since the dataset files are provided in this repository.

```
cd collect_initial_data
python collect_data_vine_hopper_8d.py
python collect_data_vine_striker.py
```

* Train Interaction Policy

```
python train_interaction.py HopperInteraction -e 8
python train_interaction.py StrikerInteraction -e 8
```

* Train Task Policy
```
python train.py HopperTaskReset --epi_folder data/Exp180923_HopperInteraction_0/ --epi_itr 200 --params 8
python train.py StrikerTaskReset --epi_folder data/Exp180921_StrikerInteraction_3/ --epi_itr 200
```

* Train Average Baseline or Oracle
```
python train.py HopperAvg
python train.py HopperOracle
```

**Evaluate**
```
 python evaluate_hopper_with_policy.py data/Hopper/HopperTaskReset_0730/Exp180730_HopperTaskReset_6 --epi_folder data/Hopper/HopperInteraction_8/Exp180727_HopperInteraction_0/ --epi_itr 200
```