#!/bin/bash


basepath="/home/bhatti_uhn"
datasetid=878
basedataset="UHN-MedImg3D-ML-quiz"
prepare_val=0

# Prepare the dataset
python -u /home/bhatti_uhn/nnU-net/r1_initial_data_preparer.py --basepath $basepath --datasetid $datasetid --basedataset $basedataset --prepare_val $prepare_val

nnUNetv2_plan_and_preprocess -d $datasetid --verify_dataset_integrity