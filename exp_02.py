import os
import sys
import shutil
import json

# set environment variable here.
os.environ["nnUNet_preprocessed"] = "/home/bhatti_uhn/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/home/bhatti_uhn/nnUNet_results"
os.environ["nnUNet_raw"] = "/home/bhatti_uhn/nnUNet_raw"

# Ensure that environment variables are set correctly # from run_training.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from nnunetv2.dataset_conversion import generate_dataset_json
from nnunetv2.run.run_training import run_training_entry
import logging

logging.basicConfig(level=logging.WARNING)

fold = 0

# Define the necessary arguments
args = [
    "script_name",  # This is a placeholder for the script name
    "Dataset876_UHNMedImg3D",  # dataset_name_or_id
    "3d_fullres",  # configuration
    f"{fold}",  # fold
    '-tr', 'nnUNetSegClsTrainer',  # optional: trainer_class_name
    # '-p', 'nnUNetPlans',  # optional: plans_identifier
]

# Set sys.argv to the list of arguments
sys.argv = args

# Run the training entry function
run_training_entry()