import os
import sys
import shutil
import json
import multiprocessing

# set environment variable here.
os.environ["nnUNet_preprocessed"] = "/home/bhatti_uhn/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/home/bhatti_uhn/nnUNet_results"
os.environ["nnUNet_raw"] = "/home/bhatti_uhn/nnUNet_raw"

# Ensure that environment variables are set correctly # from run_training.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import nnunetv2
from nnunetv2.dataset_conversion import generate_dataset_json
from nnunetv2.run.run_training import run_training_entry
import logging
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

logging.basicConfig(level=logging.WARNING)
from nnunetv2.training.nnUNetTrainer.nnUnetSegClsTrainer import nnUnetSegClsTrainer

fold = 'all'

# Define the necessary arguments
args = [
    "script_name",  # This is a placeholder for the script name
    "Dataset876_UHNMedImg3D",  # dataset_name_or_id
    "3d_fullres",  # configuration
    f"{fold}",  # fold
    '-tr', 'nnUnetSegClsTrainer',  # optional: trainer_class_name
    # '-p', 'nnUNetPlans',  # optional: plans_identifier
]

# Set sys.argv to the list of arguments
sys.argv = args

if __name__ == '__main__':
    # This is necessary for Windows and macOS to prevent the 'RuntimeError: An attempt has been made to start a \
    # new process before the current process has finished its bootstrapping phase.'
    multiprocessing.set_start_method('spawn', force=True)
    run_training_entry()

    # TODO: Add the code to run the inference here. Done!
    # TODO: add a bit more complex decoder for classification task. Done. Debug necessary.
    # TODO: Start looking at the reporting stuff. Done
    # What needs to be reported and get prediction results from validation set and test set provided in the Quiz. Done
    # TODO: List down all the changes and additions that I have made to the nnU-Net codebase. Important.