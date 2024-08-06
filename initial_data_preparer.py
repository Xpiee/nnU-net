import os
import shutil
import logging
import SimpleITK as sitk
import numpy as np

basePath = "/home/bhatti_uhn"

# set environment variable here.
os.environ["nnUNet_preprocessed"] = f"{basePath}/nnUNet_preprocessed"
os.environ["nnUNet_results"] = f"{basePath}/nnUNet_results"
os.environ["nnUNet_raw"] = f"{basePath}/nnUNet_raw"

# Ensure that environment variables are set correctly
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from nnunetv2.dataset_conversion import generate_dataset_json
from nnunetv2.run.run_training import run_training_entry

logging.basicConfig(level=logging.WARNING)

class InitialDataPreparer:
    
    def __init__(self, base_dir, nnunet_base, train_dir, test_dir):
        self.base_dir = base_dir
        self.nnunet_base = nnunet_base
        self.images_tr_dir = os.path.join(nnunet_base, 'imagesTr')
        self.labels_tr_dir = os.path.join(nnunet_base, 'labelsTr')
        self.images_ts_dir = os.path.join(nnunet_base, 'imagesTs')
        self.trainSrc = os.path.join(self.base_dir, train_dir)
        self.testSrc = os.path.join(self.base_dir, test_dir)
        self.train_subdirs = sorted([sdir for sdir in os.listdir(self.trainSrc) if not sdir.startswith('.')])

        os.makedirs(self.images_tr_dir, exist_ok=True)
        os.makedirs(self.labels_tr_dir, exist_ok=True)
        os.makedirs(self.images_ts_dir, exist_ok=True)
        
        self.channel_names = {0: "CT"}
        self.labels = {'background': 0, 'pancreas': 1, 'lesion': 2}
        self.num_training_cases = len(os.listdir(self.images_tr_dir))
        self.file_ending = '.nii.gz'

    def copy_files_to_raw(self):
        for subdir in self.train_subdirs:
            print(f"Processing {subdir}...")
            train_subdir = os.path.join(self.trainSrc, subdir)
            files = [s for s in sorted(os.listdir(train_subdir)) if not s.startswith('.')]
            for fileName in files:
                if len(fileName.split('_')) == 4:
                    tr_copyPath = os.path.join(train_subdir, fileName)
                    logging.info(f"Copying {tr_copyPath} to {self.images_tr_dir}")
                    shutil.copy(tr_copyPath, self.images_tr_dir)
                elif len(fileName.split('_')) == 3:
                    shutil.copy(os.path.join(train_subdir, fileName), self.labels_tr_dir)
                else: 
                    raise ValueError(f"File {fileName} does not match the expected format.")

    def copy_testFiles_to_raw(self):
        files = [s for s in sorted(os.listdir(self.testSrc)) if not s.startswith('.')]
        for fileName in files:
            shutil.copy(os.path.join(self.testSrc, fileName), self.images_ts_dir)

    @staticmethod
    def correct_labels(image):
        array = sitk.GetArrayFromImage(image)
        array = np.int64(array)  # Correcting the label values to int64
        corrected_image = sitk.GetImageFromArray(array)
        corrected_image.CopyInformation(image)
        return corrected_image

    def correct_all_type_labels(self):
        labelImagesInDir = [sdir for sdir in sorted(os.listdir(self.labels_tr_dir)) if not sdir.startswith('.')]
        for labelImg in labelImagesInDir:
            imagePath = os.path.join(self.labels_tr_dir, labelImg)
            image = sitk.ReadImage(imagePath)
            corrected_image = self.correct_labels(image)
            sitk.WriteImage(corrected_image, imagePath)

    @staticmethod
    def correct_spacing(image_path, seg_path):
        image = sitk.ReadImage(image_path)
        seg = sitk.ReadImage(seg_path)
        image_spacing = image.GetSpacing()
        seg_spacing = seg.GetSpacing()
        if not np.allclose(image_spacing, seg_spacing, atol=1e-7):
            print(f"Correcting spacing for {seg_path}")
            seg.SetSpacing(image_spacing)
            sitk.WriteImage(seg, seg_path)
            print(f"Corrected segmentation saved to {seg_path}")

    def generate_dataset_json_from_data(self):
        generate_dataset_json.generate_dataset_json(self.nnunet_base, self.channel_names, self.labels, self.num_training_cases, self.file_ending)
    
    def fix_spacing_issue_in_segmentation_files(self):
        imageFilesInDir = [sdir for sdir in sorted(os.listdir(self.images_tr_dir)) if not sdir.startswith('.')]
        for image in imageFilesInDir:
            imgPath = os.path.join(self.images_tr_dir, image)
            segPath = os.path.join(self.labels_tr_dir, image.replace('_0000', ''))
            self.correct_spacing(imgPath, segPath)

    def run_data_preparer(self):
        self.copy_files_to_raw()

        if self.trainSrc != 'validation':
            self.copy_testFiles_to_raw()

        self.correct_all_type_labels()
        self.generate_dataset_json_from_data()
        self.fix_spacing_issue_in_segmentation_files()

if __name__ == "__main__":
    base_dir = '/home/bhatti_uhn/Dataset/UHN-MedImg3D-ML-quiz'
    nnunet_base = '/home/bhatti_uhn/nnUNet_raw/Dataset876_UHNMedImg3D'
    
    trainDir = 'train'
    testDir = 'test'
    idp = InitialDataPreparer(base_dir, nnunet_base, trainDir, testDir)
    idp.run_data_preparer()

    ## for validation set
    nnunet_base_val = '/home/bhatti_uhn/nnUNet_raw/Dataset877_UHNMedImg3DVAL'
    idp_val = InitialDataPreparer(base_dir, nnunet_base_val, 'validation', 'test')
    idp_val.run_data_preparer()

    ### Run integrity check on the dataset and prepare preprocessed dataset.
    # !nnUNetv2_plan_and_preprocess -d 876 --verify_dataset_integrity