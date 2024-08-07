import os
import shutil
import logging
import SimpleITK as sitk
import numpy as np
import argparse

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
        self.num_training_cases = len(os.listdir(self.images_tr_dir))
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", help="set base path to dataset", type=str)
    parser.add_argument("--datasetid", help="set dataset id", type=int)
    parser.add_argument("--basedataset", help="set name of base dataset", type=str)
    parser.add_argument("--prepare_val", help="prepare validation dataset, 0: False", type=int)

    args = parser.parse_args()
    # basePath = "/home/bhatti_uhn"
    basePath = args.basepath
    datasetID = args.datasetid
    baseDataset = args.basedataset

    assert datasetID > 800, "Dataset ID should be greater than 800. Just to be safe. I used 876 for the experiments."

    # set environment variable here.
    os.environ["nnUNet_preprocessed"] = f"{basePath}/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = f"{basePath}/nnUNet_results"
    os.environ["nnUNet_raw"] = f"{basePath}/nnUNet_raw"

    # Ensure that environment variables are set correctly
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from nnunetv2.dataset_conversion import generate_dataset_json
    # base_dir = f'{basePath}/Dataset/UHN-MedImg3D-ML-quiz'
    base_dir = f'{basePath}/Dataset/{baseDataset}'

    # check if the dataset is present in base_dir
    assert os.path.exists(base_dir), f"Dataset {baseDataset} not found in {base_dir}. Please check the dataset path."

    nnunet_base = f'{basePath}/nnUNet_raw/Dataset{datasetID}_UHNMedImg3D'
    
    trainDir = 'train'
    testDir = 'test'

    idp = InitialDataPreparer(base_dir, nnunet_base, trainDir, testDir)
    idp.run_data_preparer()

    prepareValidation = False if args.prepare_val == 0 else True
    
    # not mixing validation and train/test data
    valDatasetID = datasetID + 1
    if prepareValidation:
        ## for validation set
        nnunet_base_val = f'{basePath}/nnUNet_raw/Dataset{valDatasetID}_UHNMedImg3DVAL'
        idp_val = InitialDataPreparer(base_dir, nnunet_base_val, 'validation', testDir)
        idp_val.run_data_preparer()

    ## use r1_dataprep.sh to run the whole script with verifying dataset integrity