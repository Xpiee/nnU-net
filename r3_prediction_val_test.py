from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data_cls import nnUNetSegClsPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import multiprocessing
import argparse

# nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction
# /home/bhatti_uhn/nnUNet_results/Dataset876_UHNMedImg3D/nnUnetSegClsTrainer__nnUNetPlans__3d_fullres
# /home/bhatti_uhn/nnUNet_raw/Dataset876_UHNMedImg3D/imagesTs

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetid", help="set dataset id", type=int)
    parser.add_argument("--overwrite", help="set name of base dataset", type=int)
    parser.add_argument("--prepare_val", help="prepare validation dataset, 0: False", type=int)

    multiprocessing.set_start_method('spawn', force=True)

    args = parser.parse_args()
    datasetID = args.datasetid
    is_overwrite = False if args.overwrite == 0 else True

    predictor = nnUNetSegClsPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(
            nnUNet_results,
            f"Dataset{datasetID}_UHNMedImg3D/nnUnetSegClsTrainer__nnUNetPlans__3d_fullres",
        ),
        use_folds="all",
        checkpoint_name="checkpoint_final.pth",
    )

    # FOR TEST SET!
    predictor.predict_from_files(
        join(nnUNet_raw, f"Dataset{datasetID}_UHNMedImg3D/imagesTs"),
        join(nnUNet_raw, f"Dataset{datasetID}_UHNMedImg3D/imagesTs_3d_fullres"),
        save_probabilities=False,
        overwrite=is_overwrite,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    prepareValidation = False if args.prepare_val == 0 else True

    if prepareValidation:
        # FOR Validation SET!
        valDatasetID = datasetID + 1
        predictor.predict_from_files(
            join(nnUNet_raw, f"Dataset{valDatasetID}_UHNMedImg3DVAL/imagesTr"),
            join(nnUNet_raw, f"Dataset{valDatasetID}_UHNMedImg3DVAL/imagesTr_3d_fullres"),
            save_probabilities=False,
            overwrite=is_overwrite,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )