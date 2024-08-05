from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data_cls import nnUNetSegClsPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import multiprocessing

# nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction
# /home/bhatti_uhn/nnUNet_results/Dataset876_UHNMedImg3D/nnUnetSegClsTrainer__nnUNetPlans__3d_fullres
# /home/bhatti_uhn/nnUNet_raw/Dataset876_UHNMedImg3D/imagesTs
# instantiate the nnUNetPredictor

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)

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
            "Dataset876_UHNMedImg3D/nnUnetSegClsTrainer__nnUNetPlans__3d_fullres",
        ),
        use_folds="all",
        checkpoint_name="checkpoint_final.pth",
    )
    # variant 1: give input and output folders

    # FOR TEST SET!
    # predictor.predict_from_files(
    #     join(nnUNet_raw, "Dataset876_UHNMedImg3D/imagesTs"),
    #     join(nnUNet_raw, "Dataset876_UHNMedImg3D/imagesTs_3d_fullres"),
    #     save_probabilities=False,
    #     overwrite=False,
    #     num_processes_preprocessing=2,
    #     num_processes_segmentation_export=2,
    #     folder_with_segs_from_prev_stage=None,
    #     num_parts=1,
    #     part_id=0,
    # )


    # FOR Validation SET!
    predictor.predict_from_files(
        join(nnUNet_raw, "Dataset877_UHNMedImg3DVAL/imagesTr"),
        join(nnUNet_raw, "Dataset877_UHNMedImg3DVAL/imagesTr_3d_fullres"),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )



# if __name__ == "__main__":
#     from nnunetv2.paths import nnUNet_results, nnUNet_raw
#     import torch
#     from batchgenerators.utilities.file_and_folder_operations import join
#     from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
#     from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

#     # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

#     # instantiate the nnUNetPredictor
#     predictor = nnUNetPredictor(
#         tile_step_size=0.5,
#         use_gaussian=True,
#         use_mirroring=True,
#         perform_everything_on_device=True,
#         device=torch.device("cuda", 0),
#         verbose=False,
#         verbose_preprocessing=False,
#         allow_tqdm=True,
#     )
#     # initializes the network architecture, loads the checkpoint
#     predictor.initialize_from_trained_model_folder(
#         join(
#             nnUNet_results,
#             "Dataset876_UHNMedImg3D/nnUnetSegClsTrainer__nnUNetPlans__3d_fullres",
#         ),
#         use_folds="all",
#         checkpoint_name="checkpoint_final.pth",
#     )
#     # variant 1: give input and output folders
#     predictor.predict_from_files(
#         join(nnUNet_raw, "Dataset876_UHNMedImg3D/imagesTs"),
#         join(nnUNet_raw, "Dataset876_UHNMedImg3D/imagesTs_3d_fullres"),
#         save_probabilities=False,
#         overwrite=False,
#         num_processes_preprocessing=2,
#         num_processes_segmentation_export=2,
#         folder_with_segs_from_prev_stage=None,
#         num_parts=1,
#         part_id=0,
#     )
