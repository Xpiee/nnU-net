# Enhanced nnUNet Framework with Classification

## Overview
This repository includes modifications to the nnUNet framework to support classification tasks alongside segmentation. The enhancements involve integrating classification capabilities at various stages of the nnUNet pipeline. 

## Key Changes
### 1. Added Classification Support
- Integrated classification functionality into the nnUNet framework.
- Enabled the framework to handle both segmentation and classification tasks.

### 2. New Architectures Implemented
- **UNet Encoder-Decoder with Simple Classification Head**: This implementation utilizes the UNet encoder-decoder architecture, adding a straightforward classification head for classification tasks.
- **UNet Encoder-Decoder with Attention Mechanisms**: Implemented a model combining the UNet encoder-decoder with self-attention and cross-attention layers for enhanced classification performance.

## Implementation Details
### 1. Encoder-Decoder with Classification Head
- **Encoder**: Utilizes the standard UNet encoder to extract features.
- **Decoder**: Standard UNet decoder for segmentation tasks.
- **Classification Head**: A simple, fully connected layer added to the end of the encoder for classification tasks.

### 2. Encoder-Decoder with Attention Mechanisms [WIP]
- **Self-Attention**: Applied self-attention layers to the encoder outputs to capture dependencies within the feature maps.
- **Cross-Attention**: Employed cross-attention layers between the memory vector and encoder outputs to enhance classification.
- **Memory Vector**: A fixed dimension memory vector (200x320) used to store global context for classification.


## Contact
For any questions or further discussion, feel free to contact me.

# Original nnUNet Framework 
Click [here](https://github.com/MIC-DKFZ/nnUNet) and [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the v2 and v1 one instead. Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended.

## What is nnU-Net?
nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It analyzes the provided training cases and automatically configures a matching U-Net-based segmentation pipeline. No expertise required on your end! You can simply train the models and use them for your application.

**nnU-Net has been evaluated on 23 datasets from biomedical competitions and has scored several first places on open leaderboards. It continues to be a baseline and method development framework in the community.**

Please cite the [following paper](https://www.nature.com/articles/s41592-020-01008-z) when using nnU-Net:
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

---

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
