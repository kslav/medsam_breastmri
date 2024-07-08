# MedSAM for segmenting breast lesions on MRI

PURPOSE: To fine-tune the MedSAM foundation model by Ma et al. (2024) *Communications* on breast MRI data for segmenting breast lesions.

MOTIVATION: Segmentation remains an important task in medical image analysis for identifying structures of interest, such as cancers, for downstream analysis tasks. Breast cancers are notoriously difficult to segment due to varying enhancement patterns of lesions on dynamic contrast-enhanced MRI and occlusion by dense tissues. Here we aim to evaluate the ability of the MedSAM foundation model to segment breast cancers on MRI when fine-tuned on a breast MRI dataset. 

<img src="assets/Figure1.png" width="800">

Currently, MedSAM (and SAM) requires a bounding box that identifies the region in which the object of interest lies. This prompt generation, however, still necessitate the involvement of a radiologist to identify breast lesions and delinate bounding boxes for each slice in an imaging volume. To circumvent this step, we are investigating whether MedSAM can be fine-tuned on breast MRI data with a fixed bounding box that includes the entire breast tissue. 

PRELIMINARY RESULTS:

<img src="assets/Figure2.png" width="800">


*This is a work in progress and is continuously updated*
TODO
- Implement ddp to investigate larger batch sizes 
- Explore additional methods for data preprocessing of MRIs outside of those discussed by Ma et al. 
- External validation 

References:
1. Ma et al. (2024). *Communications*. https://doi.org/10.1038/s41467-024-44824-z
2. Kirillov et al. (2023). *arXiv*. https://arxiv.org/abs/2304.02643
