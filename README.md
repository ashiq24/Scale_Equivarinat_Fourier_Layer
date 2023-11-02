# Truly Scale-Equivariant Deep Nets with Fourier Layers

### [[Project Page]](TBA) [[Paper (NeurIPS 2023)]](https://neurips.cc/virtual/2023/poster/71980)

[Md Ashiqur Rahman](https://sites.google.com/view/ashiqurrahman/curriculum-vitae?authuser=0),
[Raymond A. Yeh](https://www.raymond-yeh.com/)
Purdue University

<p align="center">
<img src='https://raymondyeh07.github.io/learnable_polyphase_sampling/resource/pipeline.png' width=800>
</p>


# Overview
This is the official implementation of "Truly Scale-Equivariant Deep Nets with Fourier Layers" accepted at NeurIPS 2023.

## Setup Dependencies
For a full list of requirements, please refer to ***Scale_Equivarinat_Fourier_Layer/requirements.txt***. To install the dependencies, please execute:

```bash
pip install -r requirements.txt
```


## Demo
For notebook demonstrations of our proposed LPS (LPD and LPU) layers, please refer to the ***demo*** directory.

To run the notebook, please execute:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge matplotlib
jupyter-notebook demo
```

## Results and Pre-trained models

#### Experiments Setup for Image Classification (ImageNet)

- Download the ILSVRC2012 dataset from its [official repository](https://image-net.org/challenges/LSVRC/2012/), uncompress it into the dataset folder (e.g. `/learn_poly_sampling/datasets/ILSVRC2012`) and split it into train and val partitions using [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

- Classification accuracy or shift consistency can be computed by setting the `--eval_mode` flag as either `class_accuracy` or `shift_consistency`, respectively.

#### Results for Image Classification (ImageNet)
To reproduce our results in Tab. 2 \& 3, run the scripts included in ```learn_poly_sampling/scripts``` with our pre-trained models.

### [Pre-trained Classification Models](https://uofi.box.com/s/pql7u3c7x8zifp0m46xhe2uduwwazcad)
Please refer to the link above to download all our pre-trained classification models. Note that our evaluation scripts assume the checkpoints are stored at ```learn_poly_sampling/checkpoints/classification```.