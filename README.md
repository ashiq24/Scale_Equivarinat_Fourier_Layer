

# Truly Scale-Equivariant Deep Nets with Fourier Layers



### [[Project Page]](https://ashiq24.github.io/Scale_Equivarinat_Fourier_Layer/) [[Paper (NeurIPS 2023)]](https://arxiv.org/abs/2311.02922) [[Quick Start Demo] ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fKHxYw1QxJ1CWpDFGLdl8Im83GnfAbFC?usp=sharing)

[Md Ashiqur Rahman](https://sites.google.com/view/ashiqurrahman/curriculum-vitae?authuser=0),
[Raymond A. Yeh](https://www.raymond-yeh.com/)

Department of Computer Science, Purdue University

**Generic Model**
![Animation](https://github.com/ashiq24/Scale_Equivarinat_Fourier_Layer/blob/main/vizs/image_feature_animation_6_base.gif)
**Ours**
![Animation](https://raw.githubusercontent.com/ashiq24/Scale_Equivarinat_Fourier_Layer/refs/heads/main/vizs/image_feature_animation_6_ours_.gif)

**Our features remain consistent across varying input resolutions, dynamically adapting their spatial resolution to match the input image.**

# Overview
This is the official implementation of "Truly Scale-Equivariant Deep Nets with Fourier Layers" accepted at NeurIPS 2023.

In computer vision, models must be able to adapt to changes in image resolution to effectively carry out tasks such as image segmentation; This is known as scale-equivariance. Recent works have made progress in developing scale-equivariant convolutional neural networks, e.g., through weight-sharing and kernel resizing. However, these networks are not truly scale-equivariant in practice. Specifically, they do not consider anti-aliasing as they formulate the down-scaling operation in the continuous domain. To address this shortcoming, we directly formulate down-scaling in the discrete domain with consideration of anti-aliasing. We then propose a novel architecture based on Fourier layers to achieve truly scale-equivariant deep nets, i.e., absolute zero equivariance-error. Following prior works, we test this model on MNIST-scale and STL-10 datasets. Our proposed model achieves competitive classification performance while maintaining zero equivariance-error.

## Setup Dependencies
For a full list of requirements, please refer to ***Scale_Equivarinat_Fourier_Layer/requirements.txt***. To install the dependencies, please execute:

```bash
pip install -e .
```


## Demo
A notebook containing a demonstration of the scale equivariant layer and their uses in Deep neural networks is available in the notebook ```demo_and_quickstart.ipynb```. The notebook can also be executed on Google Colab by following the link  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fKHxYw1QxJ1CWpDFGLdl8Im83GnfAbFC?usp=sharing)

## Using the model
The scale equivariant convolution and non-linearities works in complex Fourier domain. So the input and output both in terms of Fourier Co-efficient.




## Running the Models

Steps:
- Download the project
- Update the ```model``` flag to select desired model
- Update the ```project_name``` flag to match the Neptune project
- Update the ```data_path``` to the dataset loaction.

The execute the following commands

```bash
python3 train_script GPU_ID
```

```train_script: train_1d.py, train_mnist.py, train_stl.py```

```GPU_ID: int, device id of the GPU to train on```
