

# Truly Scale-Equivariant Deep Nets with Fourier Layers



### [[Project Page]](https://ashiq24.github.io/Scale_Equivarinat_Fourier_Layer/) [[Paper (NeurIPS 2023)]](https://arxiv.org/abs/2311.02922) [[Quick Start Demo] ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fKHxYw1QxJ1CWpDFGLdl8Im83GnfAbFC?usp=sharing)

[Md Ashiqur Rahman](https://sites.google.com/view/ashiqurrahman/curriculum-vitae?authuser=0),
[Raymond A. Yeh](https://www.raymond-yeh.com/)

Department of Computer Science, Purdue University

**Generic Model**
![Animation](https://github.com/ashiq24/Scale_Equivarinat_Fourier_Layer/blob/main/vizs/image_feature_animation_6_base.gif)
**Ours**
![Animation](https://raw.githubusercontent.com/ashiq24/Scale_Equivarinat_Fourier_Layer/refs/heads/main/vizs/image_feature_animation_6_ours_.gif)

### 🧭 Resolution-Aware Feature Representation

The proposed layers maintain consistent feature representations across varying input resolutions by dynamically adapting to the spatial scale of the input image. This ensures robustness and stability when processing images of different sizes.


# Overview
This is the official implementation of **"Truly Scale-Equivariant Deep Nets with Fourier Layers"**, accepted at NeurIPS 2023.

We address the challenge of scale-equivariance in vision models by proposing a novel architecture based on **Fourier layers** that achieves **zero equivariance error**. Unlike prior approaches that assume a continuous domain and overlook anti-aliasing, our method formulates downscaling directly in the **discrete domain** with anti-aliasing built in. Evaluated on the MNIST scale and STL-10, our model demonstrates competitive classification performance while ensuring exact scale invariance.


## 🛠️ Setup & Installation

First, clone the repository and install the package in **editable mode**:

```bash
git clone https://github.com/ashiq24/scale_equivariant_fourier_layer.git
cd scale_equivariant_fourier_layer
pip install -e .
```
Make sure you have the following libraries installed:

- PyTorch (any version compatible with your hardware: CUDA or CPU)

- NumPy

- SciPy

⚠️ Note: This package does not have a strict version dependency on PyTorch.
You can use it with PyTorch, which is compatible with your system. 🎉

## 🚀 Getting Started with Truly Scale-Equivariant Fourier Layers
This example demonstrates how to use our localized spectral convolution and scale-equivariant non-linearity + pooling layer, tailored for image inputs at multiple resolutions. 📐📷

```python
from scale_eq.layers.spectral_conv import SpectralConv2dLocalized
from scale_eq.layers.scalq_eq_nonlin import scaleEqNonlinMaxp
import torch
```
### 🧠 Model Setup
🌀 1. Localized Spectral Convolution Layer

```python
seq_conv = SpectralConv2dLocalized(
    in_channel=3,          # Input image has 3 channels (e.g., RGB)
    out_channel=32,        # Output has 32 channels
    global_modes=28,       # Captures global patterns (~ half of input size for speed)
    local_modes=7          # Captures fine-scale features
)
```
🔍 Local Model behaves like a traditional filter (e.g., capturing edges or textures).

🌐 Global Model should be close to the input image resolution (128), but can be smaller to reduce computation.
In this example, we use 28 to save on compute while still learning from global features.

### ⚡ 2. Scale-Equivariant Non-Linearity with Max Pooling

```python
seq_nonlin = scaleEqNonlinMaxp(
    torch.nn.functional.sigmoid,  # Non-linearity to apply (can be ReLU, GELU, etc.)
    base_res=32,                  # Minimum resolution (lowest scale)
    normalization=None,           # Optional: apply normalization
    max_res=128,                  # Maximum resolution (highest scale)
    increment=32,                 # Controls scale skipping (1 = full equivariance, >1 = faster)
    channels=32,                  # Number of feature channels
    pool_window=2                 # Window size for max pooling
)
```
🔁 Scale Equivariance: The same pattern at different scales should yield similar outputs — this module enforces that.

⚖️ Trade-off: increment=1 means no scales skipped (full equivariance, but slow).
Setting it to a higher value (like 32) means some scales are skipped — it's still effective, but faster.

🏊‍♂️ Pooling adds robustness and spatial invariance to features.

### 🧪 Inference Example

The scale equivariant convolution and non-linearities work in the complex Fourier domain. So the input and output are both in terms of the Fourier Coefficient. Let's see these in action: 

``` python
random_input = torch.randn(1, 3, 128, 128).to(torch.float)

with torch.no_grad():
    # Step 1: Forward FFT for frequency domain processing
    conv_out = seq_conv(torch.fft.fft2(random_input, norm='forward'))
    
    # Step 2: Apply scale-equivariant nonlinearity + pooling
    nonlin_out = seq_nonlin(conv_out)
    
    # Step 3: Inverse FFT to bring back to spatial domain
    real_output = torch.fft.ifft2(nonlin_out, norm='forward').real

    print("Input Shape: ", random_input.shape)
    print("Convolved Output Shape: ", real_output.shape)
```

## Demo: How to use the Fourier layers in a model
A notebook containing a demonstration of the scale equivariant layer and its uses in Deep neural networks is available in the notebook ```demo_and_quickstart.ipynb```. The notebook can also be executed on Google Colab by following the link  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fKHxYw1QxJ1CWpDFGLdl8Im83GnfAbFC?usp=sharing)

## Running the Experiments
To regenerate the experiment of the paper,r please install the dependencies in `regen_results_req.txt`
Steps:
- Download the project
- Update the ```model``` flag to select desired model
- Update the ```project_name``` flag to match the Neptune project
- Update the ```data_path``` to the dataset loaction.

Execute the following commands

```bash
python3 train_script GPU_ID
```

```train_script: train_1d.py, train_mnist.py, train_stl.py```

```GPU_ID: int, device id of the GPU to train on```
