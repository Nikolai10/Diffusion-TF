# InDI (TensorFlow 2)

This repository provides a TensorFlow implementation of InDI based on:

- [Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration (TMLR 2023)](https://arxiv.org/abs/2303.11435).

> **Abstract** <br>
> Inversion by Direct Iteration (InDI) is a new formulation for supervised image restoration that avoids the so-called ``regression to the mean'' effect and produces more realistic and detailed images than existing regression-based methods. It does this by gradually improving image quality in small steps, similar to generative denoising diffusion models. Image restoration is an ill-posed problem where multiple high-quality images are plausible reconstructions of a given low-quality input. Therefore, the outcome of a single step regression model is typically an aggregate of all possible explanations, therefore lacking details and realism. The main advantage of InDI is that it does not try to predict the clean target image in a single step but instead gradually improves the image in small steps, resulting in better perceptual quality. While generative denoising diffusion models also work in small steps, our formulation is distinct in that it does not require knowledge of any analytic form of the degradation process. Instead, we directly learn an iterative restoration process from low-quality and high-quality paired examples. InDI can be applied to virtually any image degradation, given paired training data. In conditional denoising diffusion image restoration the denoising network generates the restored image by repeatedly denoising an initial image of pure noise, conditioned on the degraded input. Contrary to conditional denoising formulations, InDI directly proceeds by iteratively restoring the input low-quality image, producing high-quality results on a variety of image restoration tasks, including motion and out-of-focus deblurring, super-resolution, compression artifact removal, and denoising.

## Install

```bash
$ git clone https://github.com/Nikolai10/Diffusion-TF.git 
```

## Usage

```python
import sys
sys.path.append('/content/Diffusion-TF/projects/InDI') # adjust path to your needs
```

```python
# requires TFP https://www.tensorflow.org/probability
from model import InDIModel

# create time-conditional U-Net
network = ...
# create InDI model
model = InDIModel(network=network, norm="L1", p_key="linear_0", eps=0.01, brownian_motion=False)
# compile the model
model.compile(optimizer=...),
# train the model
model.fit(...)

# inference
model.sample_InDI(y, nb_step=...)
```

## Tutorials
### 2D tutorial (toy example) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1POqMX1VTz_BSs4426Y6iXSx6tmw1WSUi?usp=sharing)

Learn to train InDI to overcome the "regression to the mean effect" for 2d restoration.

<p align="center">
   <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI/res/doc/figures/2d.gif" width="50%" />
  </p>

### Compression artifact removal tutorial [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1rim3c1Evf0Q2dJpFsdLYogrFqAjIh2dE?usp=sharing)

Learn to train InDI for JPEG compression artifact removal.

<p align="center">
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI/res/doc/figures/tree_ref.png" width="45%" />
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI/res/doc/figures/tree_1.png" width="45%" />
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI/res/doc/figures/tree_10.png" width="45%" />
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI/res/doc/figures/tree_100.png" width="45%" />
</p>

## Related work

- official implementation: n/a