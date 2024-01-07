# IADB (TensorFlow 2)

This repository provides a TensorFlow implementation of IADB based on:

- [Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model (Siggraph 2023)](https://arxiv.org/abs/2305.03486).

> **Abstract** <br>
> We derive a minimalist but powerful deterministic denoising-diffusion model. While denoising diffusion has shown great success in many domains, its underlying theory remains largely inaccessible to non-expert users. Indeed, an understanding of graduate-level concepts such as Langevin dynamics or score matching appears to be required to grasp how it works. We propose an alternative approach that requires no more than undergrad calculus and probability. We consider two densities and observe what happens when random samples from these densities are blended (linearly interpolated). We show that iteratively blending and deblending samples produces random paths between the two densities that converge toward a deterministic mapping. This mapping can be evaluated with a neural network trained to deblend samples. We obtain a model that behaves like deterministic denoising diffusion: it iteratively maps samples from one density (e.g., Gaussian noise) to another (e.g., cat images). However, compared to the state-of-the-art alternative, our model is simpler to derive, simpler to implement, more numerically stable, achieves higher quality results in our experiments, and has interesting connections to computer graphics.

## Install

```bash
$ git clone https://github.com/Nikolai10/Diffusion-TF.git
```

## Usage

```python
import sys
sys.path.append('/content/Diffusion-TF/projects/IADB') # adjust path to your needs
```

```python
from model import IADBModel

# create time-conditional U-Net
network = ...
# create InDI model
model = IADBModel(network=network)
# compile the model
model.compile(optimizer=...),
# train the model
model.fit(...)

# inference
model.sample_iadb(x0, nb_step=...)
```

## Tutorials
### 2D tutorial (toy example) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1EprIq_ZmHjmmDIPqrZylnH8W2ksPAFcG?usp=sharing)

Learn to create mappings between arbitrary densities.

<p align="center">
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/IADB/res/assets/x0.png" width="25%" />
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/IADB/res/doc/figures/2d_1e_128s.gif" width="25%" />
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/IADB/res/assets/x1.png" width="25%" />
</p>

### Oxford Flowers 102 tutorial [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1wy6c6bm192YbU35rJh-NCACUMwsWgKrz?usp=sharing)

Learn to generate beautiful flowers from gaussian noise.

<p align="center">
  <img src="https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/IADB/res/doc/figures/OxfordFlowers102_250e_128s.gif" />
</p>

## Related work

- official PyTorch implementation: https://github.com/tchambon/IADB